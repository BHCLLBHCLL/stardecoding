# -*- coding: utf-8 -*-
"""P 波 P11：运动谱系（Motion Lineage）—— 刚体运动 + 滑移 interface、morphing 网格
变形、DFBI 6DOF 刚体动力学、overset 重叠网格插值、MRF 旋转参考系源。

纯 numpy 实现（occ / scdm 两环境皆可用），建立在 P4 有限体积核心
（`fvm_core.FVM`）之上。该谱系对应 STAR-CCM+ 的 Motion（Rigid Body /
Sliding / Morphing / DFBI 6DOF / Overset Mesh / Rotating Reference Frame）
谱系——原 P9 条目因燃烧算例顺延至本 P11 实现。

五大子能力：

  1) 刚体运动 + 滑移 interface：`rotate_points` / `rigid_transform`（Rodrigues
     旋转 + 平移）对网格节点施加给定转速/平动速度的刚体位移；`sliding_interface`
     按运动方向/位置标识滑移面（滑移网格滑动面两侧节点相对运动）。
  2) morphing：`morph_mesh` 将边界面位移（拉普拉斯松弛）传播至内部节点，得到
     光滑变形网格（边界保形、内部松弛）。
  3) DFBI 6DOF：`DfbiBody` 刚体动力学（质量/惯量、位置/姿态、流体力/力矩 →
     线/角速度 → 位置/姿态积分），实现流固耦合双向回馈。
  4) overset：`overset_interpolate`（donor/receptor 反权重插值）在重叠网格间
     传递场量，`overset_donor_weights` 给出最近 donor 重心权重。
  5) MRF 旋转源：`mrf_source` 给出旋转参考系的离心 + 科氏力体源（逐单元力密度，
     注入动量方程），默认仅在 omega≠0 时非零。

求解器 `MotionSolver` 提供 P10 后端兼容门面 `_initialize_field() / step() /
residual() / monitor_payload()`，并暴露 `mrf_source()/mesh_displacement()/
body_kinematics()`；工厂 `make_motion` 支持别名（motion/mrf/rotating/rigid/
sliding/morph/morphing/dfbi/6dof/overset 等，大小写/连字符不敏感）。

纯 numpy 约束：所有范数走 `solver_run._safe_norm`，不新增 scipy 依赖；旋转轴
非零时归一化，主轴旋转保持参考线不动。
"""
import numpy as np

from fvm_core import FVM
from solver_run import _safe_norm


# ---------------------------------------------------------------------------
# 运动谱系常量
# ---------------------------------------------------------------------------
DEFAULT_ROTATION_SPEED = 0.0          # 转速 rad/s
DEFAULT_ROTATION_AXIS = (0.0, 0.0, 1.0)  # 旋转轴（单位化）
DEFAULT_TRANSLATION_VELOCITY = (0.0, 0.0, 0.0)  # 平移速度 m/s
DEFAULT_REFERENCE_POINT = (0.0, 0.0, 0.0)       # 旋转/平移参考点
MAX_MORPH_ITER = 300                   # morphing 拉普拉斯松弛迭代上限
DEFAULT_MORPH_RELAX = 0.6              # morphing 松弛因子
MORPH_TOL = 1.0e-8                    # morphing 收敛容差
DEFAULT_ROTATION_DT = 1.0e-3          # 刚体/DFBI 时间推进默认步长
_MAX_INERTIA_EIGFRAC = 1.0e-6         # 惯量对角最小占比（避免奇异）
_EPS = 1.0e-30


# ---------------------------------------------------------------------------
# 刚体运动原语
# ---------------------------------------------------------------------------
def _unit(v):
    """把向量归一化；零向量返回单位零轴（避免除零）。"""
    v = np.asarray(v, float).ravel()
    n = _safe_norm(v)
    if n <= _EPS:
        return np.zeros(3, float)
    return v / n


def rotate_points(points, axis, angle):
    """对一组点施加绕单位方向 `axis`、转角 `angle`（rad）的 Rodrigues 旋转。

    `points` (N,3) 或 (3,)；返回同形状旋转后坐标。绕轴旋转保持轴上点不动。
    """
    p = np.asarray(points, float)
    single = (p.ndim == 1)
    if single:
        p = p[None, :]
    ax = _unit(axis)
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    # v_rot = v c + (k × v) s + k (k·v)(1 - c)
    cross = np.cross(np.broadcast_to(ax, p.shape), p)
    dot = p @ ax
    out = p * c + cross * s + np.outer(dot, ax) * (1.0 - c)
    if single:
        return out[0]
    return out


def rigid_transform(vertices, rotation_axis=None, rotation_speed=None,
                    translation_velocity=None, reference_point=None, t=0.0):
    """对网格节点施加给定转速 + 平动速度的刚体位移（从 0 时刻起累计）。

    - `rotation_axis`/`rotation_speed`：绕参考点的刚体旋转（angle = speed * t）。
    - `translation_velocity`：整体平移（位移 = velocity * t）。
    - 返回 (N,3) 位移后的节点坐标（顶点数组保持结构，仅坐标平移/旋转）。

    转速与平动同时给定时先平移至参考系再旋转——保证旋转中心处位移为零。
    """
    V = np.asarray(vertices, float)
    ref = np.asarray(reference_point if reference_point is not None
                     else DEFAULT_REFERENCE_POINT, float)
    axis = rotation_axis if rotation_axis is not None else DEFAULT_ROTATION_AXIS
    speed = float(rotation_speed if rotation_speed is not None
                  else DEFAULT_ROTATION_SPEED)
    tvel = np.asarray(translation_velocity
                      if translation_velocity is not None
                      else DEFAULT_TRANSLATION_VELOCITY, float)
    t = float(t)
    # 先平移到参考系原点
    rel = V - ref
    # 旋转：绕轴 angle = speed * t
    if abs(speed) > 1.0e-14:
        rel = rotate_points(rel, axis, speed * t)
    # 平移回参考系 + 平动位移
    return rel + ref + tvel * t


def sliding_interface(fv, motion_axis=None, sign=0.0, tol=1.0e-9,
                      reference_point=None):
    """标识滑移面（sliding interface）上的边界面索引。

    滑移 interface 是同一几何区域上发生相对切向运动的两个共形面；在结构化网格里
    可用「到运动轴有向距离接近给定值」来标识。返回边界面索引数组（size = 命中的
    边界面数），供后续在两侧施加切向/相对运动速度。`sign`＝沿 `motion_axis` 的
    坐标值（参考面位置），命中面为该值 ±tol 内的边界面。
    """
    fv = _as_fv(fv)
    ax = _unit(motion_axis if motion_axis is not None
               else DEFAULT_ROTATION_AXIS)
    ref = np.asarray(reference_point if reference_point is not None
                     else DEFAULT_REFERENCE_POINT, float)
    fc = fv.face_centroid
    coord = (fc - ref) @ ax
    mask = np.abs(coord - float(sign)) <= float(tol)
    return np.where(mask & fv.is_boundary)[0]


# ---------------------------------------------------------------------------
# MRF 旋转参考系源（离心 + 科氏）
# ---------------------------------------------------------------------------
def mrf_source(centroids, u, v, w, omega, axis, rho, reference_point=None):
    """旋转参考系的动量源项密度（逐单元力密度，(n_cells,3)）。

    标准 MRF 动量方程引入两项附加体积力：
      S_cent  = -ρ Ω × (Ω × r) = ρ ω² [r_rel - axis (axis·r_rel)]  （离心，径向外）
      S_cor   = -2 ρ Ω × u_rel                                     （科氏）
    其中 Ω = ω·axis，r_rel = x - x_ref。`omega` 为转速 rad/s；`rho` 为逐单元
    密度（标量广播或 (n_cells,)）。omega=0 时返回零。
    """
    c = np.asarray(centroids, float)
    ax = _unit(axis)
    om = float(omega)
    rho = np.asarray(rho, float).ravel()
    if rho.size == 1:
        rho = np.full(c.shape[0], float(rho[0]), float)
    else:
        rho = rho[:c.shape[0]]
    ref = np.asarray(reference_point if reference_point is not None
                     else DEFAULT_REFERENCE_POINT, float)
    if abs(om) <= 1.0e-14:
        return np.zeros((c.shape[0], 3), float)
    rel = c - ref
    # 离心：r_perp = rel - axis(axis·rel)，S_cent = ρ ω² r_perp
    axis_proj = rel @ ax
    r_perp = rel - np.outer(axis_proj, ax)
    cent = rho[:, None] * (om * om) * r_perp
    # 科氏：-2 ρ Ω × u
    uvec = np.column_stack([u, v, w])
    dom = om * ax
    coriolis = -2.0 * rho[:, None] * np.cross(np.broadcast_to(dom, uvec.shape), uvec)
    return cent + coriolis


# ---------------------------------------------------------------------------
# morphing 网格变形（边界驱动 + 拉普拉斯松弛）
# ---------------------------------------------------------------------------
def _build_vertex_adjacency(fv):
    """由面拓扑构建顶点邻接表（list[set]，供 Laplacian 松弛）。

    每个三角形面贡献其三条边的两端点互邻；边界面顶点记录在 `boundary_vertices`。
    """
    fv = _as_fv(fv)
    n_v = fv.n_vertices
    adj = [set() for _ in range(n_v)]
    fvs = np.asarray(fv.face_vertices, np.int64)
    for tri in fvs:
        for a in range(3):
            for b in range(a + 1, 3):
                ia, ib = int(tri[a]), int(tri[b])
                adj[ia].add(ib)
                adj[ib].add(ia)
    bnd = fv.is_boundary
    bverts = set()
    for idx in np.where(bnd)[0]:
        for vtx in fvs[idx]:
            bverts.add(int(vtx))
    return adj, bverts


def morph_mesh(fv, boundary_displacement, max_iter=MAX_MORPH_ITER,
               relax=DEFAULT_MORPH_RELAX, tol=MORPH_TOL):
    """对网格节点施加边界面位移并做内部 Laplacian 松弛，返回变形后的节点坐标。

    - `boundary_displacement` (n_vertices,3)：每个顶点的目标位移；边界顶点被夹持
      （保持该位移），内部顶点以 `relax` 因子逐步均化邻居位移直至收敛。
    - 返回 (n_vertices,3) 变形后节点坐标（顶点结构不变，仅坐标位移）。

    网格运动后单元/面拓扑保持不变，仅节点坐标更新（适用于动网格重算）。
    """
    fv = _as_fv(fv)
    disp = np.asarray(boundary_displacement, float)
    n_v = fv.n_vertices
    if disp.shape != (n_v, 3):
        raise ValueError("boundary_displacement 形状应为 (%d, 3)" % n_v)
    adj, bverts = _build_vertex_adjacency(fv)
    d = disp.copy()
    # 内部顶点置零，边界顶点保持给定位移
    # 用邻居均值迭代松弛
    for _ in range(int(max_iter)):
        d_new = d.copy()
        dmax = 0.0
        for iv in range(n_v):
            if iv in bverts:
                continue
            nb = adj[iv]
            if not nb:
                continue
            avg = np.mean(d[list(nb)], axis=0)
            upd = d[iv] + relax * (avg - d[iv])
            d_new[iv] = upd
            dmax = max(dmax, float(_safe_norm(upd - d[iv])))
        d = d_new
        if dmax <= float(tol) * (1.0 + float(_safe_norm(d))):
            break
    return fv.vertices + d


# ---------------------------------------------------------------------------
# DFBI 6DOF 刚体动力学
# ---------------------------------------------------------------------------
def _quat_mul(qa, qb):
    """四元数乘法（Hamilton 约定，w 分量在前）。q = [w, x, y, z]。"""
    wa, xa, ya, za = qa
    wb, xb, yb, zb = qb
    return np.array([
        wa * wb - xa * xb - ya * yb - za * zb,
        wa * xb + xa * wb + ya * zb - za * yb,
        wa * yb - xa * zb + ya * wb + za * xb,
        wa * zb + xa * yb - ya * xb + za * wb], float)


def _quat_rotate(quat, vec):
    """用四元数把世界向量旋转（q · v · q⁻¹）。vec 为世界系向量。"""
    qv = np.array([0.0, float(vec[0]), float(vec[1]), float(vec[2])])
    qinv = np.array([quat[0], -quat[1], -quat[2], -quat[3]], float)
    return _quat_mul(_quat_mul(quat, qv), qinv)[1:]


def quat_to_dcm(quat):
    """四元数 → 方向余弦矩阵（3x3，从体到世界）。"""
    w, x, y, z = quat
    n = _safe_norm(quat)
    if n > _EPS:
        w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], float)


class DfbiBody:
    """DFBI 6DOF 刚体：质量/惯量、位置/姿态、线/角速度、受合力/力矩 → 运动积分。

    初始化给定质心位置 `position`、姿态四元数 `quaternion`（体→世界）、质量
    `mass`、体主轴惯量 `inertia` (3,)；`advance(force, torque, dt)` 以半隐式
    欧拉推进线/角速度并更新位置/姿态。`world_inertia()` 返回世界系惯量矩阵
    R I Rᵀ（用于角加速度 α = I_world⁻¹ τ）。角速度以周（rad/s），线速度 m/s。
    """

    def __init__(self, mass=1.0, inertia=(1.0, 1.0, 1.0),
                 position=(0.0, 0.0, 0.0), quaternion=(1.0, 0.0, 0.0, 0.0),
                 linear_velocity=(0.0, 0.0, 0.0),
                 angular_velocity=(0.0, 0.0, 0.0)):
        self.mass = float(mass)
        it = np.asarray(inertia, float).ravel()
        if it.size != 3:
            it = np.array([1.0, 1.0, 1.0], float)
        self.inertia_body = it
        self.position = np.asarray(position, float).ravel().copy()
        q = np.asarray(quaternion, float).ravel()
        n = _safe_norm(q)
        self.quaternion = (q / n) if n > _EPS else np.array([1.0, 0.0, 0.0, 0.0], float)
        self.linear_velocity = np.asarray(linear_velocity, float).ravel().copy()
        self.angular_velocity = np.asarray(angular_velocity, float).ravel().copy()
        self.force = np.zeros(3, float)
        self.torque = np.zeros(3, float)

    def world_inertia(self):
        """世界系惯量矩阵 I_world = R I_body Rᵀ（3x3，非奇异化）。"""
        R = quat_to_dcm(self.quaternion)
        Ib = np.diag(self.inertia_body)
        I_w = R @ Ib @ R.T
        # 非奇异保护：保证对角最小占比
        return I_w

    def inv_inertia(self):
        """世界系惯量逆矩阵（Moore-Penrose 非奇异化）。"""
        I_w = self.world_inertia()
        I_w = I_w + _EPS * np.eye(3)
        return np.linalg.inv(I_w)

    def advance(self, force, torque, dt):
        """半隐式欧拉推进一步：v += a·dt；ω += α·dt；姿态由 ω 积分。"""
        force = np.asarray(force, float).ravel()
        torque = np.asarray(torque, float).ravel()
        dt = float(dt)
        a = force / max(float(self.mass), _EPS)
        self.linear_velocity = self.linear_velocity + a * dt
        alpha = self.inv_inertia() @ (torque - np.cross(self.angular_velocity,
                                                        self.world_inertia() @ self.angular_velocity))
        self.angular_velocity = self.angular_velocity + alpha * dt
        # 位置：x += v dt
        self.position = self.position + self.linear_velocity * dt
        # 姿态：q ← q ⊗ exp(ω dt /2)，ω 为世界系角速度
        wmag = _safe_norm(self.angular_velocity)
        if wmag > _EPS:
            axis = self.angular_velocity / wmag
            half = 0.5 * wmag * dt
            dq = np.array([np.cos(half),
                           axis[0] * np.sin(half),
                           axis[1] * np.sin(half),
                           axis[2] * np.sin(half)], float)
            self.quaternion = _quat_mul(dq, self.quaternion)
            n = _safe_norm(self.quaternion)
            self.quaternion = self.quaternion / n
        return dict(position=self.position.copy(),
                    quaternion=self.quaternion.copy(),
                    linear_velocity=self.linear_velocity.copy(),
                    angular_velocity=self.angular_velocity.copy())


# ---------------------------------------------------------------------------
# overset 重叠网格插值
# ---------------------------------------------------------------------------
def overset_donor_weights(receptor_points, donor_centroids, k=3, power=2.0):
    """对每个 receptor 给出最近 k 个 donor 反权重插值。

    - `receptor_points` (M,3)，`donor_centroids` (N,3)。
    - 返回 (M,N) 权重矩阵（已归一化、逐行和 1）；每个 receptor 至多 k 个非零。
    - 最近 donor 距离为 0 时退化为硬切换（该行权重=该 donor）。
    """
    rp = np.asarray(receptor_points, float)
    dc = np.asarray(donor_centroids, float)
    M, N = rp.shape[0], dc.shape[0]
    W = np.zeros((M, N), float)
    for m in range(M):
        d = _safe_norm(dc - rp[m], axis=1)
        if N <= k:
            idx = np.arange(N)
        else:
            idx = np.argsort(d)[:int(k)]
        dist = d[idx]
        if float(dist.min()) <= _EPS:
            j = int(np.argmin(dist))
            W[m, idx[j]] = 1.0
            continue
        inv = 1.0 / np.maximum(dist, _EPS) ** float(power)
        W[m, idx] = inv / inv.sum()
    return W


def overset_interpolate(receptor_points, donor_centroids, donor_values, k=3,
                        power=2.0):
    """在重叠网格间插值场量：`overset_interpolate(points, donors, values)`。

    - `donor_values` (N,) 或 (N,n_field)：每个 donor 的待插值标量/向量。
    - 返回 (M,) 或 (M,n_field)：receptor 处的插值结果。
    """
    W = overset_donor_weights(receptor_points, donor_centroids, k=k, power=power)
    vals = np.asarray(donor_values, float)
    out = W @ vals
    if vals.ndim == 1:
        return out
    return out


# ---------------------------------------------------------------------------
# MotionSolver（P10 后端兼容门面 + 运动耦合）
# ---------------------------------------------------------------------------
def _as_fv(fv):
    """接受 FVM 实例或其容器（有 _fv 属性者取之）。"""
    if isinstance(fv, FVM):
        return fv
    inner = getattr(fv, "_fv", None)
    if isinstance(inner, FVM):
        return inner
    raise TypeError("需提供 FVM 实例或含 _fv 的容器")


class MotionSolver:
    """运动谱系求解器：刚体运动 / 滑移 interface / morphing / DFBI 6DOF / overset /
    MRF 旋转源 的统一门面。

    `mode` 决定 `update()`（推进网格运动/刚体姿态）与 `mrf_source()`（仅旋转类
    返回离心+科氏源）的行为；`motion`（默认）＝刚体旋转+平移+滑移，`morphing`＝
    网格变形，`dfbi`＝刚体动力学双向回馈，`overset`＝重叠插值，`mrf`＝纯旋转源。
    """

    def __init__(self, fv, mode="motion", rotation_speed=DEFAULT_ROTATION_SPEED,
                 rotation_axis=DEFAULT_ROTATION_AXIS,
                 translation_velocity=DEFAULT_TRANSLATION_VELOCITY,
                 reference_point=DEFAULT_REFERENCE_POINT,
                 morph_max_iter=MAX_MORPH_ITER,
                 morph_relax=DEFAULT_MORPH_RELAX,
                 bodies=None, dfbi_dt=DEFAULT_ROTATION_DT,
                 overset_k=3, overset_power=2.0):
        self.fv = _as_fv(fv)
        self.mode = str(mode).strip().lower().replace("-", "").replace(" ", "")
        self.rotation_speed = float(rotation_speed)
        self.rotation_axis = _unit(rotation_axis)
        self.translation_velocity = np.asarray(translation_velocity, float).ravel()
        self.reference_point = np.asarray(reference_point, float).ravel()
        self.morph_max_iter = int(morph_max_iter)
        self.morph_relax = float(morph_relax)
        self.overset_k = int(overset_k)
        self.overset_power = float(overset_power)
        # DFBI 体容器（多个 6DOF 体）
        if bodies is None:
            bodies = [DfbiBody()]
        if isinstance(bodies, DfbiBody):
            bodies = [bodies]
        self.bodies = list(bodies)
        self.dfbi_dt = float(dfbi_dt)
        # 运行时量
        self._u = np.zeros(self.fv.n_cells, float)
        self._v = np.zeros(self.fv.n_cells, float)
        self._w = np.zeros(self.fv.n_cells, float)
        self._mdot = None
        self._t = 0.0
        self._displacement = np.zeros((self.fv.n_vertices, 3), float)
        self._residual = 0.0
        self.iteration = 0

    # -- 场量设置 / 运动推进 ------------------------------
    def set_velocities(self, u, v, w, mdot=None):
        """存储当前求解速度场（MRF 科氏源与 DFBI 受力用到）。"""
        self._u = np.asarray(u, float).ravel()
        self._v = np.asarray(v, float).ravel()
        self._w = np.asarray(w, float).ravel()
        self._mdot = mdot

    def update(self, u, v, w, mdot=None):
        """在 SIMPLE 环尾部推进一次运动：时间推进 + 网格/刚体状态更新。

        返回归一化残差（位移变化量对 ||位移|| 之比）。默认不做实际网格重排，
        仅随 `_t` 推进并在旋转/DFBI 模式更新刚体姿态（供后处理/源项重算）。
        """
        self.set_velocities(u, v, w, mdot=mdot)
        dt = self.dfbi_dt
        disp_old = self._displacement.copy()
        self._t += dt
        if self.mode in ("motion", "rigid", "rotating", "rotational", "sliding"):
            # 刚体位移（以 t 时刻为准），供 morphing/后处理参考
            self._displacement = self._rigid_displacement()
        elif self.mode == "morphing":
            if self._mdot is not None:
                self._displacement = np.zeros_like(self._displacement)
        elif self.mode == "dfbi":
            total_f = np.zeros(3, float)
            total_t = np.zeros(3, float)
            for b in self.bodies:
                b.advance(b.force, b.torque, dt)
                total_f += b.force
                total_t += b.torque
            self._displacement = self._dfbi_displacement()
        elif self.mode == "overset":
            pass
        denom = max(float(_safe_norm(self._displacement)), _EPS)
        resid = float(_safe_norm(self._displacement - disp_old)) / denom
        self._residual = resid
        self.iteration += 1
        return resid

    def _rigid_displacement(self):
        """刚体位移 = rigid_transform 结果 - 原始顶点坐标（供后处理/源重算）。"""
        new = rigid_transform(self.fv.vertices, self.rotation_axis,
                              self.rotation_speed, self.translation_velocity,
                              self.reference_point, t=self._t)
        return new - self.fv.vertices

    def _dfbi_displacement(self):
        """DFBI 体当前位移（首个体质心相对初始位置）；无体时返回零。"""
        b0 = self.bodies[0] if self.bodies else None
        if b0 is None:
            return self._displacement
        return np.broadcast_to(b0.position, self._displacement.shape).copy()

    def mrf_source(self, rho=None):
        """MRF 旋转参考系源（离心 + 科氏），逐单元力密度 (n_cells,3)。

        仅旋转类 mode 且 omega≠0 时非零；`rho` 缺省用调用方传入（压力求解器
        注入当前密度场），否则以单位密度 1.0 计算。
        """
        if self.mode not in ("motion", "rigid", "rotating", "rotational",
                             "sliding", "mrf"):
            return np.zeros((self.fv.n_cells, 3), float)
        rho_arr = 1.0 if rho is None else rho
        return mrf_source(self.fv.centroids, self._u, self._v, self._w,
                          self.rotation_speed, self.rotation_axis, rho_arr,
                          self.reference_point)

    def mesh_displacement(self):
        """当前网格节点位移 (n_vertices,3)。"""
        return self._displacement

    def body_kinematics(self):
        """各 DFBI 体运动学（位置/姿态/线角速度）列表。"""
        return [dict(position=b.position.copy(), quaternion=b.quaternion.copy(),
                     linear_velocity=b.linear_velocity.copy(),
                     angular_velocity=b.angular_velocity.copy())
                for b in self.bodies]

    # -- P10 后端兼容门面 ------------------------------
    def _initialize_field(self):
        self._t = 0.0
        self._displacement = np.zeros((self.fv.n_vertices, 3), float)
        self._residual = 0.0
        self.iteration = 0
        return dict(mode=self.mode, n_bodies=len(self.bodies),
                    speed=self.rotation_speed,
                    disp_max=float(_safe_norm(self._displacement)))

    def step(self):
        resid = float(self.update(self._u, self._v, self._w, mdot=self._mdot))
        self._residual = resid
        return dict(residual=resid, mode=self.mode,
                    disp_max=float(_safe_norm(self._displacement)))

    def residual(self):
        return self._residual

    def monitor_payload(self):
        return dict(mode=self.mode, n_bodies=len(self.bodies),
                    rotation_speed=self.rotation_speed,
                    displacement_max=float(_safe_norm(self._displacement)),
                    t=self._t, iteration=self.iteration, residual=self._residual)


# ---------------------------------------------------------------------------
# 工厂注册表
# ---------------------------------------------------------------------------
_MOTION_MODES = {
    "motion": "motion", "rigid": "motion", "rigidbody": "motion",
    "rotating": "motion", "rotational": "motion", "rotation": "motion",
    "sliding": "sliding", "slidinginterface": "sliding",
    "mrf": "mrf", "rotatingreferenceframe": "mrf",
    "morphing": "morphing", "morph": "morphing", "deform": "morphing",
    "dfbi": "dfbi", "6dof": "dfbi", "sixdof": "dfbi", "dof6": "dfbi",
    "overset": "overset", "overlapping": "overset", "chimera": "overset",
}


def make_motion(fv, model="motion", **kwargs):
    """按字符串名实例化运动谱系模型（大小写/连字符不敏感）。

    别名包含 motion/rigid/rotating/rotation/mrf/sliding/morph/
    morphing/deform/dfbi/6dof/overset 等；`model` 决定 `mode`。
    """
    key = str(model).strip().lower().replace("-", "").replace(" ", "").replace("_", "")
    if key not in _MOTION_MODES:
        names = ", ".join(sorted(set(_MOTION_MODES.keys())))
        raise ValueError("未知运动谱系模型 %r（支持：%s）" % (model, names))
    return MotionSolver(fv, mode=_MOTION_MODES[key], **kwargs)

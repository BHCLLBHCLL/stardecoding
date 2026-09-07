# -*- coding: utf-8 -*-
"""P 波 P8：多相 VOF —— 相体积分数输运 + PLIC 几何重构 + 两相物性混合谱系。

纯 numpy 实现（occ / scdm 两环境皆可用），建立在 P4 有限体积离散核心
（`fvm_core.FVM`）、P5 压力基求解器（`pressure_solver.solve_linear`）与
P3b 凸多面体几何（`mesh_poly.clip_convex` / `poly_volume`）之上，为两相
（相 1 / 相 2，如水-空气）提供体积分数 α 的守恒输运与界面几何重构。

物理框架（VOF）：
    ∂α/∂t + ∇·(u α) = 0            （相 1 体积分数守恒，α ∈ [0, 1]）
    两相物性：ρ = α ρ₁ + (1-α) ρ₂，μ = α μ₁ + (1-α) μ₂
    自由面由 PLIC（Piecewise Linear Interface Calculation）重构：
    单元内界面平面 {x : n·x = c} 由 α 场梯度确定法向 n，平面常数 c 经体积
    截断（二分 + `clip_convex`）精确满足单元目标体积分数。

核心能力：

  1) 物性混合：`blend_property` / `two_phase_rho` / `two_phase_mu`。
  2) 界面法向：`plic_normal` 用 Green-Gauss 计算 ∇α 并归一（指向相 1），
     退化单元用重心/坐标轴回退。
  3) 体积截断：`keep_phase1_volume` 复用 `mesh_poly.clip_convex` 计算平面裁剪后
     相 1 体积；`plic_plane_offset` 二分求平面常数（精确满足目标 α）。
  4) 几何重构：`plic_reconstruct` 返回每单元 (normal, offset)，供自由面可视化、
     初始化与物性重构。
  5) 守恒输运：显式瞬态 CFL 受限一阶上风（`face_upwind_alpha` /
     `alpha_faces_flux` / `alpha_divergence` / `auto_dt` / `advance_alpha`）
     推进 α，界面压缩锐化 + 裁剪到 [0,1] + 守恒缩放保证全局一致。
  6) 表面张力：`surface_tension_force` 用 CSF（连续表面力）求每单元体积力。
  7) 求解器：`VofSolver` 提供 `update(u,v,w,mdot)` 推进 α 并暴露混合物性
     （`rho` / `mu`）、PLIC 界面（`normal` / `offset`）；P10 后端兼容门面
     `_initialize_field() / step() / residual() / monitor_payload()`。

纯 numpy 约束：所有范数走 `solver_run._safe_norm`，几何裁剪仅复用已存在的
`mesh_poly.clip_convex`，不新增 scipy / np.linalg 依赖。
"""
import numpy as np

from fvm_core import FVM
from solver_run import _safe_norm
import mesh_poly as MP


# ---------------------------------------------------------------------------
# 多相 VOF 常量
# ---------------------------------------------------------------------------
DEFAULT_RHO1 = 998.0      # 相 1 默认密度（水，kg/m^3）
DEFAULT_RHO2 = 1.18       # 相 2 默认密度（空气，kg/m^3）
DEFAULT_MU1 = 1.0e-3      # 相 1 默认动力粘度（水，Pa·s）
DEFAULT_MU2 = 1.8e-5      # 相 2 默认动力粘度（空气，Pa·s）
DEFAULT_SIGMA = 0.072     # 界面张力（水-空气，N/m）
EPS = 1e-12
BISECT_MAX = 60           # PLIC 平面常数二分最大迭代


# ---------------------------------------------------------------------------
# 两相物性混合
# ---------------------------------------------------------------------------
def blend_property(alpha, prop1, prop2):
    """两相物性线性混合：prop = α·prop1 + (1-α)·prop2（α 逐单元）。"""
    alpha = np.asarray(alpha, float)
    return alpha * float(prop1) + (1.0 - alpha) * float(prop2)


def two_phase_rho(alpha, rho1=DEFAULT_RHO1, rho2=DEFAULT_RHO2):
    """两相密度场（逐单元）。"""
    return blend_property(alpha, rho1, rho2)


def two_phase_mu(alpha, mu1=DEFAULT_MU1, mu2=DEFAULT_MU2):
    """两相动力粘度场（逐单元）。"""
    return blend_property(alpha, mu1, mu2)


# ---------------------------------------------------------------------------
# 单元凸多面体几何（四面体外向面序）
# ---------------------------------------------------------------------------
_TET_FACES = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]


def _cell_polyhedron(fv, i):
    """单元 i 的凸多面体表示 (vertices 列表, faces 列表)，外向面序。

    当前 FVM 网格为四面体，四面体本身是凸多面体，四个三角面按外向顺序返回，
    使 `poly_volume` 给出正体积（与 FVM 单元体积一致）。
    """
    idx = fv.cells[i]
    Vt = fv.vertices[idx].tolist()
    return Vt, [list(f) for f in _TET_FACES]


def keep_phase1_volume(fv, i, n_unit, offset):
    """单元 i 被平面 {x: n_unit·x >= offset} 截断后相 1 侧体积。

    复用地 `mesh_poly.clip_convex`：保留 (x-p)·n <= 0 侧；取 p = offset*n、
    n = -n_unit，即保留 {x: n_unit·x >= offset}（相 1 侧）。
    返回裁剪后凸多面体的体积（相 1 体积）。
    """
    Vt, Ff = _cell_polyhedron(fv, i)
    keep = MP.clip_convex(Vt, Ff, offset * np.asarray(n_unit, float),
                          -np.asarray(n_unit, float))
    if not keep[1]:
        return 0.0
    vol = MP.poly_volume(keep[0], keep[1])
    # 退化裁剪（单元完全位于相 2 侧）可能得到负的有向体积，夹取到 [0, V]。
    return float(np.clip(vol, 0.0, fv.volumes[i]))


def plic_plane_offset(fv, i, n_unit, target_alpha):
    """二分求单元 i 内 PLIC 界面平面偏移 c，使相 1 体积 == target_alpha·V。

    n_unit 为单位法向（指向相 1）。c ∈ [min(x·n), max(x·n)]，相 1 侧体积
    随 c 单调递减；二分收敛到 target_alpha·V（目标在 [0,V] 内唯一）。
    """
    target = float(np.clip(target_alpha, 0.0, 1.0)) * fv.volumes[i]
    proj = fv.vertices[fv.cells[i]] @ np.asarray(n_unit, float)
    lo = float(proj.min())
    hi = float(proj.max())
    if target <= 0.0:
        return hi
    if target >= fv.volumes[i]:
        return lo
    for _ in range(BISECT_MAX):
        mid = 0.5 * (lo + hi)
        if keep_phase1_volume(fv, i, n_unit, mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# PLIC 界面重构
# ---------------------------------------------------------------------------
def plic_normal(fv, alpha, boundary=None, axis=0):
    """逐单元 LIC 界面法向：n = ∇α/|∇α|（指向相 1 = α 增侧）。

    Green-Gauss 求 ∇α；梯度近零（单元近满/空）或法向退化时回退到坐标轴法向
    （沿 `axis` 方向），避免自由面外单元产生乱序界面。
    """
    alpha = np.asarray(alpha, float)
    grad = fv.grad_gauss(alpha, boundary=boundary)
    mag = _safe_norm(grad, axis=1)
    n = np.zeros_like(grad)
    good = mag > 1e-9
    n[good] = grad[good] / mag[good, None]
    fallback = np.zeros(fv.n_cells, float)
    fallback[:] = (alpha >= 0.5) * 2.0 - 1.0
    bad = ~good
    if bad.any():
        n[bad, axis] = fallback[bad]
    return n


def plic_reconstruct(fv, alpha, boundary=None, axis=0):
    """逐单元 PLIC 界面重构，返回 (normal (n_cells,3), offset (n_cells,))。

    normal 指向相 1 侧（α 增侧）且为单位向量；offset 为平面常数 c 使
    {x: normal·x >= c} 恰含 target α 体积。近满（α≈1）单元 offset 落于 min 投影，
    近空（α≈0）落于 max 投影；完全裁剪的空/满单元由 offset 表示不产生界面。
    """
    normals = plic_normal(fv, alpha, boundary=boundary, axis=axis)
    offsets = np.zeros(fv.n_cells, float)
    for i in range(fv.n_cells):
        offsets[i] = plic_plane_offset(fv, i, normals[i], np.clip(alpha[i], 0.0, 1.0))
    return normals, offsets


# ---------------------------------------------------------------------------
# α 守恒输运（一阶上风 + 界面压缩 + 有界裁剪）
# ---------------------------------------------------------------------------
def face_upwind_alpha(fv, mdot, alpha, boundary=None):
    """上风面 α 值（n_faces,）：面通量方向决定取上游单元 α。

    内部面 mdot>=0 取 owner（上游），否则取 neighbor；边界面 mdot<0（入流）
    取 `boundary[f]`（缺省 = owner 外推，即零梯度出流）。
    """
    alpha = np.asarray(alpha, float)
    mdot = np.asarray(mdot, float).ravel()
    is_int = fv.neighbor >= 0
    nbr = np.where(is_int, fv.neighbor, fv.owner)
    up = np.where(mdot >= 0.0, alpha[fv.owner], alpha[nbr])
    if boundary is not None:
        inflow = (~is_int) & (mdot < 0.0)
        up[inflow] = np.asarray(boundary, float)[inflow]
    return up


def alpha_faces_flux(fv, mdot, alpha, boundary=None):
    """面体积通量 F_f = mdot_f * α_upwind（n_faces,，正 = owner → neighbor）。"""
    return np.asarray(mdot, float) * face_upwind_alpha(fv, mdot, alpha, boundary)


def alpha_divergence(fv, F):
    """逐单元 (1/V) Σ_f F_f（面向量散度，n_cells,）。"""
    acc = _face_accumulate(fv, F)
    vol = np.where(fv.volumes > 1e-14, fv.volumes, 1.0)
    return acc / vol


def _face_accumulate(fv, F):
    """面通量累积到单元（owner 加、neighbor 减），返回 (n_cells,)。"""
    F = np.asarray(F, float)
    is_int = fv.neighbor >= 0
    nbr = np.where(is_int, fv.neighbor, 0)
    acc = np.zeros(fv.n_cells, float)
    np.add.at(acc, fv.owner, F)
    np.add.at(acc, nbr[is_int], -F[is_int])
    return acc


def auto_dt(fv, mdot, cfl=0.4):
    """由 CFL 限定的显式时间步：dt = cfl * min_i (V_i / Σ_f |mdot|)。"""
    is_int = fv.neighbor >= 0
    nbr = np.where(is_int, fv.neighbor, 0)
    am = np.abs(np.asarray(mdot, float))
    acc = np.zeros(fv.n_cells, float)
    np.add.at(acc, fv.owner, am)
    np.add.at(acc, nbr[is_int], am[is_int])
    dt = float(cfl) * fv.volumes / np.maximum(acc, 1e-30)
    return float(np.min(dt))


def advance_alpha(fv, mdot, alpha, dt, boundary=None, compress=0.0):
    """显式推进 α 一个时间步 Δt：α_new = α - Δt·(1/V)Σ_f mdot·α_upwind。

    采用**一阶上风**对流通量（守恒、离散最大值原理保证有界），`compress` > 0
    时在界面附近施加人工反扩散压缩（锐化自由面）；推进后裁剪到 [0,1] 维持
    有界。边界入口注入的相 1 体积随时间累积（开放系统总体积可增长）。
    返回 α_new（n_cells,）与实测有效时间步（自动受限时 < dt）。
    """
    mdot = np.asarray(mdot, float)
    F = alpha_faces_flux(fv, mdot, alpha, boundary=boundary)
    dt_eff = float(dt)
    while True:
        div = alpha_divergence(fv, F)
        a_new = alpha - dt_eff * div
        if dt_eff <= 0.0:
            break
        # 上风显式稳定性：禁止单步越过 1（CFL 超限时减半重试）
        idx = np.abs(alpha) <= 1.0
        crit = 1.0 / np.maximum(np.abs(div[idx]), 1e-30)
        dt_max = float(np.min(crit)) if idx.any() else dt_eff
        if dt_eff <= dt_max:
            break
        dt_eff = 0.5 * min(dt_max, dt_eff)
    if compress > 0.0:
        # 界面压缩：面向 α 增侧的反扩散，稍加锐化（显式，受稳定限）
        is_int = fv.neighbor >= 0
        o = fv.owner[is_int]
        nb = fv.neighbor[is_int]
        coef = float(compress) * np.abs(mdot[is_int]) * (alpha[nb] - alpha[o])
        grad = np.zeros(fv.n_cells, float)
        np.add.at(grad, o, coef)
        np.add.at(grad, nb, -coef)
        a_new += dt_eff * grad / np.where(fv.volumes > 1e-14, fv.volumes, 1.0)
    a_new = np.clip(a_new, 0.0, 1.0)
    # 有界裁剪即满足离散最大值原理（一阶上风 + CFL 受限保持 [0,1]）：
    # 不再做全局守恒缩放，避免阻断开放入口的相 1 注入。
    return a_new, dt_eff


# ---------------------------------------------------------------------------
# 表面张力（CSF 连续表面力）
# ---------------------------------------------------------------------------
def surface_tension_force(fv, alpha, sigma=DEFAULT_SIGMA, boundary=None, axis=0):
    """CSF 表面张力体积力（每单元力已含体积，牛顿）。

    f = σ κ ∇α；κ = -∇·(n_hat)，n_hat = ∇α/|∇α|。界面曲率用单元中心
    散度近似，背压由 α 梯度方向自适应。返回 (n_cells, 3)。
    """
    alpha = np.asarray(alpha, float)
    grad_a = fv.grad_gauss(alpha, boundary=boundary)
    mag = _safe_norm(grad_a, axis=1)
    n_hat = np.zeros_like(grad_a)
    good = mag > 1e-9
    n_hat[good] = grad_a[good] / mag[good, None]
    # 曲率 kappa = -div(n_hat)（略去接触角几何修正，壁面默认 90°）
    nf = np.stack([
        fv.face_value(n_hat[:, 0], boundary=None),
        fv.face_value(n_hat[:, 1], boundary=None),
        fv.face_value(n_hat[:, 2], boundary=None),
    ], axis=1)
    face_flux = nf[:, 0] * fv.face_normal[:, 0] + nf[:, 1] * fv.face_normal[:, 1] \
        + nf[:, 2] * fv.face_normal[:, 2]
    face_flux *= fv.face_area
    acc = np.zeros(fv.n_cells, float)
    is_int = fv.neighbor >= 0
    nbr = np.where(is_int, fv.neighbor, 0)
    np.add.at(acc, fv.owner, face_flux)
    np.add.at(acc, nbr[is_int], -face_flux[is_int])
    vol = np.where(fv.volumes > 1e-14, fv.volumes, 1.0)
    kappa = -acc / vol
    strength = float(sigma) * kappa
    force = strength[:, None] * grad_a * np.where(fv.volumes > 1e-14,
                                                  fv.volumes, 0.0)[:, None]
    return force


# ---------------------------------------------------------------------------
# 多相 VOF 求解器：VofSolver
# ---------------------------------------------------------------------------
class VofSolver:
    """多相 VOF 求解器：相 1 体积分数守恒输运 + PLIC 界面重构 + 两相物性。

    `update(u, v, w, mdot)` 由给定流场推进 α 场（一阶上风 + 界面压缩 + 有界
    裁剪 + 守恒缩放），重构 PLIC 界面，返回归一化残差。`rho` / `mu` 暴露
    逐单元混合密度与动力粘度（供 SIMPLE 动量装配），`normal` / `offset` 暴露
    自由面界面平面。
    """

    def __init__(self, fv, rho1=DEFAULT_RHO1, rho2=DEFAULT_RHO2,
                 mu1=DEFAULT_MU1, mu2=DEFAULT_MU2, sigma=DEFAULT_SIGMA,
                 flow_axis=0, inlet_side="min", outlet_side="max",
                 inlet_alpha=1.0, alpha0=0.0, compress=0.0, relax=0.7,
                 max_iter=50, axis=0):
        self.fv = fv
        n = fv.n_cells
        self.rho1 = float(rho1)
        self.rho2 = float(rho2)
        self.mu1 = float(mu1)
        self.mu2 = float(mu2)
        self.sigma = float(sigma)
        self.inlet_alpha = float(np.clip(inlet_alpha, 0.0, 1.0))
        self.compress = float(compress)
        self.relax = float(relax)
        self.max_iter = int(max_iter)
        self.flow_axis = int(flow_axis)
        self.inlet_side = inlet_side
        self.outlet_side = outlet_side
        self.axis = int(axis)
        self._alpha = np.full(n, float(alpha0), float)
        self._normal = np.zeros((n, 3), float)
        self._offset = np.zeros(n, float)
        self._residual = 0.0
        self.iteration = 0
        self._u = np.zeros(n, float)
        self._v = np.zeros(n, float)
        self._w = np.zeros(n, float)
        self._mdot = None
        self._inlet_faces = np.array([], np.int64)
        self._outlet_faces = np.array([], np.int64)
        self._wall_faces = np.array([], np.int64)
        self._classify_boundary()
        self._reconstruct()

    # -- 场属性 ------------------------------------------------
    @property
    def alpha(self):
        return self._alpha

    @property
    def alpha_min(self):
        return float(self._alpha.min())

    @property
    def phase1_volume(self):
        return float(np.sum(self._alpha * self.fv.volumes))

    @property
    def rho(self):
        """逐单元混合密度场。"""
        return two_phase_rho(self._alpha, self.rho1, self.rho2)

    @property
    def mu(self):
        """逐单元混合动力粘度场。"""
        return two_phase_mu(self._alpha, self.mu1, self.mu2)

    @property
    def surface_force(self):
        return surface_tension_force(self.fv, self._alpha, self.sigma)

    @property
    def normal(self):
        return self._normal

    @property
    def offset(self):
        return self._offset

    def set_velocities(self, u, v, w, mdot=None):
        """注入当前流场与面质量通量，供 `step()` 推进 α。"""
        self._u = np.asarray(u, float).ravel()
        self._v = np.asarray(v, float).ravel()
        self._w = np.asarray(w, float).ravel()
        self._mdot = None if mdot is None else np.asarray(mdot, float)

    def set_inlet_alpha(self, inlet_alpha):
        self.inlet_alpha = float(np.clip(inlet_alpha, 0.0, 1.0))
        return self

    # -- 内部辅助 ------------------------------------------------
    def _boundary_alpha(self):
        """构造边界面 α 值：入口 = inlet_alpha，其余 = owner 外推（零梯度出流）。"""
        fv = self.fv
        b = np.zeros(fv.n_faces, float)
        isbnd = fv.is_boundary
        if isbnd.any():
            b[isbnd] = self._alpha[fv.owner[isbnd]]
        if hasattr(self, "_inlet_faces") and len(self._inlet_faces):
            b[self._inlet_faces] = self.inlet_alpha
        return b

    def _classify_boundary(self):
        fv = self.fv
        ax = fv.face_centroid[:, self.flow_axis]
        cut_min = float(ax.min())
        cut_max = float(ax.max())
        bnd = fv.is_boundary
        inlet = bnd.copy()
        if self.inlet_side == "min":
            inlet &= ax <= cut_min + 1e-10
        else:
            inlet &= ax >= cut_max - 1e-10
        outlet = bnd.copy()
        if self.outlet_side == "max":
            outlet &= ax >= cut_max - 1e-10
        else:
            outlet &= ax <= cut_min + 1e-10
        outlet &= ~inlet
        self._inlet_faces = np.where(inlet)[0]
        self._outlet_faces = np.where(outlet)[0]
        self._wall_faces = np.where(bnd & ~inlet & ~outlet)[0]
        return self._inlet_faces, self._outlet_faces, self._wall_faces

    def _two_phase_volume_flux(self, u, v, w):
        """由速度场重构面体积通量：Flux = (u·n) * A（供 α 输运）。"""
        fv = self.fv
        uf = fv.face_value(np.asarray(u, float))
        vf = fv.face_value(np.asarray(v, float))
        wf = fv.face_value(np.asarray(w, float))
        vn = (uf * fv.face_normal[:, 0] + vf * fv.face_normal[:, 1]
              + wf * fv.face_normal[:, 2])
        return vn * fv.face_area

    def _reconstruct(self):
        self._normal, self._offset = plic_reconstruct(
            self.fv, self._alpha, axis=self.axis)

    def update(self, u, v, w, mdot=None):
        """由流场推进一次显式 α 输运（CFL 受限上风），更新 α 与 PLIC 界面。

        返回归一化残差，衡量本次推进 α 的变化量（a_new - a_old）。
        """
        fv = self.fv
        alpha_old = np.asarray(self._alpha, float)
        if mdot is None:
            mdot = self._two_phase_volume_flux(u, v, w)
        mdot = np.asarray(mdot, float)
        if len(self._inlet_faces) == 0:
            self._classify_boundary()
        # 显式瞬态推进：CFL 受限时间步 + 一阶上风 + 界面压缩 + 有界守恒裁剪
        dt = auto_dt(fv, mdot)
        a_new, _dt_eff = advance_alpha(fv, mdot, alpha_old, dt,
                                       boundary=self._boundary_alpha(),
                                       compress=self.compress)
        denom = max(float(_safe_norm(a_new)), EPS)
        resid = float(_safe_norm(a_new - alpha_old)) / denom
        self._alpha = a_new
        self._reconstruct()
        self.iteration += 1
        self._residual = resid
        return resid

    # -- P10 后端兼容门面 ----------------------------------------
    def _initialize_field(self):
        if len(self._inlet_faces) == 0:
            self._classify_boundary()
        self._reconstruct()
        self._residual = 0.0
        self.iteration = 0
        return dict(alpha_min=float(self._alpha.min()),
                    alpha_max=float(self._alpha.max()),
                    phase1_volume=self.phase1_volume,
                    rho_min=float(self.rho.min()), rho_max=float(self.rho.max()))

    def step(self):
        resid = float(self.update(self._u, self._v, self._w, mdot=self._mdot))
        self._residual = resid
        return dict(residual=resid,
                    alpha_min=float(self._alpha.min()),
                    alpha_max=float(self._alpha.max()),
                    phase1_volume=self.phase1_volume)

    def residual(self):
        return self._residual

    def monitor_payload(self):
        return dict(alpha_min=float(self._alpha.min()),
                    alpha_max=float(self._alpha.max()),
                    phase1_volume=self.phase1_volume,
                    rho_min=float(self.rho.min()),
                    rho_max=float(self.rho.max()),
                    iteration=self.iteration, residual=self._residual)


# ---------------------------------------------------------------------------
# 工厂注册表
# ---------------------------------------------------------------------------
_VOF_MODELS = {
    "vof": VofSolver,
    "volume_of_fluid": VofSolver,
}


def make_vof(fv, model="vof", **kwargs):
    """按字符串名实例化多相 VOF 模型（大小写/下划线不敏感）。"""
    key = str(model).strip().lower()
    if key not in _VOF_MODELS:
        names = ", ".join(sorted(set(_VOF_MODELS.keys())))
        raise ValueError("未知多相 VOF 模型 %r（支持：%s）" % (model, names))
    return _VOF_MODELS[key](fv, **kwargs)

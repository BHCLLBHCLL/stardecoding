# -*- coding: utf-8 -*-
"""P 波 P8：DPM 离散相 —— 拉格朗日粒子（parcel）注入 + 运动方程积分 + 连续相耦合。

纯 numpy 实现（occ / scdm 两环境皆可用），建立在 P4 有限体积离散核心
（`fvm_core.FVM`）之上，为多相流中的弥散离散相（粒子/液滴/气泡）提供**单相耦合**
（one-way）的拉格朗日追踪：入口注入、拖曳 + 重力/浮力运动积分、跨四面体单元定位、
连续相动量源（`coupling_source()`）与粒子轨（`trajectories()`）。与 VOF（两相）
Mixture（多相体积分数）构成 P8 多相谱系的欧拉-拉格朗日互补层次。

运动方程（单 parcel，m_p = ρ_p π d³/6）：
    dX_p/dt = U_p
    dU_p/dt = (U_f - U_p)/τ + g (ρ_p - ρ_f)/ρ_p
    τ = ρ_p d_p²/(18 μ_f) · 24/(C_D Re)     （Schiller-Naumann 拖曳松弛时间）
    Re = ρ_f |U_f - U_p| d_p / μ_f
位置/速度积分采用**半隐式欧拉**（先更新速度、再更新位置），拖曳主导时无条件稳定。

连续相耦合：`coupling_source()` 返回逐单元力密度（SIMPLE 动量 RHS 体源项）
    S_mom[c] = -Σ_{p∈c} m_p (U_f - U_p)/τ / V_c
（粒子对流体反作用力，即流体从拖曳中获得/损失的动量密度）。

纯 numpy 约束：所有范数走 `solver_run._safe_norm`，拖曳系数复用
`mixture.drag_coefficient`（Schiller-Naumann），不新增 scipy 依赖。
"""
import numpy as np

from fvm_core import FVM
from solver_run import _safe_norm
from mixture import drag_coefficient as _drag_coefficient


# ---------------------------------------------------------------------------
# 多相 DPM 常量
# ---------------------------------------------------------------------------
DEFAULT_RHO_P = 2500.0          # 粒子密度（kg/m^3）
DEFAULT_DIA_P = 1.0e-4          # 粒子直径（m）
DEFAULT_GRAVITY = (0.0, 0.0, -9.81)
DEFAULT_DENSITY_F = 998.0       # 连续相密度缺省（水）
DEFAULT_MU_F = 1.0e-3           # 连续相动力粘度缺省（水，Pa·s）
DEFAULT_MASS_FLOW = 1.0e-4      # 每 parcel 代表的入口质量流量（kg/s）
DEFAULT_PARCELS = 10            # 每次注入的 parcel 数
DEFAULT_SPEED = 1.0             # 入口流速缺省
EPS = 1e-12
LOC_TOL = 1e-9                  # 四面体定位容差
INJ_OFFSET = 1e-6               # 入口注入向域内偏移（m）
STATUS_ACTIVE = 0               # 仍在域内
STATUS_ESCAPED = -1             # 离开计算域
STATUS_STUCK = 1                # 壁面滞留


# ---------------------------------------------------------------------------
# 四面体点定位（barycentric）
# ---------------------------------------------------------------------------
def _signed_vol(a, b, c, d):
    """带号体积的 6 倍：det(b-a, c-a, d-a)（a/b/c/d 形状 (M,3) 或 (3,)）。"""
    e1 = b - a
    e2 = c - a
    e3 = d - a
    return (e1[..., 0] * (e2[..., 1] * e3[..., 2] - e2[..., 2] * e3[..., 1])
            - e1[..., 1] * (e2[..., 0] * e3[..., 2] - e2[..., 2] * e3[..., 0])
            + e1[..., 2] * (e2[..., 0] * e3[..., 1] - e2[..., 1] * e3[..., 0]))


def locate_cells(fv, positions, tol=LOC_TOL):
    """为每个点定位所在四面体单元（barycentric 体积坐标），逐点向量化查所有单元。

    对正定向四面体 (A,B,C,D)，体积坐标 λ = (λ_A, λ_B, λ_C, λ_D) 由子体积与参考
    体积之比求得；点位于单元内当且仅当全部 λ ≥ -tol。返回 (n,) 单元索引数组，
    域外（未命中任何单元）返回 -1。
    """
    V = fv.vertices
    C = fv.cells
    A = V[C[:, 0]]
    B = V[C[:, 1]]
    Dc = V[C[:, 2]]
    Dv = V[C[:, 3]]
    ref = _signed_vol(A, B, Dc, Dv)                         # 6*参考体积（>0）
    ref = np.where(np.abs(ref) > 1e-14, ref, 1.0)
    out = np.full(len(positions), -1, np.int64)
    for i, p in enumerate(positions):
        la = _signed_vol(p, B, Dc, Dv) / ref
        lb = _signed_vol(A, p, Dc, Dv) / ref
        lc = _signed_vol(A, B, p, Dv) / ref
        ld = _signed_vol(A, B, Dc, p) / ref
        ok = (la >= -tol) & (lb >= -tol) & (lc >= -tol) & (ld >= -tol)
        idx = np.where(ok)[0]
        if idx.size:
            out[i] = int(idx[0])
    return out


# ---------------------------------------------------------------------------
# 离散相求解器：DpmSolver
# ---------------------------------------------------------------------------
class DpmSolver:
    """DPM 离散相求解器：拉格朗日粒子注入、运动积分、位置定位与连续相耦合。

    `update(u, v, w)` 注入新 parcel、以自动限定的 dt 半隐式欧拉推进运动，返回归一化
    滑移残差；`coupling_source()` 暴露逐单元力密度供 SIMPLE 动量体源，`trajectories()`
    暴露各时刻粒子轨快照。P10 后端兼容门面 `_initialize_field() / step() /
    residual() / monitor_payload()`。
    """

    def __init__(self, fv, rho_p=DEFAULT_RHO_P, d_p=DEFAULT_DIA_P,
                 gravity=DEFAULT_GRAVITY, flow_axis=0, inlet_side="min",
                 outlet_side="max", inlet_velocity=None, inlet_speed=DEFAULT_SPEED,
                 mass_flow=DEFAULT_MASS_FLOW, parcels_per_step=DEFAULT_PARCELS,
                 rho_f=DEFAULT_DENSITY_F, mu_f=DEFAULT_MU_F, cfl=0.3, axis=0):
        self.fv = fv
        self.rho_p = float(rho_p)
        self.d_p = float(d_p)
        self.gravity = tuple(float(g) for g in gravity)
        self.flow_axis = int(flow_axis)
        self.inlet_side = inlet_side
        self.outlet_side = outlet_side
        self.cfl = float(cfl)
        self.axis = int(axis)
        if inlet_velocity is None:
            iv = [0.0, 0.0, 0.0]
            iv[self.flow_axis] = float(inlet_speed) * (1.0 if inlet_side == "min" else -1.0)
            self.inlet_velocity = tuple(iv)
        else:
            self.inlet_velocity = tuple(float(v) for v in inlet_velocity)
        self.mass_flow = float(mass_flow)
        self.parcels_per_step = int(parcels_per_step)
        # 粒子状态（p 个 parcel）
        self._pos = np.zeros((0, 3), float)
        self._vel = np.zeros((0, 3), float)
        self._dia = np.zeros(0, float)
        self._rho_p = np.zeros(0, float)
        self._mass = np.zeros(0, float)
        self._mass_flow = np.zeros(0, float)
        self._cell_id = np.zeros(0, np.int64)
        self._status = np.zeros(0, np.int64)
        self._history = []                 # 轨迹快照 list[dict(t,pos,vel,cell_id,status)]
        self._time = 0.0
        self._u = None
        self._v = None
        self._w = None
        self._grad_u = None
        self._grad_v = None
        self._grad_w = None
        # 连续相物性（cell 场）
        self._rho_f_cell = np.full(fv.n_cells, float(rho_f), float)
        self._mu_f_cell = np.full(fv.n_cells, float(mu_f), float)
        self._residual = 0.0
        self.iteration = 0
        self._inlet_faces = np.array([], np.int64)
        self._outlet_faces = np.array([], np.int64)
        self._wall_faces = np.array([], np.int64)
        self._classify_boundary()

    # -- 场属性 ------------------------------------------------
    @property
    def positions(self):
        return self._pos

    @property
    def velocities(self):
        return self._vel

    @property
    def cell_ids(self):
        return self._cell_id

    @property
    def status(self):
        return self._status

    @property
    def n_particles(self):
        return int(len(self._pos))

    @property
    def n_active(self):
        return int(np.sum(self._status == STATUS_ACTIVE))

    @property
    def n_escaped(self):
        return int(np.sum(self._status == STATUS_ESCAPED))

    def set_fluid(self, rho, mu):
        """设置连续相密度/粘度（标量或逐单元数组），供拖曳/浮力计算。"""
        self._rho_f_cell = np.full(
            self.fv.n_cells, float(rho), float) if np.isscalar(rho) \
            else np.asarray(rho, float).ravel()
        self._mu_f_cell = np.full(
            self.fv.n_cells, float(mu), float) if np.isscalar(mu) \
            else np.asarray(mu, float).ravel()
        return self

    def set_velocities(self, u, v, w):
        """设置连续相速度场并缓存其梯度（Taylor 重构供粒子插值）。"""
        self._u = np.asarray(u, float).ravel()
        self._v = np.asarray(v, float).ravel()
        self._w = np.asarray(w, float).ravel()
        fv = self.fv
        b = np.zeros(fv.n_faces, float)
        bnd = fv.is_boundary
        b[bnd] = self._u[fv.owner[bnd]]
        self._grad_u = fv.grad_gauss(self._u, boundary=b)
        b[bnd] = self._v[fv.owner[bnd]]
        self._grad_v = fv.grad_gauss(self._v, boundary=b)
        b[bnd] = self._w[fv.owner[bnd]]
        self._grad_w = fv.grad_gauss(self._w, boundary=b)
        return self

    # -- 边界分类 ------------------------------------------------
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

    # -- 粒子插值 ------------------------------------------------
    def _interp_velocity(self, p, cid):
        """粒子处连续相速度（Taylor 重构：U(p) = U_c + grad U·(p-x_c)）。"""
        n = len(p)
        U = np.zeros((n, 3), float)
        if n == 0:
            return U
        dx = p - self.fv.centroids[cid]
        gu = self._grad_u[cid]
        gv = self._grad_v[cid]
        gw = self._grad_w[cid]
        U[:, 0] = self._u[cid] + (gu[:, 0] * dx[:, 0] + gu[:, 1] * dx[:, 1]
                                  + gu[:, 2] * dx[:, 2])
        U[:, 1] = self._v[cid] + (gv[:, 0] * dx[:, 0] + gv[:, 1] * dx[:, 1]
                                  + gv[:, 2] * dx[:, 2])
        U[:, 2] = self._w[cid] + (gw[:, 0] * dx[:, 0] + gw[:, 1] * dx[:, 1]
                                  + gw[:, 2] * dx[:, 2])
        return U

    # -- 入口注入 ------------------------------------------------
    def inject(self):
        """在当前入口面注入一批 parcel（进入域内并定位所在单元）。"""
        if len(self._inlet_faces) == 0:
            return
        n_inj = self.parcels_per_step
        faces = self._inlet_faces
        pick = np.array([faces[i % len(faces)] for i in range(n_inj)], np.int64)
        fc = self.fv.face_centroid[pick]
        nrm = self.fv.face_normal[pick]
        nrm_norm = np.maximum(_safe_norm(nrm, axis=1), EPS)
        inward = (-nrm / nrm_norm[:, None]) * INJ_OFFSET
        pos = fc + inward
        vel = np.broadcast_to(np.asarray(self.inlet_velocity, float), (n_inj, 3))
        dia = np.full(n_inj, self.d_p, float)
        rho_p = np.full(n_inj, self.rho_p, float)
        mass = rho_p * np.pi / 6.0 * dia ** 3
        mf = np.full(n_inj, self.mass_flow / max(n_inj, 1), float)
        cid = locate_cells(self.fv, pos)
        self._pos = np.concatenate([self._pos, pos], axis=0)
        self._vel = np.concatenate([self._vel, vel], axis=0)
        self._dia = np.concatenate([self._dia, dia])
        self._rho_p = np.concatenate([self._rho_p, rho_p])
        self._mass = np.concatenate([self._mass, mass])
        self._mass_flow = np.concatenate([self._mass_flow, mf])
        self._cell_id = np.concatenate([self._cell_id, cid])
        self._status = np.concatenate([self._status, np.where(
            cid >= 0, STATUS_ACTIVE, STATUS_ESCAPED).astype(np.int64)])
        return n_inj

    # -- 自动时间步 ------------------------------------------------
    def auto_dt(self, dt=None):
        """由松弛时间与 CFL 限定的保守推进步长。dt 显式给定时优先。"""
        if dt is not None:
            return max(float(dt), 1e-8)
        fv = self.fv
        u_ref = 1.0
        if self._u is not None and len(self._u):
            u_ref = max(float(_safe_norm(self._u)), 1e-6)
            if self._v is not None and len(self._v):
                u_ref = max(u_ref, float(_safe_norm(self._v)))
            if self._w is not None and len(self._w):
                u_ref = max(u_ref, float(_safe_norm(self._w)))
        dx_min = float(np.min(fv.volumes)) ** (1.0 / 3.0)
        tau = self._gamma_minimum()
        dt_cfl = self.cfl * dx_min / max(u_ref, 1e-6)
        dt_tau = 0.5 * tau if np.isfinite(tau) and tau > 0 else 1e-3
        return max(min(dt_cfl, dt_tau), 1e-8)

    def _gamma_minimum(self):
        """最小拖曳松弛时间（对当前 active 粒子），无粒子时返回 1。"""
        act = self._status == STATUS_ACTIVE
        if not act.any():
            return 1.0
        d = self._dia[act]
        rp = self._rho_p[act]
        mu = self._mu_f_cell[self._cell_id[act]]
        tau = np.maximum(rp * np.maximum(d, EPS) ** 2,
                         1e-30) / (18.0 * np.maximum(mu, EPS))
        t = float(np.min(tau))
        return t if np.isfinite(t) and t > 0 else 1.0

    # -- 运动推进 ------------------------------------------------
    def advance(self, dt):
        """半隐式欧拉推进 active 粒子运动一步，并定位/标记离开域粒子。"""
        if dt <= 0.0:
            return
        act = self._status == STATUS_ACTIVE
        if not act.any():
            return
        idx = np.where(act)[0]
        p = self._pos[idx]
        v = self._vel[idx]
        cid = self._cell_id[idx]
        rho_f = self._rho_f_cell[cid]
        mu_f = self._mu_f_cell[cid]
        Uf = self._interp_velocity(p, cid)
        dv = Uf - v
        spd = _safe_norm(dv, axis=1)
        re = np.maximum(rho_f * spd * self._dia[idx] / np.maximum(mu_f, EPS), EPS)
        cd = _drag_coefficient(re)
        tau = (self._rho_p[idx] * np.maximum(self._dia[idx], EPS) ** 2
               / (18.0 * np.maximum(mu_f, EPS))
               * (24.0 / np.maximum(cd * re, EPS)))
        a_drag = dv / np.maximum(tau[:, None], EPS)
        g = np.asarray(self.gravity, float)
        g_term = g[None, :] * ((self._rho_p[idx] - rho_f)
                               / np.maximum(self._rho_p[idx], EPS))[:, None]
        a = a_drag + g_term
        v_new = v + dt * a
        x_new = p + dt * v_new
        self._vel[idx] = v_new
        self._pos[idx] = x_new
        new_cid = locate_cells(self.fv, x_new)
        self._cell_id[idx] = new_cid
        self._status[idx] = np.where(new_cid < 0, STATUS_ESCAPED, STATUS_ACTIVE)
        return

    def update(self, u, v, w, dt=None):
        """注入新 parcel 并推进一个自动步长，返回归一化滑移残差。"""
        self.set_velocities(u, v, w)
        self.inject()
        if self.n_active == 0:
            self._residual = 0.0
            return 0.0
        d = self.auto_dt(dt)
        self.advance(d)
        self._snapshot()
        self.iteration += 1
        self._time += d
        self._residual = self._slip_residual()
        return self._residual

    def _snapshot(self):
        self._history.append(dict(
            t=self._time,
            pos=np.asarray(self._pos, float).copy(),
            vel=np.asarray(self._vel, float).copy(),
            cell_id=np.asarray(self._cell_id, np.int64).copy(),
            status=np.asarray(self._status, np.int64).copy()))

    def _slip_residual(self):
        """归一化滑移残差：加权 |U_f - U_p|^2 与加权 |U_f|^2 的 RMS 比。"""
        act = self._status == STATUS_ACTIVE
        if not act.any():
            return 0.0
        idx = np.where(act)[0]
        p = self._pos[idx]
        cid = self._cell_id[idx]
        Uf = self._interp_velocity(p, cid)
        sl = self._vel[idx] - Uf
        wm = np.maximum(self._mass[idx], EPS)
        num = float(np.sqrt(np.sum(wm * (sl * sl).sum(axis=1))))
        den = float(np.sqrt(np.sum(wm * (Uf * Uf).sum(axis=1))))
        return float(num) / max(den, EPS)

    # -- 连续相耦合 ------------------------------------------------
    def coupling_source(self):
        """逐单元力密度（n_cells, 3）：粒子对流体反作用力 = -Σ m_p (U_f-U_p)/τ / V_c。"""
        fv = self.fv
        S = np.zeros((fv.n_cells, 3), float)
        act = self._status == STATUS_ACTIVE
        if not act.any():
            return S
        idx = np.where(act)[0]
        p = self._pos[idx]
        cid = self._cell_id[idx]
        rho_f = self._rho_f_cell[cid]
        mu_f = self._mu_f_cell[cid]
        Uf = self._interp_velocity(p, cid)
        dv = Uf - self._vel[idx]
        spd = _safe_norm(dv, axis=1)
        re = np.maximum(rho_f * spd * self._dia[idx] / np.maximum(mu_f, EPS), EPS)
        cd = _drag_coefficient(re)
        tau = (self._rho_p[idx] * np.maximum(self._dia[idx], EPS) ** 2
               / (18.0 * np.maximum(mu_f, EPS))
               * (24.0 / np.maximum(cd * re, EPS)))
        force_on_particle = self._mass[idx][:, None] * dv / np.maximum(
            tau[:, None], EPS)
        cell_force = np.zeros((fv.n_cells, 3), float)
        np.add.at(cell_force, cid, -force_on_particle)
        vol = np.where(fv.volumes > 1e-14, fv.volumes, 1.0)
        S = cell_force / vol[:, None]
        return S

    def trajectories(self):
        """返回粒子轨快照列表（含 t/pos/vel/cell_id/status）。"""
        return self._history

    # -- P10 后端兼容门面 ----------------------------------------
    def _initialize_field(self):
        if len(self._inlet_faces) == 0:
            self._classify_boundary()
        self._history = []
        self._time = 0.0
        self._residual = 0.0
        self.iteration = 0
        return dict(n_particles=self.n_particles, n_active=self.n_active,
                    n_escaped=self.n_escaped)

    def step(self):
        if self._u is None or self._v is None or self._w is None:
            self._residual = 0.0
            return dict(residual=0.0, n_particles=self.n_particles)
        resid = float(self.update(self._u, self._v, self._w))
        return dict(residual=resid, n_particles=self.n_particles,
                    n_active=self.n_active, n_escaped=self.n_escaped,
                    max_speed=float(self.max_speed()))

    def residual(self):
        return self._residual

    def max_speed(self):
        act = self._status == STATUS_ACTIVE
        if not act.any():
            return 0.0
        return float(np.max(_safe_norm(self._vel[act], axis=1)))

    def monitor_payload(self):
        act = self._status == STATUS_ACTIVE
        if act.any():
            mean_pos = self._pos[act].mean(axis=0)
        else:
            mean_pos = np.zeros(3, float)
        return dict(n_particles=self.n_particles, n_active=self.n_active,
                    n_escaped=self.n_escaped, max_speed=self.max_speed(),
                    mean_x=float(mean_pos[0]), mean_y=float(mean_pos[1]),
                    mean_z=float(mean_pos[2]),
                    dia_min=float(np.min(self._dia)) if len(self._dia) else self.d_p,
                    dia_max=float(np.max(self._dia)) if len(self._dia) else self.d_p,
                    coupling_max=float(np.max(_safe_norm(
                        self.coupling_source(), axis=1)))
                    if act.any() and self.n_active else 0.0,
                    time=self._time, iteration=self.iteration,
                    residual=self._residual)


# ---------------------------------------------------------------------------
# 工厂注册表
# ---------------------------------------------------------------------------
_DPM_MODELS = {
    "dpm": DpmSolver,
    "discrete_phase": DpmSolver,
    "lagrangian": DpmSolver,
}


def make_dpm(fv, model="dpm", **kwargs):
    """按字符串名实例化 DPM 离散相模型（大小写/下划线不敏感）。"""
    key = str(model).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _DPM_MODELS:
        names = ", ".join(sorted(set(_DPM_MODELS.keys())))
        raise ValueError("未知 DPM 离散相模型 %r（支持：%s）" % (model, names))
    return _DPM_MODELS[key](fv, **kwargs)

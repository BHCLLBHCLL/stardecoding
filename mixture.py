# -*- coding: utf-8 -*-
"""P 波 P8：多相 Mixture —— 多相体积分数输运 + 代数滑移（drift-flux）封闭 + 混合物性。

纯 numpy 实现（occ / scdm 两环境皆可用），建立在 P4 有限体积离散核心
（`fvm_core.FVM`）之上，为多相混合物（相 0 连续相 + 相 1..N-1 弥散相）提供
体积分数 α_k 的守恒输运与代数滑移（drift-flux）封闭，是 VOF（两相）+ 多相
推广的中间层次。

物理框架（Mixture / drift-flux）：
    ∂α_k/∂t + ∇·(α_k u_m) = -∇·(α_k u_dr,k)   （弥散相 k 体积分数守恒）
    ρ_m = Σ_k α_k ρ_k；μ_m = Σ_k α_k μ_k        （混合物性）
    u_k = u_m + u_dr,k                          （漂移速度封闭）
    u_dr,k = u_slip,k - Σ_j α_j u_slip,j       （Σ_k α_k u_dr,k = 0 保总体积守恒）
    u_slip,k = 由 Schiller-Naumann 拖曳平衡求得的滑移速度（浮力导致，达标定）

核心能力：

  1) 物性混合：`blend_nphase` 加权叠加，`mixture_rho` / `mixture_mu`。
  2) 拖曳系数：`drag_coefficient`（Schiller-Naumann 分段）。
  3) 滑移速度：`slip_velocity`（浮力-拖曳平衡达标定，Stokes 起步 + 迭代）。
  4) 漂移速度：`drift_velocities`（代数 drift-flux 封闭，Σ_k α_k u_dr,k = 0）。
  5) 守恒输运：`advance_mixture`（混合速度上风对流 + 漂移通量 + 有界裁剪保 α∈[0,1]）。
  6) 求解器：`MixtureSolver` 提供 `update(u,v,w,mdot)` 推进各弥散相体积分数并暴露
     混合密度（`rho`）/混合粘度（`mu`）/滑移（`slip`）/漂移（`drift`）；P10 后端
     兼容门面 `_initialize_field() / step() / residual() / monitor_payload()`。

纯 numpy 约束：所有范数走 `solver_run._safe_norm`，不新增 scipy 依赖。
"""
import numpy as np

from fvm_core import FVM
from solver_run import _safe_norm
import vof as _vof


# ---------------------------------------------------------------------------
# 多相 Mixture 常量
# ---------------------------------------------------------------------------
DEFAULT_RHOS = [998.0, 1.18, 800.0]      # 相 0 水 / 相 1 气 / 相 2 油（kg/m^3）
DEFAULT_MUS = [1.0e-3, 1.8e-5, 5.0e-3]    # 相 0 水 / 相 1 气 / 相 2 油（Pa·s）
DEFAULT_DIAMETERS = [0.0, 5.0e-4, 1.0e-3]  # 相 0 连续相无粒径 / 气 / 油（m）
DEFAULT_GRAVITY = (0.0, 0.0, -9.81)
EPS = 1e-12
MAX_SLIP_ITER = 30        # 滑移速度拖曳平衡迭代上限


# ---------------------------------------------------------------------------
# 多相物性混合
# ---------------------------------------------------------------------------
def blend_nphase(alphas, props):
    """N 相物性线性混合：prop = Σ_k α_k prop_k（α 逐单元，props 逐相）。"""
    alphas = np.asarray(alphas, float)
    props = np.asarray(props, float)
    return np.sum(alphas * props, axis=-1)


def mixture_rho(alphas, rhos=DEFAULT_RHOS):
    """N 相密度场（逐单元）。"""
    return blend_nphase(np.asarray(alphas, float), np.asarray(rhos, float))


def mixture_mu(alphas, mus=DEFAULT_MUS):
    """N 相动力粘度场（逐单元）。"""
    return blend_nphase(np.asarray(alphas, float), np.asarray(mus, float))


# ---------------------------------------------------------------------------
# 拖曳系数 / 滑移速度
# ---------------------------------------------------------------------------
def drag_coefficient(re):
    """Schiller-Naumann 单球拖曳系数：C_D = 24/Re (1+0.15 Re^0.687) 在 Re<1000，
    否则 C_D = 0.44（牛顿区）。re 为标量或数组；re≈0 由调用方夹取避免除零。
    """
    # re 夹取到 EPS 下限，避免 re=0 时 np.where 两分支同时求值产生除零 inf 告警
    # （调用方本已夹取 re≥EPS，此处为公共函数自防御）
    re = np.maximum(np.asarray(re, float), EPS)
    cd = np.where(re < 1000.0, 24.0 / re * (1.0 + 0.15 * re ** 0.687), 0.44)
    return cd


def slip_velocity(d, rho_p, rho_m, mu_m, gvec=DEFAULT_GRAVITY,
                  max_iter=MAX_SLIP_ITER):
    """弥散相滑移速度（相 p 在混合物 m 中因浮力-拖曳平衡的达标定速度）。

    量纲：v = (ρ_p - ρ_m) d² g /(18 μ_m)（Stokes 起步），再以 Schiller-Naumann
    迭代 `(2/3) (ρ_p-ρ_m) g d /(ρ_m C_D)` 修正到高 Re。方向沿重力方向（ρ_p<ρ_m 时
    上浮、> 时下沉）。`rho_m` / `mu_m` 为逐单元数组，`d` / `rho_p` 为弥散相标量，
    返回 (n_cells, 3)。重力近零时返回零速度场。
    """
    g = np.asarray(gvec, float).ravel()
    gmag = _safe_norm(g)
    rho_m = np.asarray(rho_m, float).ravel()
    mu_m = np.asarray(mu_m, float).ravel()
    out = np.zeros(rho_m.shape + (3,), float)
    if gmag <= 1e-12:
        return out
    ghat = g / gmag
    d = float(d)
    rho_p = float(rho_p)
    diff = rho_p - rho_m
    v = diff * d * d * gmag / (18.0 * np.maximum(mu_m, EPS))
    sign = np.sign(diff)
    for _ in range(int(max_iter)):
        re = np.maximum(rho_m * np.abs(v) * d / np.maximum(mu_m, EPS), EPS)
        cd = drag_coefficient(re)
        cd = np.where(np.abs(v) < EPS, 24.0 / np.maximum(re, EPS), cd)
        v2 = (4.0 / 3.0) * diff * gmag * d / np.maximum(rho_m * cd, EPS)
        v_new = sign * np.sqrt(np.maximum(v2, 0.0))
        if np.allclose(v_new, v, rtol=1e-6, atol=1e-12):
            v = v_new
            break
        v = v_new
    out[:, :] = v[:, None] * ghat[None, :]
    return out


def drift_velocities(alphas, slip):
    """代数 drift-flux 漂移速度矩阵 u_dr,k = u_slip,k - Σ_j α_j u_slip,j。

    保证 Σ_k α_k u_dr,k = 0（总体积漂移通量守恒，配合上风对流使相体积分数
    无源守恒）。`alphas` (n_cells, n_phases)，`slip` (n_cells, n_phases, 3)；
    连续相（相 0）滑移为 0，漂移由弥散相滑移反推。返回 (n_cells, n_phases, 3)。
    """
    alphas = np.asarray(alphas, float)
    slip = np.asarray(slip, float)
    w = alphas[:, :, None] * slip       # (n_cells, n_phases, 3)
    s = np.sum(w, axis=1)[:, None, :]   # (n_cells, 1, 3) = Σ_j α_j u_slip,j
    return slip - s


# ---------------------------------------------------------------------------
# α 守恒输运（混合速度上风对流 + 漂移通量 + 有界裁剪）
# ---------------------------------------------------------------------------
def _face_accumulate(fv, F):
    """面通量累积到单元（owner 加、neighbor 减），返回 (n_cells,)。"""
    F = np.asarray(F, float)
    is_int = fv.neighbor >= 0
    nbr = np.where(is_int, fv.neighbor, 0)
    acc = np.zeros(fv.n_cells, float)
    np.add.at(acc, fv.owner, F)
    np.add.at(acc, nbr[is_int], -F[is_int])
    return acc


def volume_flux(fv, u, v, w):
    """由速度场重构面体积通量：Flux = (u·n) * A（供 α 输运）。"""
    uf = fv.face_value(np.asarray(u, float))
    vf = fv.face_value(np.asarray(v, float))
    wf = fv.face_value(np.asarray(w, float))
    vn = (uf * fv.face_normal[:, 0] + vf * fv.face_normal[:, 1]
          + wf * fv.face_normal[:, 2])
    return vn * fv.face_area


def face_upwind_alphas(fv, mdot, alphas, boundary=None):
    """各相上风面值（n_faces, n_phases）：面通量方向决定取上游单元 α。

    内部面 mdot>=0 取 owner（上游），否则取 neighbor；边界面 mdot<0（入流）
    取 `boundary[f]`（缺省 = owner 外推，即零梯度出流）。
    """
    mdot = np.asarray(mdot, float).ravel()
    is_int = fv.neighbor >= 0
    nbr = np.where(is_int, fv.neighbor, fv.owner)
    up_idx = np.where(mdot >= 0.0, fv.owner, nbr)
    up = alphas[up_idx]                     # (n_faces, n_phases)
    if boundary is not None:
        inflow = (~is_int) & (mdot < 0.0)
        up[inflow] = np.asarray(boundary, float)[inflow]
    return up


def advance_mixture(fv, mdot, alphas, drift, dt, boundary=None, max_iter=8):
    """显式推进 N 相体积分数一个时间步 Δt（弥散相 1..N-1，连续相 0 由 1-Σ 补足）。

    相 k（弥散）对流通量 F = mdot·α_k,upwind（上风）+ 漂移通量
    `α_k,face * (u_dr,k·n) * A`；推进后把弥散相投影到单形（各 α_k∈[0,1] 且
    Σ_k α_k ≤ 1），连续相 α_0 = 1 - Σ_{k≥1} α_k。返回 α_new（n_cells, n_phases）
    与实测有效时间步（CFL 受限时 < dt）。
    """
    mdot = np.asarray(mdot, float)
    alphas = np.asarray(alphas, float)
    drift = np.asarray(drift, float)
    nph = alphas.shape[1]
    disp = np.arange(1, nph)
    vol = np.where(fv.volumes > 1e-14, fv.volumes, 1.0)
    a_new = alphas.copy()
    dt_eff = float(dt)
    for _ in range(int(max_iter)):
        divs = []
        for k in disp:
            up = face_upwind_alphas(fv, mdot, alphas, boundary=boundary)[:, k]
            conv = mdot * up
            af = fv.face_value(alphas[:, k])
            drif = drift[:, k, :]
            df = np.stack([fv.face_value(drif[:, i]) for i in range(3)], axis=1)
            drift_flux = af * (df[:, 0] * fv.face_normal[:, 0]
                               + df[:, 1] * fv.face_normal[:, 1]
                               + df[:, 2] * fv.face_normal[:, 2])
            drift_flux = drift_flux * fv.face_area
            divs.append(_face_accumulate(fv, conv + drift_flux) / vol)
        divs = np.stack(divs, axis=1)           # (n_cells, n_disp)
        a_new[:, disp] = alphas[:, disp] - dt_eff * divs
        # 半隐式有界：若单步越界则减半时间步重试
        lo = (a_new[:, disp] < -1e-12).any()
        hi = (a_new[:, disp] > 1.0 + 1e-12).any()
        if not (lo or hi):
            break
        dt_eff = 0.5 * dt_eff
    a_new[:, disp] = np.clip(a_new[:, disp], 0.0, 1.0)
    s = np.sum(a_new[:, disp], axis=1)
    over = s > 1.0
    if over.any():
        a_new[over, disp] /= s[over, None]
    a_new[:, disp] = np.clip(a_new[:, disp], 0.0, 1.0)
    a_new[:, 0] = 1.0 - np.sum(a_new[:, disp], axis=1)
    a_new = np.clip(a_new, 0.0, 1.0)
    return a_new, dt_eff


# ---------------------------------------------------------------------------
# 多相 Mixture 求解器：MixtureSolver
# ---------------------------------------------------------------------------
class MixtureSolver:
    """多相 Mixture 求解器：N 相体积分数守恒输运 + 代数滑移封闭 + 混合物性。

    `update(u, v, w, mdot)` 由给定混合流场推进各弥散相体积分数（上风 + 漂移通量 +
    有界投影），更新混合物性与滑移/漂移速度，返回归一化残差。`rho` / `mu` 暴露
    逐单元混合密度与动力粘度（供 SIMPLE 动量装配），`slip` / `drift` / `alphas`
    暴露相滑移速度、漂移速度与体积分数矩阵。
    """

    def __init__(self, fv, rhos=DEFAULT_RHOS, mus=DEFAULT_MUS,
                 diameters=DEFAULT_DIAMETERS, gravity=DEFAULT_GRAVITY,
                 flow_axis=0, inlet_side="min", outlet_side="max",
                 inlet_alphas=None, alpha0=None, relax=0.7, max_iter=50,
                 cfl=0.4, axis=0):
        self.fv = fv
        self.rhos = np.asarray(rhos, float)
        self.mus = np.asarray(mus, float)
        dd = np.asarray(diameters, float).ravel()
        nph = len(self.rhos)
        if len(self.mus) != nph:
            raise ValueError("Mixture rhos/mus 长度不一致：%d vs %d"
                             % (nph, len(self.mus)))
        self.diameters = np.zeros(nph, float)
        if len(dd) == nph:
            self.diameters[:] = dd
        elif len(dd) == 0:
            pass
        else:
            raise ValueError("Mixture 粒径长度应为 %d 或 0" % nph)
        self.gravity = tuple(float(g) for g in gravity)
        self.flow_axis = int(flow_axis)
        self.inlet_side = inlet_side
        self.outlet_side = outlet_side
        self.relax = float(relax)
        self.max_iter = int(max_iter)
        self.cfl = float(cfl)
        self.axis = int(axis)
        self._disp = np.arange(1, nph)
        # 入口 / 初值弥散相体积分数
        ia = np.zeros(nph - 1, float) if inlet_alphas is None \
            else np.asarray(inlet_alphas, float).ravel()
        if len(ia) != nph - 1:
            raise ValueError("Mixture 入口相分数应长度 %d" % (nph - 1))
        self.inlet_alphas = np.clip(ia, 0.0, 1.0)
        a0 = np.zeros(nph - 1, float) if alpha0 is None \
            else np.asarray(alpha0, float).ravel()
        if len(a0) != nph - 1:
            raise ValueError("Mixture 初值相分数应长度 %d" % (nph - 1))
        a0 = np.clip(a0, 0.0, 1.0)
        if np.sum(a0) > 1.0:
            a0 = a0 / np.sum(a0)
        self._alphas = np.zeros((fv.n_cells, nph), float)
        self._alphas[:, 0] = 1.0
        self._alphas[:, self._disp] = a0[None, :]
        self._alphas[:, 0] = 1.0 - np.sum(a0)
        self._slip = np.zeros((fv.n_cells, nph, 3), float)
        self._drift = np.zeros((fv.n_cells, nph, 3), float)
        self._residual = 0.0
        self.iteration = 0
        self._u = np.zeros(fv.n_cells, float)
        self._v = np.zeros(fv.n_cells, float)
        self._w = np.zeros(fv.n_cells, float)
        self._mdot = None
        self._inlet_faces = np.array([], np.int64)
        self._outlet_faces = np.array([], np.int64)
        self._wall_faces = np.array([], np.int64)
        self._classify_boundary()
        self._update_phase_kinematics()

    # -- 场属性 ------------------------------------------------
    @property
    def alphas(self):
        return self._alphas

    @property
    def n_phases(self):
        return self._alphas.shape[1]

    @property
    def rho(self):
        """逐单元混合密度场。"""
        return mixture_rho(self._alphas, self.rhos)

    @property
    def mu(self):
        """逐单元混合动力粘度场。"""
        return mixture_mu(self._alphas, self.mus)

    @property
    def slip(self):
        return self._slip

    @property
    def drift(self):
        return self._drift

    def phase_volume(self, k):
        """相 k 体积（α_k·V 逐单元累加）。"""
        return float(np.sum(self._alphas[:, k] * self.fv.volumes))

    @property
    def wall_volume(self):
        """边界壁面单元数（诊断用）。"""
        return int(len(self._wall_faces))

    def set_velocities(self, u, v, w, mdot=None):
        """注入当前流场与面混合体积通量，供 `step()` 推进 α。"""
        self._u = np.asarray(u, float).ravel()
        self._v = np.asarray(v, float).ravel()
        self._w = np.asarray(w, float).ravel()
        self._mdot = None if mdot is None else np.asarray(mdot, float)

    def set_inlet_alphas(self, inlet_alphas):
        """更新入口弥散相体积分数。"""
        ia = np.asarray(inlet_alphas, float).ravel()
        if len(ia) != self.n_phases - 1:
            raise ValueError("Mixture 入口相分数应长度 %d" % (self.n_phases - 1))
        self.inlet_alphas = np.clip(ia, 0.0, 1.0)
        return self

    # -- 内部辅助 ------------------------------------------------
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

    def _boundary_alphas(self):
        """构造边界面 N 相 α 值：弥散相入口 = inlet_alphas，其余 = owner 外推。"""
        fv = self.fv
        b = np.zeros((fv.n_faces, self.n_phases), float)
        b[:] = self._alphas[fv.owner]
        if len(self._inlet_faces):
            b[self._inlet_faces, 1:] = self.inlet_alphas[None, :]
        return b

    def _slip_per_phase(self):
        """逐弥散相滑移速度（n_cells, n_disp, 3），由混合密度/粘度 + 粒径算得。"""
        n = self.fv.n_cells
        rm = self.rho
        mm = self.mu
        out = np.zeros((n, len(self._disp), 3), float)
        for j, k in enumerate(self._disp):
            out[:, j, :] = slip_velocity(
                self.diameters[k], self.rhos[k], rm, mm,
                self.gravity, max_iter=MAX_SLIP_ITER)
        return out

    def _update_phase_kinematics(self):
        """由当前相分数更新滑移速度场与漂移速度场。"""
        sp = np.zeros((self.fv.n_cells, self.n_phases, 3), float)
        sdisp = self._slip_per_phase()
        sp[:, 1:, :] = sdisp
        self._slip = sp
        self._drift = drift_velocities(self._alphas, sp)

    def update(self, u, v, w, mdot=None):
        """由混合流场推进一次各弥散相显式输运，更新 α 与滑移/漂移速度。

        返回归一化残差，衡量本次推进 α 的变化量（Δα 对 ||α||）。
        """
        fv = self.fv
        alphas_old = np.asarray(self._alphas, float)
        if mdot is None:
            mdot = volume_flux(fv, u, v, w)
        mdot = np.asarray(mdot, float)
        if len(self._inlet_faces) == 0:
            self._classify_boundary()
        acc = self._flux_magnitude(mdot)
        dt_arr = float(self.cfl) * fv.volumes / np.maximum(acc, 1e-30)
        dt = float(np.min(dt_arr)) if dt_arr.size else 0.0
        if not np.isfinite(dt) or dt <= 0.0:
            dt = 1e-6
        a_new, _dt_eff = advance_mixture(
            fv, mdot, alphas_old, self._drift, dt,
            boundary=self._boundary_alphas())
        denom = max(float(_safe_norm(a_new)), EPS)
        resid = float(_safe_norm(a_new - alphas_old)) / denom
        self._alphas = a_new
        self._update_phase_kinematics()
        self.iteration += 1
        self._residual = resid
        return resid

    def _flux_magnitude(self, mdot):
        """逐单元面通量绝对值累积（对流 |mdot| + 各弥散相漂移通量，供 CFL dt）。"""
        fv = self.fv
        mdot = np.asarray(mdot, float)
        face_mag = np.abs(mdot).copy()
        if self._drift is not None and self._drift.size:
            for k in range(1, self.n_phases):
                af = fv.face_value(self._alphas[:, k])
                drif = self._drift[:, k, :]
                df = np.stack([fv.face_value(drif[:, i])
                               for i in range(3)], axis=1)
                dfn = (df[:, 0] * fv.face_normal[:, 0]
                       + df[:, 1] * fv.face_normal[:, 1]
                       + df[:, 2] * fv.face_normal[:, 2])
                face_mag = face_mag + np.abs(af * dfn * fv.face_area)
        is_int = fv.neighbor >= 0
        nbr = np.where(is_int, fv.neighbor, 0)
        acc = np.zeros(fv.n_cells, float)
        np.add.at(acc, fv.owner, face_mag)
        np.add.at(acc, nbr[is_int], face_mag[is_int])
        return acc

    # -- P10 后端兼容门面 ----------------------------------------
    def _initialize_field(self):
        if len(self._inlet_faces) == 0:
            self._classify_boundary()
        self._update_phase_kinematics()
        self._residual = 0.0
        self.iteration = 0
        return dict(n_phases=self.n_phases,
                    alpha_min=float(self._alphas.min()),
                    alpha_max=float(self._alphas.max()),
                    rho_min=float(self.rho.min()), rho_max=float(self.rho.max()))

    def step(self):
        resid = float(self.update(self._u, self._v, self._w, mdot=self._mdot))
        self._residual = resid
        return dict(residual=resid, n_phases=self.n_phases,
                    alpha_min=float(self._alphas.min()),
                    alpha_max=float(self._alphas.max()))

    def residual(self):
        return self._residual

    def monitor_payload(self):
        return dict(n_phases=self.n_phases,
                    alpha_min=float(self._alphas.min()),
                    alpha_max=float(self._alphas.max()),
                    rho_min=float(self.rho.min()), rho_max=float(self.rho.max()),
                    phase_volumes=[self.phase_volume(k)
                                   for k in range(self.n_phases)],
                    iteration=self.iteration, residual=self._residual)


# ---------------------------------------------------------------------------
# 工厂注册表
# ---------------------------------------------------------------------------
_MIXTURE_MODELS = {
    "mixture": MixtureSolver,
    "mix": MixtureSolver,
    "drift_flux": MixtureSolver,
}


def make_mixture(fv, model="mixture", **kwargs):
    """按字符串名实例化多相 Mixture 模型（大小写/下划线不敏感）。"""
    key = str(model).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _MIXTURE_MODELS:
        names = ", ".join(sorted(set(_MIXTURE_MODELS.keys())))
        raise ValueError("未知多相 Mixture 模型 %r（支持：%s）" % (model, names))
    return _MIXTURE_MODELS[key](fv, **kwargs)

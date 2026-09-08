# -*- coding: utf-8 -*-
"""P 波 P10：多相欧拉-欧拉（Eulerian-Eulerian）—— 相体积分数守恒输运 + 相间作用力
（拖曳 / 升力 / 虚拟质量 / 壁面润滑）+ 群体平衡（聚并 / 破碎 / 成核）。

纯 numpy 实现（occ / scdm 两环境皆可用），建立在 P4 有限体积离散核心
（`fvm_core.FVM`）与 P8 多相 Mixture（`mixture.py`，drift-flux 代数滑移）之上。
相对 Mixture（仅浮力-拖曳代数滑移封闭），欧拉-欧拉引入**完整的相间作用力平衡**：
弥散相 k 的相对速度不再只由浮力-拖曳决定，而是把 升力 / 虚拟质量 / 壁面润滑
作为附加体积力加入 `u_rel,k = u_slip,k + F_other,k / K_drag,k`，再经
`drift_velocities` 投影（保证 Σ_k α_k u_dr,k = 0）得到逐相漂移速度；同时以
群体平衡核（聚并 / 破碎 / 成核）作为相分数的源项。该模型对应
`star.multiphase`（EulerianPhase + PhaseInteraction + Population Balance）谱系。

物理框架（欧拉-欧拉 / 代数滑移封闭 + 相间作用力）：
    ∂α_k/∂t + ∇·(α_k u_m) = -∇·(α_k u_dr,k) + S_pb,k   （相 k 体积分数守恒）
    ρ_m = Σ_k α_k ρ_k；μ_m = Σ_k α_k μ_k                  （混合物性）
    u_k = u_m + u_dr,k                                    （逐相速度封闭）
    u_dr,k = u_rel,k - Σ_j α_j u_rel,j                    （Σ_k α_k u_dr,k = 0）
    u_rel,k = u_slip,k + F_lift,k + F_vm,k + F_wl,k / K_k （相间作用力平衡）

核心能力：

  1) 拖曳交换系数：`drag_exchange`/`interphase_drag`（Schiller-Naumann + Wen-Yu
     空泡率修正 f = α_c^-2.65，低 Re 回归 Stokes 系数 18μ α_d/d²）。
  2) 升力：`lift_force`（Legendre-Magnaudet 简式，-C_L α_d ρ_c u_rel × (∇×u_m)）。
  3) 虚拟质量力：`virtual_mass_force`（C_vm α_d ρ_c (u_m·∇)u_m）。
  4) 壁面润滑：`wall_lubrication_force`（Antal：C_wl = max(0, Cw1 + Cw2 d/y_w)，
     近壁把相推离壁面）。
  5) 群体平衡：`coalescence_kernel` / `breakup_kernel` / `nucleation_rate` +
     `population_balance_source`（把质量转移到较大/较小/最小相，源 Σ_k S_k = 0）。
  6) 求解器：`EulerianMultiphaseSolver` 提供 `update(u,v,w,mdot)` 推进各相体积分数
     与逐相速度，暴露混合密度/粘度/逐相速度/相间力；P10 后端兼容门面
     `_initialize_field() / step() / residual() / monitor_payload()`。

纯 numpy 约束：所有范数走 `solver_run._safe_norm`，不新增 scipy 依赖；拖曳/升力/
虚拟质量/壁面润滑/群体平衡核均可独立开关（默认仅拖曳 + 可选升力开启，虚拟质量/
壁面润滑/群体平衡默认关闭以保证数值稳定与确定性的测试）。
"""
import numpy as np

from fvm_core import FVM
from solver_run import _safe_norm
from mixture import (
    blend_nphase, mixture_rho, mixture_mu, drag_coefficient, slip_velocity,
    drift_velocities, volume_flux, advance_mixture, _face_accumulate,
    EPS, MAX_SLIP_ITER, DEFAULT_RHOS, DEFAULT_MUS, DEFAULT_DIAMETERS,
    DEFAULT_GRAVITY,
)


# ---------------------------------------------------------------------------
# 欧拉-欧拉多相常量
# ---------------------------------------------------------------------------
DEFAULT_LIFT_COEF = 0.5          # Legendre-Magnaudet 升力系数
DEFAULT_VIRTUAL_MASS_COEF = 0.5  # 虚拟质量力系数（附加质量）
DEFAULT_WALL_LUB_COEF_CW1 = -0.01   # Antal 壁面润滑常数 Cw1
DEFAULT_WALL_LUB_COEF_CW2 = 0.05    # Antal 壁面润滑常数 Cw2
DEFAULT_COALESCENCE_RATE = 0.0    # 聚并核常数（1/s）
DEFAULT_BREAKUP_RATE = 0.0        # 破碎核常数（1/s）
DEFAULT_NUCLEATION_RATE = 0.0     # 成核核常数（1/s）
MAX_INTERACTION_ITER = 12          # 相间作用力平衡迭代上限
MAX_WALL_FACES = 4096              # 壁面润滑最近壁面搜索上限


# ---------------------------------------------------------------------------
# 相间作用力：拖曳交换系数
# ---------------------------------------------------------------------------
def drag_exchange(d, rho_c, mu_c, umag, alpha_d, alpha_c,
                  model="schiller_naumann"):
    """相间动量交换系数 K（单位体积混合物的力密度 / 相对速度，kg/m^3/s）。

    标准形式 K = (3/4) C_D ρ_c α_d |u_rel| / d · f(α_c)，其中 f 为 Wen-Yu 空泡率
    修正 f(α_c) = α_c^-2.65（弥散相稠密时拖曳增强）。低 Re 回归 Stokes 系数
    K_stokes = 18 μ_c α_d / d²（C_D = 24/Re 时自动收敛），避免 |u_rel|→0 时
    数值除零。`model` 预留（"schiller_naumann" 为默认单球分段拖曳）。
    """
    d = float(d)
    rho_c = np.asarray(rho_c, float).ravel()
    mu_c = np.asarray(mu_c, float).ravel()
    umag = np.maximum(np.asarray(umag, float).ravel(), EPS)
    alpha_d = np.asarray(alpha_d, float).ravel()
    alpha_c = np.asarray(alpha_c, float).ravel()
    f_void = np.clip(alpha_c ** (-2.65), 1.0, 1e4)
    re = rho_c * umag * d / np.maximum(mu_c, EPS)
    # 低 Re 段直接取 Stokes 极限（K 与 umag 无关），高 Re 走 Schiller-Naumann
    stokes = 18.0 * mu_c * alpha_d / (d * d) * f_void
    cd = drag_coefficient(re)
    std = 0.75 * cd * rho_c * alpha_d * umag / d * f_void
    low = re < 1.0e-4
    return np.where(low, stokes, std)


def interphase_drag(d, rho_c, mu_c, u_rel, alpha_d, alpha_c,
                    model="schiller_naumann"):
    """拖曳力密度 F_drag = K · u_rel（弥散相作用于连续相的反作用力取负）。

    `u_rel` (n_cells,3)；返回 (K[n_cells], F[n_cells,3])。
    """
    u_rel = np.asarray(u_rel, float)
    umag = _safe_norm(u_rel, axis=1)
    K = drag_exchange(d, rho_c, mu_c, umag, alpha_d, alpha_c, model=model)
    F = K[:, None] * u_rel
    return K, F


# ---------------------------------------------------------------------------
# 相间作用力：升力（Legendre-Magnaudet 简式）
# ---------------------------------------------------------------------------
def lift_force(alpha_d, rho_c, u_rel, curl_u, coef=DEFAULT_LIFT_COEF):
    """升力密度 F_lift = -C_L α_d ρ_c u_rel × (∇×u_m)。

    `alpha_d` (n_cells,)，`rho_c` 标量（连续相密度），`u_rel`/`curl_u` (n_cells,3)。
    返回 (n_cells,3)。coef≤0 时返回零（关闭）。默认 coef=0.5（球形小颗粒）。
    """
    coef = float(coef)
    if abs(coef) <= 1e-12:
        return np.zeros_like(np.asarray(u_rel, float))
    alpha_d = np.asarray(alpha_d, float).ravel()
    return -coef * alpha_d[:, None] * float(rho_c) * np.cross(u_rel, curl_u)


# ---------------------------------------------------------------------------
# 相间作用力：虚拟质量力
# ---------------------------------------------------------------------------
def virtual_mass_force(alpha_d, rho_c, acc, coef=DEFAULT_VIRTUAL_MASS_COEF):
    """虚拟质量力密度 F_vm = C_vm α_d ρ_c a_m（a_m 为混合流场加速度）。

    采用对流加速度 a_m = (u_m·∇)u_m 作为 a_m 的稳态近似；coef≤0 时返回零。
    `acc` (n_cells,3)。返回 (n_cells,3)。
    """
    coef = float(coef)
    if abs(coef) <= 1e-12:
        return np.zeros_like(np.asarray(acc, float))
    alpha_d = np.asarray(alpha_d, float).ravel()
    return coef * alpha_d[:, None] * float(rho_c) * np.asarray(acc, float)


# ---------------------------------------------------------------------------
# 相间作用力：壁面润滑（Antal）
# ---------------------------------------------------------------------------
def wall_lubrication_force(alpha_d, rho_c, u_rel, wall_dist, wall_normal,
                           d, coef_cw1=DEFAULT_WALL_LUB_COEF_CW1,
                           coef_cw2=DEFAULT_WALL_LUB_COEF_CW2):
    """壁面润滑力密度 F_wl = -C_wl α_d ρ_c |u_rel|²/d · n_wall（近壁把相推离）。

    C_wl = max(0, Cw1 + Cw2 d/y_w)。`wall_dist` (n_cells,)，`wall_normal`
    (n_cells,3)（指向壁面，故力取 -n_wall 为远离壁面）。`d` 为弥散相粒径。
    返回 (n_cells,3)；远壁或 coef 无效时返回零。
    """
    d = float(d)
    if d <= 0.0:
        return np.zeros_like(np.asarray(u_rel, float))
    alpha_d = np.asarray(alpha_d, float).ravel()
    y = np.maximum(np.asarray(wall_dist, float).ravel(), EPS)
    u_rel = np.asarray(u_rel, float)
    umag2 = np.sum(u_rel * u_rel, axis=1)
    cwl = np.maximum(float(coef_cw1) + float(coef_cw2) * d / y, 0.0)
    mag = cwl * alpha_d * float(rho_c) * umag2 / d
    return -mag[:, None] * np.asarray(wall_normal, float)


# ---------------------------------------------------------------------------
# 群体平衡核：聚并 / 破碎 / 成核
# ---------------------------------------------------------------------------
def coalescence_kernel(alpha_i, alpha_j, di, dj, rate=DEFAULT_COALESCENCE_RATE):
    """聚并核（单位体积速率 1/s）：K_ij = rate · (d_i + d_j)³ 归一化。

    返回逐单元速率阵（n_cells,）——把相 i 的质量转移到相 j（j 更大）。rate=0
    时恒 0。作为默认常数核，物理上随粒径三次方增强碰撞截面。
    """
    rate = float(rate)
    if abs(rate) <= 1e-14:
        return np.zeros_like(np.asarray(alpha_i, float))
    a = np.asarray(alpha_i, float).ravel()
    b = np.asarray(alpha_j, float).ravel()
    scale = (float(di) + float(dj))
    return rate * (scale ** 3) * a * b


def breakup_kernel(alpha_j, dj, rate=DEFAULT_BREAKUP_RATE):
    """破碎核（单位体积速率 1/s）：K_j = rate · (1 - α_j) —— 破碎率随弥散相
    体积分数增长但有上限（空泡率越高破碎越强，接近拥挤时受限）。返回 (n_cells,)。
    """
    rate = float(rate)
    if abs(rate) <= 1e-14:
        return np.zeros_like(np.asarray(alpha_j, float))
    a = np.asarray(alpha_j, float).ravel()
    return rate * np.multiply(a, np.clip(1.0 - a, 0.0, 1.0))


def nucleation_rate(alpha_cont, rate=DEFAULT_NUCLEATION_RATE):
    """成核核（单位体积速率 1/s）：从连续相生成最小弥散相，正比于连续相体积分数。
    """
    rate = float(rate)
    if abs(rate) <= 1e-14:
        return np.zeros_like(np.asarray(alpha_cont, float))
    return rate * np.asarray(alpha_cont, float).ravel()


def population_balance_source(alphas, diameters, coalescence=DEFAULT_COALESCENCE_RATE,
                              breakup=DEFAULT_BREAKUP_RATE,
                              nucleation=DEFAULT_NUCLEATION_RATE):
    """群体平衡源项 dα_k/dt（n_cells, n_phases），Σ_k S_k = 0（守恒）。

      - 聚并：对 i<j（diam 从小到大），把相 i 转移到相 j（j 更大）。
      - 破碎：对 j>1，把相 j 转移到较小相（分配权重 1/(j) 给每个更小相）。
      - 成核：从连续相（相 0）转移到最小弥散相（相 1）。

    所有核可独立为 0；全部为 0 时返回全零（无源）。
    """
    alphas = np.asarray(alphas, float)
    diam = np.asarray(diameters, float).ravel()
    nph = alphas.shape[1]
    src = np.zeros_like(alphas)
    if nph < 2:
        return src
    co = float(coalescence)
    br = float(breakup)
    nu = float(nucleation)
    # 聚并：i→j（i<j）；按 j 从大到小遍历避免同一步序连加失真
    if abs(co) > 1e-14:
        for i in range(1, nph):
            for j in range(i + 1, nph):
                k_ij = coalescence_kernel(alphas[:, i], alphas[:, j],
                                          diam[i], diam[j], co)
                src[:, i] -= k_ij
                src[:, j] += k_ij
    # 破碎：j→{k<j}；源降为 j、升为 k（等权分配）
    if abs(br) > 1e-14:
        for j in range(2, nph):
            k_j = breakup_kernel(alphas[:, j], diam[j], br)
            n_small = j - 1
            share = k_j / float(n_small)
            src[:, j] -= k_j
            for k in range(1, j):
                src[:, k] += share
    # 成核：0→1
    if abs(nu) > 1e-14:
        k_n = nucleation_rate(alphas[:, 0], nu)
        src[:, 0] -= k_n
        src[:, 1] += k_n
    return src


def _mixture_velocity_acceleration(fv, u, v, w, boundary=None):
    """混合流场对流加速度 a_m = (u_m·∇)u_m（n_cells,3），供虚拟质量力。"""
    u = np.asarray(u, float).ravel()
    v = np.asarray(v, float).ravel()
    w = np.asarray(w, float).ravel()
    b = boundary
    gu = fv.grad_gauss(u, boundary=None if b is None else b[0])
    gv = fv.grad_gauss(v, boundary=None if b is None else b[1])
    gw = fv.grad_gauss(w, boundary=None if b is None else b[2])
    ax = u * gu[:, 0] + v * gu[:, 1] + w * gu[:, 2]
    ay = u * gv[:, 0] + v * gv[:, 1] + w * gv[:, 2]
    az = u * gw[:, 0] + v * gw[:, 1] + w * gw[:, 2]
    return np.stack([ax, ay, az], axis=1)


def _mixture_curl(fv, u, v, w, boundary=None):
    """混合流场旋度 ∇×u_m（n_cells,3），供升力。"""
    u = np.asarray(u, float).ravel()
    v = np.asarray(v, float).ravel()
    w = np.asarray(w, float).ravel()
    b = boundary
    gu = fv.grad_gauss(u, boundary=None if b is None else b[0])
    gv = fv.grad_gauss(v, boundary=None if b is None else b[1])
    gw = fv.grad_gauss(w, boundary=None if b is None else b[2])
    cx = gw[:, 1] - gv[:, 2]
    cy = gu[:, 2] - gw[:, 0]
    cz = gv[:, 0] - gu[:, 1]
    return np.stack([cx, cy, cz], axis=1)


# ---------------------------------------------------------------------------
# 求解器：EulerianMultiphaseSolver
# ---------------------------------------------------------------------------
class EulerianMultiphaseSolver:
    """多相欧拉-欧拉求解器：相体积分数守恒输运 + 相间作用力平衡 + 群体平衡源。

    `update(u, v, w, mdot)` 由给定混合流场推进各弥散相体积分数（上风 + 漂移通量 +
    有界投影），并按相间作用力（拖曳/升力/虚拟质量/壁面润滑）更新逐相相对/漂移速度；
    若开启群体平衡则叠加聚并/破碎/成核源。`rho` / `mu` 暴露逐单元混合密度与动力
    粘度（供 SIMPLE 动量装配）；`relative_velocities` / `drift` / `alphas` /
    `momentum_source` / `interphase_k` 暴露逐相速度与相间作用力（诊断 / 耦合）。
    """

    def __init__(self, fv, rhos=DEFAULT_RHOS, mus=DEFAULT_MUS,
                 diameters=DEFAULT_DIAMETERS, gravity=DEFAULT_GRAVITY,
                 flow_axis=0, inlet_side="min", outlet_side="max",
                 inlet_alphas=None, alpha0=None, relax=0.7, max_iter=50,
                 cfl=0.4, axis=0,
                 drag_model="schiller_naumann", enable_lift=False,
                 enable_virtual_mass=False, enable_wall_lubrication=False,
                 lift_coef=DEFAULT_LIFT_COEF,
                 virtual_mass_coef=DEFAULT_VIRTUAL_MASS_COEF,
                 wall_lub_coef_cw1=DEFAULT_WALL_LUB_COEF_CW1,
                 wall_lub_coef_cw2=DEFAULT_WALL_LUB_COEF_CW2,
                 population_balance=False,
                 coalescence_rate=DEFAULT_COALESCENCE_RATE,
                 breakup_rate=DEFAULT_BREAKUP_RATE,
                 nucleation_rate=DEFAULT_NUCLEATION_RATE,
                 interaction_relax=0.5):
        self.fv = fv
        self.rhos = np.asarray(rhos, float)
        self.mus = np.asarray(mus, float)
        dd = np.asarray(diameters, float).ravel()
        nph = len(self.rhos)
        if len(self.mus) != nph:
            raise ValueError("Eulerian rhos/mus 长度不一致：%d vs %d"
                             % (nph, len(self.mus)))
        self.diameters = np.zeros(nph, float)
        if len(dd) == nph:
            self.diameters[:] = dd
        elif len(dd) == 0:
            pass
        else:
            raise ValueError("Eulerian 粒径长度应为 %d 或 0" % nph)
        self.gravity = tuple(float(g) for g in gravity)
        self.flow_axis = int(flow_axis)
        self.inlet_side = inlet_side
        self.outlet_side = outlet_side
        self.relax = float(relax)
        self.max_iter = int(max_iter)
        self.cfl = float(cfl)
        self.axis = int(axis)
        self._disp = np.arange(1, nph)
        self.drag_model = str(drag_model)
        self.enable_lift = bool(enable_lift)
        self.enable_virtual_mass = bool(enable_virtual_mass)
        self.enable_wall_lubrication = bool(enable_wall_lubrication)
        self.lift_coef = float(lift_coef)
        self.virtual_mass_coef = float(virtual_mass_coef)
        self.wall_lub_coef_cw1 = float(wall_lub_coef_cw1)
        self.wall_lub_coef_cw2 = float(wall_lub_coef_cw2)
        self.population_balance = bool(population_balance)
        self.coalescence_rate = float(coalescence_rate)
        self.breakup_rate = float(breakup_rate)
        self.nucleation_rate = float(nucleation_rate)
        self.interaction_relax = float(interaction_relax)
        # 入口 / 初值弥散相体积分数
        ia = np.zeros(nph - 1, float) if inlet_alphas is None \
            else np.asarray(inlet_alphas, float).ravel()
        if len(ia) != nph - 1:
            raise ValueError("Eulerian 入口相分数应长度 %d" % (nph - 1))
        self.inlet_alphas = np.clip(ia, 0.0, 1.0)
        a0 = np.zeros(nph - 1, float) if alpha0 is None \
            else np.asarray(alpha0, float).ravel()
        if len(a0) != nph - 1:
            raise ValueError("Eulerian 初值相分数应长度 %d" % (nph - 1))
        a0 = np.clip(a0, 0.0, 1.0)
        if np.sum(a0) > 1.0:
            a0 = a0 / np.sum(a0)
        self._alphas = np.zeros((fv.n_cells, nph), float)
        self._alphas[:, 0] = 1.0
        self._alphas[:, self._disp] = a0[None, :]
        self._alphas[:, 0] = 1.0 - np.sum(a0)
        self._slip = np.zeros((fv.n_cells, nph, 3), float)
        self._drift = np.zeros((fv.n_cells, nph, 3), float)
        self._relative = np.zeros((fv.n_cells, nph, 3), float)
        self._phase_vel = np.zeros((fv.n_cells, nph, 3), float)
        self._interphase_k = np.zeros((fv.n_cells, nph), float)
        self._momentum_source = np.zeros((fv.n_cells, 3), float)
        self._residual = 0.0
        self.iteration = 0
        self._u = np.zeros(fv.n_cells, float)
        self._v = np.zeros(fv.n_cells, float)
        self._w = np.zeros(fv.n_cells, float)
        self._mdot = None
        self._wall_dist = np.full(fv.n_cells, 1.0e6, float)
        self._wall_normal = np.zeros((fv.n_cells, 3), float)
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
        return mixture_rho(self._alphas, self.rhos)

    @property
    def mu(self):
        return mixture_mu(self._alphas, self.mus)

    @property
    def slip(self):
        return self._slip

    @property
    def drift(self):
        return self._drift

    @property
    def relative_velocities(self):
        return self._relative

    @property
    def phase_velocities(self):
        return self._phase_vel

    @property
    def momentum_source(self):
        return self._momentum_source

    @property
    def interphase_k(self):
        return self._interphase_k

    def phase_volume(self, k):
        return float(np.sum(self._alphas[:, k] * self.fv.volumes))

    def phase_velocity(self, k):
        return self._phase_vel[:, k, :]

    @property
    def wall_volume(self):
        return int(len(self._wall_faces))

    def set_velocities(self, u, v, w, mdot=None):
        self._u = np.asarray(u, float).ravel()
        self._v = np.asarray(v, float).ravel()
        self._w = np.asarray(w, float).ravel()
        self._mdot = None if mdot is None else np.asarray(mdot, float)

    def set_inlet_alphas(self, inlet_alphas):
        ia = np.asarray(inlet_alphas, float).ravel()
        if len(ia) != self.n_phases - 1:
            raise ValueError("Eulerian 入口相分数应长度 %d" % (self.n_phases - 1))
        self.inlet_alphas = np.clip(ia, 0.0, 1.0)
        return self

    # -- 边界辅助 ------------------------------------------------
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
        self._compute_wall_proximity()
        return self._inlet_faces, self._outlet_faces, self._wall_faces

    def _compute_wall_proximity(self):
        """逐单元最近壁面距离与指向壁面的单位法向（供壁面润滑力）。"""
        fv = self.fv
        n = fv.n_cells
        self._wall_dist = np.full(n, 1.0e6, float)
        self._wall_normal = np.zeros((n, 3), float)
        wf = self._wall_faces
        if len(wf) == 0:
            return
        if len(wf) > MAX_WALL_FACES:
            wf = wf[:MAX_WALL_FACES]
        wc = fv.face_centroid[wf]
        wn = fv.face_normal[wf]
        # 距离平方矩阵 (n_cells, n_wall)，最近壁面取最小
        diff = fv.centroids[:, None, :] - wc[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        idx = np.argmin(d2, axis=1)
        dist = np.sqrt(np.take_along_axis(d2, idx[:, None], axis=1)[:, 0])
        self._wall_dist = np.maximum(dist, EPS)
        # 壁面单位法向指向体网格外部；力沿 -n_wall（远离壁面），此处存 n_wall
        self._wall_normal = wn[idx]

    def _boundary_alphas(self):
        fv = self.fv
        b = np.zeros((fv.n_faces, self.n_phases), float)
        b[:] = self._alphas[fv.owner]
        if len(self._inlet_faces):
            b[self._inlet_faces, 1:] = self.inlet_alphas[None, :]
        return b

    def _flux_magnitude(self, mdot):
        fv = self.fv
        mdot = np.asarray(mdot, float)
        face_mag = np.abs(mdot).copy()
        if self._drift is not None and self._drift.size:
            for k in range(1, self.n_phases):
                af = fv.face_value(self._alphas[:, k])
                drif = self._drift[:, k, :]
                df = np.stack([fv.face_value(drif[:, i]) for i in range(3)], axis=1)
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

    # -- 相间作用力平衡 --------------------------------------------
    def _compute_relative_velocities(self):
        """由混合流场 + 相间作用力求逐相相对速度 u_rel,k（n_cells, n_phases, 3）。

        u_rel,k = u_slip,k（浮力-拖曳）+ (F_lift+Vm+Wl)/K_drag,k；连续相 u_rel,0=0。
        """
        fv = self.fv
        n = fv.n_cells
        nph = self.n_phases
        rm = self.rho
        mm = self.mu
        um = np.stack([self._u, self._v, self._w], axis=1)
        rel = np.zeros((n, nph, 3), float)
        K = np.zeros((n, nph), float)
        acc = None
        curl = None
        if self.enable_virtual_mass:
            acc = _mixture_velocity_acceleration(fv, self._u, self._v, self._w)
        if self.enable_lift:
            curl = _mixture_curl(fv, self._u, self._v, self._w)
        rho_c = float(self.rhos[0])
        for k in self._disp:
            d = float(self.diameters[k])
            alpha_d = np.clip(self._alphas[:, k], EPS, 1.0)
            alpha_c = np.clip(self._alphas[:, 0], EPS, 1.0)
            slip = slip_velocity(d, self.rhos[k], rm, mm, self.gravity,
                                 max_iter=MAX_SLIP_ITER)
            umag = _safe_norm(slip, axis=1)
            k_drag = drag_exchange(d, rm, mm, umag, alpha_d, alpha_c,
                                   model=self.drag_model)
            K[:, k] = k_drag
            other = np.zeros((n, 3), float)
            if self.enable_lift:
                other = other + lift_force(alpha_d, rho_c, slip, curl,
                                           coef=self.lift_coef)
            if self.enable_virtual_mass:
                other = other + virtual_mass_force(alpha_d, rho_c, acc,
                                                   coef=self.virtual_mass_coef)
            if self.enable_wall_lubrication:
                other = other + wall_lubrication_force(
                    alpha_d, rho_c, slip, self._wall_dist, self._wall_normal,
                    d, coef_cw1=self.wall_lub_coef_cw1,
                    coef_cw2=self.wall_lub_coef_cw2)
            # 附加相间力带来的相对速度修正 Δu = F_other / K（K 以 Stokes 下限保稳定）
            denom = np.maximum(k_drag, EPS)
            rel[:, k, :] = slip + other / denom[:, None]
        self._relative = rel
        self._interphase_k = K
        return rel

    def _update_phase_kinematics(self):
        """由混合流场更新逐相相对/漂移/逐相速度与混合动量源。"""
        rel = self._compute_relative_velocities()
        self._slip = rel
        self._drift = drift_velocities(self._alphas, rel)
        um = np.stack([self._u, self._v, self._w], axis=1)
        self._phase_vel = um[:, None, :] + self._drift
        # 混合动量源：附加（非拖曳）相间作用力密度（升力/虚拟质量/壁面润滑）
        nph = self.n_phases
        src = np.zeros_like(um)
        if nph > 1:
            for k in self._disp:
                alpha_d = np.clip(self._alphas[:, k], EPS, 1.0)
                d = float(self.diameters[k])
                rho_c = float(self.rhos[0])
                if self.enable_lift:
                    curl = _mixture_curl(self.fv, self._u, self._v, self._w)
                    src += lift_force(alpha_d, rho_c,
                                      self._relative[:, k, :], curl,
                                      coef=self.lift_coef)
                if self.enable_virtual_mass:
                    acc = _mixture_velocity_acceleration(
                        self.fv, self._u, self._v, self._w)
                    src += virtual_mass_force(alpha_d, rho_c, acc,
                                              coef=self.virtual_mass_coef)
                if self.enable_wall_lubrication:
                    src += wall_lubrication_force(
                        alpha_d, rho_c, self._relative[:, k, :],
                        self._wall_dist, self._wall_normal, d,
                        coef_cw1=self.wall_lub_coef_cw1,
                        coef_cw2=self.wall_lub_coef_cw2)
        self._momentum_source = src

    def update(self, u, v, w, mdot=None):
        """由混合流场推进一次各弥散相显式输运（+群体平衡源），更新相分数与逐相速度。

        返回归一化残差（Δα 对 ||α||）。
        """
        self.set_velocities(u, v, w, mdot=mdot)
        fv = self.fv
        alphas_old = np.asarray(self._alphas, float)
        if mdot is None:
            mdot = volume_flux(fv, u, v, w)
        mdot = np.asarray(mdot, float)
        if len(self._inlet_faces) == 0:
            self._classify_boundary()
        self._update_phase_kinematics()
        acc = self._flux_magnitude(mdot)
        dt_arr = float(self.cfl) * fv.volumes / np.maximum(acc, 1e-30)
        dt = float(np.min(dt_arr)) if dt_arr.size else 0.0
        if not np.isfinite(dt) or dt <= 0.0:
            dt = 1e-6
        a_new, _dt_eff = advance_mixture(
            fv, mdot, alphas_old, self._drift, dt,
            boundary=self._boundary_alphas())
        if self.population_balance:
            src = population_balance_source(
                a_new, self.diameters, self.coalescence_rate,
                self.breakup_rate, self.nucleation_rate)
            a_new[:, self._disp] = a_new[:, self._disp] + dt * src[:, self._disp]
            a_new[:, self._disp] = np.clip(a_new[:, self._disp], 0.0, 1.0)
            s = np.sum(a_new[:, self._disp], axis=1)
            over = s > 1.0
            if over.any():
                a_new[over, self._disp] /= s[over, None]
            a_new[:, self._disp] = np.clip(a_new[:, self._disp], 0.0, 1.0)
            a_new[:, 0] = 1.0 - np.sum(a_new[:, self._disp], axis=1)
            a_new = np.clip(a_new, 0.0, 1.0)
        denom = max(float(_safe_norm(a_new)), EPS)
        resid = float(_safe_norm(a_new - alphas_old)) / denom
        self._alphas = a_new
        self._update_phase_kinematics()
        self.iteration += 1
        self._residual = resid
        return resid

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
                    momentum_src=float(_safe_norm(self.momentum_source)),
                    iteration=self.iteration, residual=self._residual)


# ---------------------------------------------------------------------------
# 工厂注册表
# ---------------------------------------------------------------------------
_EULERIAN_MULTIPHASE_MODELS = {
    "eulerian": EulerianMultiphaseSolver,
    "eulerian_multiphase": EulerianMultiphaseSolver,
    "ee": EulerianMultiphaseSolver,
    "euler": EulerianMultiphaseSolver,
    "euler_euler": EulerianMultiphaseSolver,
}


def make_eulerian_multiphase(fv, model="eulerian", **kwargs):
    """按字符串名实例化欧拉-欧拉多相模型（大小写/下划线不敏感）。"""
    key = str(model).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _EULERIAN_MULTIPHASE_MODELS:
        names = ", ".join(sorted(set(_EULERIAN_MULTIPHASE_MODELS.keys())))
        raise ValueError("未知欧拉-欧拉多相模型 %r（支持：%s）" % (model, names))
    return _EULERIAN_MULTIPHASE_MODELS[key](fv, **kwargs)

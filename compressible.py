# -*- coding: utf-8 -*-
"""P 波 P12：可压缩 / 密度基流（Compressible / Density-Based Flow）—— 理想气体状态
方程（EOS）+ 压力基可压缩 SIMPLE 密度耦合 + 能量-密度耦合 + 声速/马赫数诊断。

纯 numpy 实现（occ / scdm 两环境皆可用），建立在 P4 有限体积离散核心
（`fvm_core.FVM`）、P5 压力基求解器（`pressure_solver.py` 稳态不可压 SIMPLE）、
P7 能量模型（`energy.py` 温度场 T）与 P9 组分（`species.py` `gas_density` 理想气体
密度）之上。相对不可压 SIMPLE（密度恒为参考密度 `_rho0`），可压缩谱系把**密度从
常数升为状态量**：由绝对压力与温度经理想气体 EOS 求得逐单元密度 ρ = p W/(R T)，
并把压缩性以「压力修正方程附加对角项 V/(c² Δt)」与「非稳态密度质量源
(ρ − ρ_prev) V/Δt」两种形式注入压力基耦合，使连续性方程体现 ∂ρ/∂t + ∇·(ρu) = 0。
该模型对应 STAR-CCM+ 可压缩谱系（Compressible / Coupled Flow + Ideal Gas EOS）。

物理框架（压力基可压缩 / 密度基）：
    ρ = p W / (R T)                                    （理想气体 EOS）
    ∂ρ/∂t + ∇·(ρ u) = 0                                （可压缩连续性）
    ρ' = (∂ρ/∂p)_s · p' = ρ p' / (γ p) = p' / c²       （等熵压缩率 → 密度修正）
    c = √(γ R T / W) = √(γ p / ρ)                      （声速）
    Ma = |u| / c                                       （马赫数）
    ∇·u                                               （体积膨胀率 / dilatation）

核心能力：

  1) 理想气体 EOS：`ideal_gas_rho`（ρ = p W/(R T)，标量/数组皆可，T→0 保护）。
  2) 声速 / 马赫数：`sound_speed`（c = √(γ R T/W)）、`sound_speed_pr`
     （c = √(γ p/ρ) 等价形式）、`velocity_magnitude`、`mach_number`。
  3) 压缩性诊断：`dilatation`（∇·u 逐单元）、`compression_work`（p ∇·u 单位体积压缩功）、
     `density_derivative`（(∂ρ/∂p)_T = W/(R T)）。
  4) 密度-压力耦合：`density_correction`（ρ' = ρ p'/(γ p)）、
     `compressibility_diagonal`（压力修正附加对角 V/(c² Δt)）、
     `density_change_source`（非稳态密度质量源 (ρ − ρ_prev) V/Δt）。
  5) 求解器：`CompressibleSolver` 由（绝对压力、温度）经 EOS 推进密度状态，暴露
     `rho`/`pressure`/`temperature`/`sound_speed`/`mach`/`dilatation` 场与
     `face_density`（面密度，供 SIMPLE 动量/Rhie-Chow/压力修正装配）；P10 后端兼容
     门面 `_initialize_field() / step() / residual() / monitor_payload()`。

纯 numpy 约束：所有范数走 `solver_run._safe_norm`，不新增 scipy 依赖；密度欠松弛
（`relax`）保证高密度比下推进稳定，声速/马赫数/压缩性对角项均带除零保护。
"""
import numpy as np

from fvm_core import FVM
from solver_run import _safe_norm
from species import R_UNIV, DEFAULT_P_REF, DEFAULT_T_REF


# ---------------------------------------------------------------------------
# 可压缩流常量
# ---------------------------------------------------------------------------
DEFAULT_GAMMA = 1.4              # 比热比 γ = Cp/Cv（空气双原子 1.4）
DEFAULT_MW = 28.96              # 摩尔质量 W（kg/kmol，空气）
DEFAULT_DT = 1.0e-3             # 伪时间步 Δt（压力修正压缩性项与密度源用）
DEFAULT_RELAX = 0.5             # 密度欠松弛因子
MACH_EPS = 1.0e-12              # 马赫数/声速除零保护
_EPS = 1.0e-30


def _as_fv(fv):
    """接受 FVM 实例或其容器（有 _fv 属性者取之）。"""
    if isinstance(fv, FVM):
        return fv
    inner = getattr(fv, "_fv", None)
    if isinstance(inner, FVM):
        return inner
    raise TypeError("需提供 FVM 实例或含 _fv 的容器")


# ---------------------------------------------------------------------------
# 理想气体 EOS
# ---------------------------------------------------------------------------
def ideal_gas_rho(p, T, W_mix=DEFAULT_MW, R=R_UNIV):
    """理想气体密度 ρ = p W / (R T)（kg/m^3）；标量/数组皆可广播。

    `p` 为**绝对压力**（Pa），`W_mix` 为摩尔质量（kg/kmol），`R` 为通用气体常数
    （J/(kmol K)）。T→0 时以 `_EPS` 保护避免除零（返回大而有限值）。
    """
    p = np.asarray(p, float)
    T = np.asarray(T, float)
    return (p * float(W_mix)) / (float(R) * np.maximum(T, _EPS))


def density_derivative(T, W_mix=DEFAULT_MW, R=R_UNIV):
    """等温密度对压力导数 (∂ρ/∂p)_T = W/(R T)（s²/m²）；标量/数组皆可。

    注意：这是**等温**压缩率，等于 γ/c²（等熵压缩率 1/c² = ρ/(γ p) 才用于压力修正）。
    """
    T = np.asarray(T, float)
    return float(W_mix) / (float(R) * np.maximum(T, _EPS))


# ---------------------------------------------------------------------------
# 声速 / 马赫数 / 速度幅值
# ---------------------------------------------------------------------------
def sound_speed(T, W_mix=DEFAULT_MW, gamma=DEFAULT_GAMMA, R=R_UNIV):
    """声速 c = √(γ R T / W)（m/s）；标量/数组皆可广播。"""
    T = np.asarray(T, float)
    return np.sqrt(np.maximum(float(gamma) * float(R) / float(W_mix), 0.0)
                   * np.maximum(T, 0.0))


def sound_speed_pr(p, rho, gamma=DEFAULT_GAMMA):
    """声速等价形式 c = √(γ p / ρ)（m/s）；p 绝对压力、ρ 密度。"""
    p = np.asarray(p, float)
    rho = np.asarray(rho, float)
    return np.sqrt(np.maximum(float(gamma) * p / np.maximum(rho, _EPS), 0.0))


def velocity_magnitude(u, v, w):
    """速度幅值 |u| = √(u² + v² + w²)（n_cells,）。"""
    u = np.asarray(u, float).ravel()
    v = np.asarray(v, float).ravel()
    w = np.asarray(w, float).ravel()
    return np.sqrt(u * u + v * v + w * w)


def mach_number(u, v, w, c):
    """马赫数 Ma = |u| / c；`u,v,w` 为速度分量、`c` 为声速（标量或逐单元）。"""
    c = np.asarray(c, float)
    return velocity_magnitude(u, v, w) / np.maximum(c, MACH_EPS)


# ---------------------------------------------------------------------------
# 压缩性诊断：体积膨胀率 / 压缩功
# ---------------------------------------------------------------------------
def dilatation(fv, u, v, w):
    """体积膨胀率 ∇·u（1/s，n_cells,）：Σ_f (u_f·n_f) A_f / V_c。

    内部面 owner 计 +、neighbor 计 −（面法向自 owner 指向 neighbor），边界面仅 owner
    贡献，与 `PressureSolver._continuity_imbalance` 同序，保证散度守恒。
    """
    fv = _as_fv(fv)
    fu = fv.face_value(np.asarray(u, float).ravel())
    fv_v = fv.face_value(np.asarray(v, float).ravel())
    fw = fv.face_value(np.asarray(w, float).ravel())
    n = fv.face_normal
    un = fu * n[:, 0] + fv_v * n[:, 1] + fw * n[:, 2]
    flux = un * fv.face_area
    acc = np.zeros(fv.n_cells, float)
    np.add.at(acc, fv.owner, flux)
    is_int = fv.neighbor >= 0
    np.add.at(acc, fv.neighbor[is_int], -flux[is_int])
    return acc / np.maximum(fv.volumes, _EPS)


def compression_work(dil, p):
    """单位体积压缩功 p ∇·u（W/m^3 = Pa/s）。`dil` 为膨胀率、`p` 为绝对压力。"""
    return np.asarray(dil, float) * np.asarray(p, float)


# ---------------------------------------------------------------------------
# 密度-压力耦合（压力基可压缩 SIMPLE）
# ---------------------------------------------------------------------------
def density_correction(pprime, rho, p, gamma=DEFAULT_GAMMA):
    """压力修正引起的密度修正 ρ' = (∂ρ/∂p)_T p' = ρ p'/(γ p)。

    `pprime` 为压力修正场 p'（Pa）、`rho` 为当前密度、`p` 为**绝对压力**。等价于
    ρ' = p' / c²（因 c² = γ p/ρ）。p→0 以 `_EPS` 保护。
    """
    pprime = np.asarray(pprime, float)
    rho = np.asarray(rho, float)
    p = np.asarray(p, float)
    return pprime * rho / (float(gamma) * np.maximum(p, _EPS))


def compressibility_diagonal(volumes, c, dt=DEFAULT_DT):
    """压力修正方程的可压缩性附加对角项 V/(c² Δt)（量纲 m^3/(m/s)²/s = kg/(Pa s)）。

    由非稳态连续性 ∂ρ/∂t = (∂ρ/∂p) p'/Δt = p'/(c² Δt) 乘以单元体积得到；低马赫时
    该项极小（≈ V/(c² Δt)），高马赫时增强压力-密度耦合、改善条件数。
    """
    volumes = np.asarray(volumes, float)
    c = np.asarray(c, float)
    dt = max(float(dt), _EPS)
    c2 = np.maximum(c * c, _EPS)
    return volumes / (c2 * dt)


def density_change_source(rho, rho_prev, volumes, dt=DEFAULT_DT):
    """非稳态密度质量源 (ρ − ρ_prev) V/Δt（kg/s），进入连续性不平衡 RHS。

    稳态收敛（ρ→ρ_prev）时该项→0，退化为不可压连续性 ∇·(ρu)=0。
    """
    rho = np.asarray(rho, float)
    rho_prev = np.asarray(rho_prev, float)
    volumes = np.asarray(volumes, float)
    dt = max(float(dt), _EPS)
    return (rho - rho_prev) * volumes / dt


# ---------------------------------------------------------------------------
# 可压缩求解器：CompressibleSolver
# ---------------------------------------------------------------------------
class CompressibleSolver:
    """可压缩 / 密度基流求解器：理想气体 EOS 密度状态 + 声速/马赫数/膨胀率诊断 +
    压力基可压缩耦合量（面密度 / 压缩性对角 / 密度质量源 / 密度修正）。

    `update(u,v,w,mdot,T,p)` 由给定流场、温度场（可来自 `energy.py`）与表压场
    （来自 `pressure_solver.py`）经 EOS 推进逐单元密度状态（欠松弛）并更新派生量；
    `rho` 暴露密度场（供 SIMPLE `rho` 属性 / 动量装配），`face_density` 暴露面插值
    密度（供动量/Rhie-Chow/压力修正矩阵），`compressibility_diagonal` /
    `density_change_source` / `density_correction` 提供压力修正方程的可压缩性耦合项。
    """

    def __init__(self, fv, model="compressible", gamma=DEFAULT_GAMMA,
                 W=DEFAULT_MW, R=R_UNIV, p_ref=DEFAULT_P_REF,
                 T_ref=DEFAULT_T_REF, rho_ref=None, dt=DEFAULT_DT,
                 relax=DEFAULT_RELAX):
        self.fv = _as_fv(fv)
        self.model = str(model)
        self.gamma = float(gamma)
        self.W = float(W)
        self.R = float(R)
        self.p_ref = float(p_ref)
        self.T_ref = float(T_ref)
        self.dt = float(dt)
        self.relax = float(relax)
        n = self.fv.n_cells
        # 状态：表压 _p、温度 _T、密度 _rho（绝对值由 p_ref + _p 给出）
        self._p = np.zeros(n, float)
        self._T = np.full(n, float(T_ref), float)
        if rho_ref is None:
            rho_ref = float(ideal_gas_rho(self.p_ref, self.T_ref, self.W, self.R))
        self.rho_ref = float(rho_ref)
        self._rho = np.full(n, self.rho_ref, float)
        self._rho_prev = self._rho.copy()
        # 流场
        self._u = np.zeros(n, float)
        self._v = np.zeros(n, float)
        self._w = np.zeros(n, float)
        self._mdot = None
        # 派生量
        self._c = np.full(n, float(sound_speed(self.T_ref, self.W,
                                               self.gamma, self.R)), float)
        self._mach = np.zeros(n, float)
        self._dil = np.zeros(n, float)
        self._residual = 0.0
        self.iteration = 0

    # -- 场属性 ------------------------------------------------
    @property
    def rho(self):
        """逐单元密度场（n_cells,，kg/m^3）。"""
        return self._rho

    @property
    def temperature(self):
        return self._T

    @property
    def gauge_pressure(self):
        """表压场（n_cells,，Pa；相对 p_ref）。"""
        return self._p

    @property
    def pressure(self):
        """绝对压力场 p_abs = p_ref + p_gauge（n_cells,，Pa）。"""
        return self.p_ref + self._p

    @property
    def sound_speed(self):
        """逐单元声速场（n_cells,，m/s）。"""
        return self._c

    @property
    def mach(self):
        """逐单元马赫数场（n_cells,）。"""
        return self._mach

    @property
    def dilatation(self):
        """逐单元体积膨胀率 ∇·u（1/s）。"""
        return self._dil

    @property
    def compression_work(self):
        """逐单元单位体积压缩功 p ∇·u（W/m^3）。"""
        return compression_work(self._dil, self.pressure)

    @property
    def max_mach(self):
        return float(self._mach.max()) if self._mach.size else 0.0

    # -- 状态设置 / 推进 ---------------------------------------
    def set_velocities(self, u, v, w, mdot=None):
        """存储当前速度场（膨胀率/马赫数/面密度用）。"""
        self._u = np.asarray(u, float).ravel()
        self._v = np.asarray(v, float).ravel()
        self._w = np.asarray(w, float).ravel()
        self._mdot = mdot

    def set_temperature(self, T):
        """设置温度场（可来自能量模型 `energy.EnergySolver.T`）。"""
        T = np.asarray(T, float).ravel()
        if T.shape == (self.fv.n_cells,):
            self._T = T

    def set_pressure(self, p):
        """设置表压场（来自 `PressureSolver._p`）。"""
        p = np.asarray(p, float).ravel()
        if p.shape == (self.fv.n_cells,):
            self._p = p

    def _update_derived(self):
        """由当前 T/速度刷新声速、马赫数、膨胀率。"""
        self._c = np.asarray(sound_speed(self._T, self.W, self.gamma, self.R),
                             float)
        self._mach = mach_number(self._u, self._v, self._w, self._c)
        self._dil = dilatation(self.fv, self._u, self._v, self._w)

    def update(self, u, v, w, mdot=None, T=None, p=None):
        """由流场/温度/压力经 EOS 推进一次密度状态，返回归一化残差（Δρ 对 ||ρ||）。

        `T` 缺省保持当前温度场；`p` 为表压（缺省保持当前）。密度欠松弛：
        ρ_new = (1−relax) ρ_old + relax · ρ_EOS(p_abs, T)。更新后刷新派生诊断量。
        """
        self.set_velocities(u, v, w, mdot=mdot)
        if T is not None:
            self.set_temperature(T)
        if p is not None:
            self.set_pressure(p)
        rho_old = self._rho.copy()
        self._rho_prev = rho_old
        rho_eos = np.asarray(ideal_gas_rho(self.pressure, self._T, self.W, self.R),
                             float)
        self._rho = (1.0 - self.relax) * rho_old + self.relax * rho_eos
        self._update_derived()
        denom = max(float(_safe_norm(rho_old)), _EPS)
        resid = float(_safe_norm(self._rho - rho_old)) / denom
        self._residual = resid
        self.iteration += 1
        return resid

    # -- 压力基可压缩耦合量 ------------------------------------
    def face_density(self, fv=None):
        """面密度（n_faces,）：内部面取 owner/neighbor 平均，边界面取 owner。

        供 SIMPLE 动量/Rhie-Chow/压力修正矩阵使用（`PressureSolver._face_rho`）。
        """
        f = self.fv if fv is None else _as_fv(fv)
        is_int = f.neighbor >= 0
        out = np.zeros(f.n_faces, float)
        out[is_int] = 0.5 * (self._rho[f.owner[is_int]]
                             + self._rho[f.neighbor[is_int]])
        out[~is_int] = self._rho[f.owner[~is_int]]
        return out

    def compressibility_diagonal(self, dt=None):
        """压力修正方程可压缩性附加对角项 V/(c² Δt)（n_cells,）。"""
        dt = self.dt if dt is None else float(dt)
        return compressibility_diagonal(self.fv.volumes, self._c, dt)

    def density_change_source(self, dt=None):
        """非稳态密度质量源 (ρ − ρ_prev) V/Δt（n_cells,，kg/s）。"""
        dt = self.dt if dt is None else float(dt)
        return density_change_source(self._rho, self._rho_prev,
                                     self.fv.volumes, dt)

    def density_correction(self, pprime):
        """压力修正 → 密度修正 ρ' = ρ p'/(γ p_abs)（n_cells,，kg/m^3）。"""
        return density_correction(pprime, self._rho, self.pressure, self.gamma)

    # -- P10 后端兼容门面 --------------------------------------
    def _initialize_field(self):
        self._p[:] = 0.0
        self._T[:] = float(self.T_ref)
        self._rho[:] = self.rho_ref
        self._rho_prev = self._rho.copy()
        self._update_derived()
        self._residual = 0.0
        self.iteration = 0
        return dict(rho_min=float(self._rho.min()), rho_max=float(self._rho.max()),
                    c_min=float(self._c.min()), c_max=float(self._c.max()),
                    mach_max=float(self._mach.max()))

    def step(self):
        resid = float(self.update(self._u, self._v, self._w, mdot=self._mdot))
        self._residual = resid
        return dict(residual=resid, rho_min=float(self._rho.min()),
                    rho_max=float(self._rho.max()),
                    mach_max=float(self._mach.max()))

    def residual(self):
        return self._residual

    def monitor_payload(self):
        return dict(rho_min=float(self._rho.min()), rho_max=float(self._rho.max()),
                    p_min=float(self.pressure.min()),
                    p_max=float(self.pressure.max()),
                    T_min=float(self._T.min()), T_max=float(self._T.max()),
                    c_min=float(self._c.min()), c_max=float(self._c.max()),
                    mach_max=float(self._mach.max()),
                    dil_max=float(np.abs(self._dil).max()),
                    iteration=self.iteration, residual=self._residual)


# ---------------------------------------------------------------------------
# 工厂注册表
# ---------------------------------------------------------------------------
_COMPRESSIBLE_MODELS = {
    "compressible": CompressibleSolver,
    "compressible_flow": CompressibleSolver,
    "density_based": CompressibleSolver,
    "densitybased": CompressibleSolver,
    "ideal_gas": CompressibleSolver,
    "idealgas": CompressibleSolver,
    "coupl_flow": CompressibleSolver,
    "coupled_flow": CompressibleSolver,
}


def make_compressible(fv, model="compressible", **kwargs):
    """按字符串名实例化可压缩 / 密度基流模型（大小写/连字符/下划线不敏感）。"""
    key = str(model).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _COMPRESSIBLE_MODELS:
        names = ", ".join(sorted(set(_COMPRESSIBLE_MODELS.keys())))
        raise ValueError("未知可压缩模型 %r（支持：%s）" % (model, names))
    return _COMPRESSIBLE_MODELS[key](fv, model=model, **kwargs)

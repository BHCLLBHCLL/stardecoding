# -*- coding: utf-8 -*-
"""P 波 P9：燃烧（star.combustion）—— 全局 Arrhenius 反应动力学 + 火焰速度 + 点火器。

纯 numpy 实现（occ / scdm 两环境皆可用），建立在 `species.py`（组分质量分数输运）
之上，为可反应混合物提供**单步全局化学机理**的反应速率、物种源项与热释放，可与
SIMPLE 外环 + 能量模型温度场耦合。

物理框架（全局单步燃烧）：
    ν_fuel Fuel + ν_o Ox → ν_p Products
    ω = A T^β exp(-Ea/(R T)) Π_i [Reactant_i]^{ν_i}   （Arrhenius，kmol/(m^3 s)）
    S_i = ω Mw_i ν_i                                    （物种质量源 kg/(m^3 s)，Σ S_i = 0）
    Q   = ω H_rxn                                       （热释放 W/m^3，H_rxn 放热为正）
其中浓度 [C_i] = ρ Y_i / Mw_i（kmol/m^3）。扰动区由点火器（ignitor）调制反应速率。

核心能力：

  1) Arrhenius 速率：`arrhenius_rate`（A T^β exp(-Ea/RT)），T>0 保护。
  2) 全局反应：`GlobalReaction`（净计量系数 nu、反应物阶数、A/Ea/β、热释放 H_rxn），
     提供 `rate(conc,T)`（kmol/(m^3 s)）、`mass_source(Y,T,rho,Mw)`（kg/(m^3 s)，
     自动满足 Σ S_i = 0）、`heat_release(...)`（W/m^3）。
  3) 火焰速度：`laminar_flame_speed`（幂律相关）、`flame_thickness`（热扩散率/火焰速度）、
     `combustion_progress`（进度变量 c = (Y_f0 - Y_f)/Y_f0）。
  4) 点火器：`TemperatureIgnitor` / `SparkIgnitor` / `ProgressVariableIgnitor`，
     提供点火区 `factor` 与（火花）热源 `heat_source` 调制反应速率。
  5) 求解器：`CombustionModel` 组合组分输运 + 全局反应 + 点火器，
     `update(u,v,w,mdot,nu_t,T)` 推进化学源项并更新组分/热释放/火焰诊断；
     P10 后端兼容门面 `_initialize_field() / step() / residual() / monitor_payload()`。
  6) 工厂：`make_combustion` 经别名/大小写解析，未知模型报 ValueError。

纯 numpy 约束：所有范数走 `solver_run._safe_norm`，不依赖 scipy / np.linalg.norm。
"""
import numpy as np

from fvm_core import FVM
from solver_run import _safe_norm
import species as _species


# ---------------------------------------------------------------------------
# 燃烧常数
# ---------------------------------------------------------------------------
R_UNIV = 8314.462       # 通用气体常数（J/(kmol K)），与 kg/kmol 分子量配套
EPS = 1e-12

# 甲烷-空气单步全球反应缺省参数（教程级标定，非精确化学机理）
DEFAULT_NAMES = ["CH4", "O2", "CO2", "H2O", "N2"]
DEFAULT_MW = [16.04, 32.00, 44.01, 18.02, 28.01]
DEFAULT_NU = [-1.0, -2.0, 1.0, 2.0, 0.0]        # CH4 + 2O2 → CO2 + 2H2O
DEFAULT_REACT_IDX = [0, 1]                        # 反应物 CH4, O2
DEFAULT_REACT_ORDERS = [1.0, 1.0]
DEFAULT_A = 1.0e6                                 # 前置因子（一致单位，教程级）
DEFAULT_EA = 1.3e8                                # 活化能（J/kmol）
DEFAULT_BETA = 0.0                                # 温度指数
DEFAULT_H_RXN = 8.03e8                            # 放热（J/kmol 反应进度）

# 层流火焰速度幂律相关（S_L = S_L_ref * (T/T_ref)^T_exp）
FLAME_SPEED_REF = 0.4                             # m/s，甲烷-空气 300K 约 0.4
FLAME_SPEED_TREF = 300.0
FLAME_SPEED_TEXP = 1.7

# 热扩散率缺省（空气级，用于火焰厚度）
DEFAULT_ALPHA_THERMAL = 2.2e-5


# ---------------------------------------------------------------------------
# Arrhenius 速率
# ---------------------------------------------------------------------------
def arrhenius_rate(A, Ea, beta=0.0, T=300.0, R=R_UNIV):
    """Arrhenius 速率常数 k = A T^β exp(-Ea/(R T))（T 按逐单元取，>0 保护）。"""
    T = np.asarray(T, float)
    T = np.maximum(T, EPS)
    return float(A) * np.power(T, float(beta)) * np.exp(-float(Ea) / (float(R) * T))


# ---------------------------------------------------------------------------
# 全局单步反应
# ---------------------------------------------------------------------------
class GlobalReaction:
    """单步全球反应：净计量系数 + 反应物阶数 + Arrhenius 速率 + 热释放。

    `nu` 为各组分的净计量系数（反应物为负、生成物为正）；`reactant_idx` /
    `reactant_orders` 为反应物下标与其速率阶数（构成浓度幂积）。`H_rxn` 为每
    kmol 反应进度的放热量（正 = 放热）。
    """

    def __init__(self, nu, reactant_idx, reactant_orders, A=DEFAULT_A,
                 Ea=DEFAULT_EA, beta=DEFAULT_BETA, H_rxn=DEFAULT_H_RXN,
                 name="globe"):
        self.nu = np.asarray(nu, float)
        self.reactant_idx = [int(i) for i in reactant_idx]
        self.reactant_orders = [float(o) for o in reactant_orders]
        self.A = float(A)
        self.Ea = float(Ea)
        self.beta = float(beta)
        self.H_rxn = float(H_rxn)
        self.name = str(name)

    def rate(self, conc, T):
        """Arrhenius 反应进度速率（kmol/(m^3 s)）。

        `conc` 为组分区浓度（n_cells, n_sp, kmol/m^3）；`T` 为温度（n_cells, K）。
        """
        conc = np.asarray(conc, float)
        T = np.asarray(T, float)
        k = arrhenius_rate(self.A, self.Ea, self.beta, T, R_UNIV)
        w = np.ones(np.asarray(T).shape, float)
        for idx, ordr in zip(self.reactant_idx, self.reactant_orders):
            w = w * np.maximum(conc[:, idx], 0.0) ** ordr
        return k * w

    def mass_source(self, Y, T, rho, Mw):
        """各组分质量源（n_cells, n_sp, kg/(m^3 s)），自动满足 Σ_i S_i = 0。"""
        Y = np.asarray(Y, float)
        Mw = np.asarray(Mw, float)
        rho = np.asarray(rho, float)
        conc = rho[:, None] * Y / np.maximum(Mw[None, :], EPS)
        w = self.rate(conc, T)
        return w[:, None] * Mw[None, :] * self.nu[None, :]

    def heat_release(self, Y, T, rho, Mw):
        """反应热释放（n_cells, W/m^3）= ω H_rxn（放热为正）。"""
        Y = np.asarray(Y, float)
        Mw = np.asarray(Mw, float)
        rho = np.asarray(rho, float)
        conc = rho[:, None] * Y / np.maximum(Mw[None, :], EPS)
        w = self.rate(conc, T)
        return w * self.H_rxn


def make_prequick_mech(names=None, Mw=None):
    """构造甲烷-空气单步全球反应（缺省物种/计量匹配 `species.DEFAULT_NAMES`）。"""
    if names is None:
        names = list(DEFAULT_NAMES)
    if Mw is None:
        Mw = list(DEFAULT_MW)
    ns = len(names)
    nu = list(DEFAULT_NU)
    if len(nu) != ns:
        nu = nu[:ns] + [0.0] * (ns - len(nu))
    return GlobalReaction(nu, DEFAULT_REACT_IDX, DEFAULT_REACT_ORDERS,
                          A=DEFAULT_A, Ea=DEFAULT_EA, beta=DEFAULT_BETA,
                          H_rxn=DEFAULT_H_RXN, name="ch4_air")


# ---------------------------------------------------------------------------
# 火焰速度 / 热扩散率 / 进度变量
# ---------------------------------------------------------------------------
def laminar_flame_speed(T, S_L_ref=FLAME_SPEED_REF, T_ref=FLAME_SPEED_TREF,
                        T_exp=FLAME_SPEED_TEXP):
    """层流火焰速度（未燃气体）幂律相关：S_L = S_L_ref (T/T_ref)^T_exp（m/s）。"""
    T = np.asarray(T, float)
    T = np.maximum(T, EPS)
    return S_L_ref * np.power(T / T_ref, T_exp)


def flame_thickness(S_L, alpha=DEFAULT_ALPHA_THERMAL):
    """层流火焰厚度 δ_L = alpha / S_L（m）。"""
    S_L = np.asarray(S_L, float)
    return alpha / np.maximum(np.abs(S_L), EPS)


def combustion_progress(Y, fuel_idx=0, Y_fuel_inlet=None):
    """燃烧进度变量 c = (Y_f0 - Y_f)/Y_f0（未燃 0 → 已燃 1，夹取 [0,1]）。"""
    Y = np.asarray(Y, float)
    yf = Y[:, fuel_idx]
    if Y_fuel_inlet is None:
        y0 = float(np.max(yf)) if yf.size else 1.0
    else:
        y0 = float(Y_fuel_inlet)
    return np.clip((y0 - yf) / max(y0, EPS), 0.0, 1.0)


# ---------------------------------------------------------------------------
# 点火器
# ---------------------------------------------------------------------------
class TemperatureIgnitor:
    """温度点火器：温度超过阈值后进入已燃区（factor → 1）。"""

    def __init__(self, threshold=900.0):
        self.threshold = float(threshold)

    def factor(self, centroids, T=None, **kwargs):
        if T is None:
            return np.ones(len(centroids), float)
        T = np.asarray(T, float)
        return (T > self.threshold).astype(float)


class SparkIgnitor:
    """火花点火器：球心 radius 内沉积热量，超出后 factor 平滑衰减。"""

    def __init__(self, position=(0.5, 0.5, 0.5), radius=0.05, energy=1.0e5,
                 duration=1.0):
        self.position = tuple(float(p) for p in position)
        self.radius = float(radius)
        self.energy = float(energy)
        self.duration = float(max(duration, EPS))

    def factor(self, centroids, **kwargs):
        d = _safe_norm(np.asarray(centroids, float) - np.asarray(self.position),
                       axis=1)
        return np.clip(1.0 - d / max(self.radius, EPS), 0.0, 1.0)

    def heat_source(self, centroids, t=0.0):
        """火花热源（W/m^3）：在 radius 内沉积 energy/duration 功率密度。"""
        f = self.factor(centroids)
        return f * self.energy / max(self.duration, EPS)


class ProgressVariableIgnitor:
    """进度变量点火器：燃烧进度超过阈值后进入已燃区（factor → 1）。"""

    def __init__(self, threshold=0.5):
        self.threshold = float(threshold)

    def factor(self, centroids, progress=None, T=None, **kwargs):
        if progress is None:
            return np.ones(len(centroids), float)
        progress = np.asarray(progress, float)
        return (progress > self.threshold).astype(float)


_IGNITORS = {
    "temperature": TemperatureIgnitor,
    "temp": TemperatureIgnitor,
    "spark": SparkIgnitor,
    "progress": ProgressVariableIgnitor,
    "progress_variable": ProgressVariableIgnitor,
}


def make_ignitor(kind="temperature", **kwargs):
    """按字符串名实例化点火器（大小写/下划线不敏感）。"""
    key = str(kind).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _IGNITORS:
        names = ", ".join(sorted(set(_IGNITORS.keys())))
        raise ValueError("未知点火器 %r（支持：%s）" % (kind, names))
    return _IGNITORS[key](**kwargs)


# ---------------------------------------------------------------------------
# 燃烧模型
# ---------------------------------------------------------------------------
class CombustionModel:
    """全局燃烧模型：组分输运 + Arrhenius 全球反应 + 点火器 + 火焰诊断。

    `update(u, v, w, mdot, nu_t, T)` 用当前温度与组分计算化学源项并推进一次组分
    输运，返回归一化残差。`Y` / `mass_fractions` 暴露组分质量分数矩阵；
    `T` 暴露温度场；`heat_release` 为逐单元反应热释放（W/m^3）；`omega` 为反应
    进度速率（kmol/(m^3 s)）；`progress` 为燃烧进度变量；`s_L` / `flame_thickness`
    为层流火焰速度与厚度；`ignited` 为点火区掩膜。
    """

    def __init__(self, fv, names=None, Mw=None, reaction=None, ignitor=None,
                 species=None, inlet_fractions=None, rho=None, T_ref=300.0,
                 flow_axis=0, inlet_side="min", outlet_side="max",
                 relax=0.7, fuel_idx=0):
        self.fv = fv
        if names is None:
            names = list(DEFAULT_NAMES)
        if Mw is None:
            Mw = list(DEFAULT_MW)
        names = list(names)
        Mw = np.asarray(Mw, float)
        if reaction is None:
            reaction = make_prequick_mech(names, Mw)
        if reaction.nu.size != len(Mw):
            raise ValueError("Reaction nu 长度 %d 与物种数 %d 不一致"
                             % (reaction.nu.size, len(Mw)))
        self.reaction = reaction
        self.ignitor = ignitor
        self.fuel_idx = int(fuel_idx)
        # 组分输运求解器：优先注入，否则构建缺省甲烷-空气工况
        if species is not None and not isinstance(species, str):
            self._sp = species
        else:
            sp_kwargs = dict(names=names, Mw=list(Mw),
                             inlet_fractions=inlet_fractions, rho=rho,
                             T_ref=T_ref, flow_axis=flow_axis,
                             inlet_side=inlet_side, outlet_side=outlet_side,
                             relax=relax)
            self._sp = _species.make_species(fv, model="species", **sp_kwargs)
        if inlet_fractions is not None:
            self._sp.set_inlet_fractions(inlet_fractions)
        self.T_ref = float(T_ref)
        self._T = np.full(fv.n_cells, T_ref, float)
        self._heat_release = np.zeros(fv.n_cells, float)
        self._omega = np.zeros(fv.n_cells, float)
        self._progress = np.zeros(fv.n_cells, float)
        self._residual = 0.0
        self.iteration = 0
        self._u = np.zeros(fv.n_cells, float)
        self._v = np.zeros(fv.n_cells, float)
        self._w = np.zeros(fv.n_cells, float)
        self._mdot = None
        self._nu_t = None

    # -- 场属性 ------------------------------------------------
    @property
    def species_solver(self):
        return self._sp

    @property
    def Y(self):
        return self._sp.Y

    @property
    def mass_fractions(self):
        return self._sp.Y

    @property
    def X(self):
        return self._sp.X

    @property
    def W_mix(self):
        return self._sp.W_mix

    @property
    def T(self):
        return self._T

    @property
    def heat_release(self):
        return self._heat_release

    @property
    def omega(self):
        return self._omega

    @property
    def progress(self):
        return self._progress

    @property
    def ignited(self):
        if self.ignitor is None:
            return np.ones(self.fv.n_cells, bool)
        f = self.ignitor.factor(self.fv.centroids, T=self._T,
                                progress=self._progress)
        return np.asarray(f, float) > 0.0

    @property
    def s_L(self):
        """层流火焰速度（m/s，标量，取当前温度场均值代入幂律相关）。"""
        T_rep = float(np.mean(self._T)) if self._T.size else self.T_ref
        return float(laminar_flame_speed(T_rep))

    @property
    def flame_thickness(self):
        return float(flame_thickness(self.s_L))

    @property
    def fuel(self):
        return self._sp.names[self.fuel_idx]

    def set_temperature(self, T):
        """注入温度场（供 Arrhenius 速率与热释放计算）。"""
        T = np.asarray(T, float).ravel()
        if T.shape != (self.fv.n_cells,):
            raise ValueError("燃烧温度场需形状 (%d,)" % self.fv.n_cells)
        self._T = T
        return self

    def set_velocities(self, u, v, w, mdot=None, nu_t=None):
        self._u = np.asarray(u, float).ravel()
        self._v = np.asarray(v, float).ravel()
        self._w = np.asarray(w, float).ravel()
        self._mdot = None if mdot is None else np.asarray(mdot, float)
        self._nu_t = None if nu_t is None else np.asarray(nu_t, float)

    def _chemical_source(self):
        """由当前温度/组分计算化学质量源（n_cells, n_sp）与热释放、进度速率。"""
        Y = self._sp.Y
        Mw = self._sp.Mw
        rho = np.maximum(self._sp.rho, 1e-3)
        w = self.reaction.rate(rho[:, None] * Y / Mw[None, :], self._T)
        if self.ignitor is not None:
            f = np.asarray(self.ignitor.factor(self.fv.centroids, T=self._T,
                                               progress=self._progress), float)
            w = w * f
        src = w[:, None] * Mw[None, :] * self.reaction.nu[None, :]
        self._omega = w
        self._heat_release = w * self.reaction.H_rxn
        self._progress = combustion_progress(Y, self.fuel_idx)
        if self.ignitor is not None and isinstance(self.ignitor, SparkIgnitor):
            self._heat_release += self.ignitor.heat_source(self.fv.centroids)
        return src

    def update(self, u, v, w, mdot=None, nu_t=None, T=None):
        """用当前流场/温度推进一次化学源项 + 组分输运，返回归一化残差。"""
        if T is not None:
            self.set_temperature(T)
        src = self._chemical_source()
        resid = self._sp.update(u, v, w, mdot=mdot, source=src, nu_t=nu_t)
        self._progress = combustion_progress(self._sp.Y, self.fuel_idx)
        self.iteration += 1
        self._residual = resid
        return resid

    # -- P10 后端兼容门面 ----------------------------------------
    def _initialize_field(self):
        self._sp._initialize_field()
        self._T[:] = self.T_ref
        self._heat_release[:] = 0.0
        self._omega[:] = 0.0
        self._progress[:] = combustion_progress(self._sp.Y, self.fuel_idx)
        self._residual = 0.0
        self.iteration = 0
        return dict(n_species=self._sp.n_species, fuel=self.fuel,
                    y_min=float(self._sp.Y.min()), y_max=float(self._sp.Y.max()),
                    heat_release_max=float(self._heat_release.max()))

    def step(self):
        resid = float(self.update(self._u, self._v, self._w, mdot=self._mdot,
                                  nu_t=self._nu_t, T=self._T))
        self._residual = resid
        p = self.monitor_payload()
        return dict(residual=resid, n_species=p["n_species"],
                    y_min=p["y_min"], y_max=p["y_max"],
                    heat_release_max=p["heat_release_max"])

    def residual(self):
        return self._residual

    def monitor_payload(self):
        return dict(n_species=self._sp.n_species, fuel=self.fuel,
                    y_min=float(self._sp.Y.min()), y_max=float(self._sp.Y.max()),
                    w_mix=float(self.W_mix.mean()),
                    heat_release_max=float(self._heat_release.max()),
                    heat_release_mean=float(self._heat_release.mean()),
                    progress_max=float(self._progress.max()),
                    s_L=self.s_L, flame_thickness=self.flame_thickness,
                    n_ignited=int(np.count_nonzero(self.ignited)),
                    iteration=self.iteration, residual=self._residual)


# ---------------------------------------------------------------------------
# 工厂注册表
# ---------------------------------------------------------------------------
_COMBUSTION_MODELS = {
    "combustion": CombustionModel,
    "globe": CombustionModel,
    "global": CombustionModel,
}


def make_combustion(fv, model="combustion", **kwargs):
    """按字符串名实例化燃烧模型（大小写/下划线不敏感）。"""
    key = str(model).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _COMBUSTION_MODELS:
        names = ", ".join(sorted(set(_COMBUSTION_MODELS.keys())))
        raise ValueError("未知燃烧模型 %r（支持：%s）" % (model, names))
    return _COMBUSTION_MODELS[key](fv, **kwargs)

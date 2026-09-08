# -*- coding: utf-8 -*-
"""P 波 P9：组分输运（star.species）—— 质量分数守恒输运 + 摩尔分数/混合摩尔质量。

纯 numpy 实现（occ / scdm 两环境皆可用），建立在 P4 有限体积离散核心
（`fvm_core.FVM`）与 P5 压力基求解器（`pressure_solver.solve_linear`）之上，
为多组分混合物提供质量分数 Y_k 的对流-扩散输运求解，并可与湍流 Sc_t 及
燃烧反应源项（`combustion.py`）耦合。

物理方程（保守形式，稳态包中每一组分 k 的质量分数，N-1 个输运 + 1 个补齐）：
    div(rho u Y_k) - div(rho D_eff grad Y_k) = S_k   （k = 0..N-2）
    Y_{N-1} = 1 - Σ_{k<N-1} Y_k                        （补齐/质量守恒）
其中 D_eff = D_mol + nu_t/Sc_t（湍流组分扩散），S_k 为化学反应源项（kg/(m^3 s)）。
组分间换算：
    X_i = (Y_i/Mw_i) / Σ_j (Y_j/Mw_j)   （质量分数 → 摩尔分数）
    Y_i = (X_i Mw_i) / Σ_j (X_j Mw_j)   （摩尔分数 → 质量分数）
    W_mix = 1 / Σ_j (Y_j/Mw_j)           （混合摩尔质量 kg/kmol）
    rho = P W_mix / (R_univ T)           （理想气体密度，R_univ = 8314.462 J/(kmol K)）

核心能力：

  1) 组分换算：`mass_to_mole_frac` / `mole_to_mass_frac` / `mixture_molar_mass` /
     `gas_density`（理想气体密度）。
  2) 输运装配/求解：`assemble_species_transport`（上风对流 + 中心扩散 + 源项 +
     零梯度/定值边界）与 `solve_species`（欠松弛一次求解）。
  3) 边界分类：`classify_species`（入口/出口/壁面全覆盖不重叠）。
  4) 求解器：`SpeciesSolver` 提供 `update(u,v,w,mdot,source)` 推进各组分质量分数
     （补齐组分 + 有界投影保 Σ Y = 1），暴露 `Y` / `X` / `W_mix` / `rho`；
     P10 后端兼容门面 `_initialize_field() / step() / residual() / monitor_payload()`。
  5) 工厂：`make_species` 经别名/大小写解析，未知模型报 ValueError。

纯 numpy 约束：所有范数走 `solver_run._safe_norm`，不依赖 scipy / np.linalg.norm。
"""
import numpy as np

from fvm_core import FVM
from pressure_solver import solve_linear
from solver_run import _safe_norm


# ---------------------------------------------------------------------------
# 组分输运常数
# ---------------------------------------------------------------------------
R_UNIV = 8314.462       # 通用气体常数（J/(kmol K)），与 kg/kmol 分子量配套
V_MOLAR_0 = 22.414       # 标况摩尔体积（m^3/kmol），供参考
DEFAULT_SCALAR_DIFF = 2.0e-5   # 分子扩散系数 D_mol（m^2/s），空气级
DEFAULT_SC_TURB = 0.9          # 湍流 Schmidt 数（组分湍流扩散 D_t = nu_t/Sc_t）
DEFAULT_P_REF = 101325.0       # 参考压力（Pa）
DEFAULT_T_REF = 300.0          # 参考温度（K）
EPS = 1e-12

# 组分默认分子量（kg/kmol）与名称——甲烷-空气单步燃烧工况
DEFAULT_NAMES = ["CH4", "O2", "CO2", "H2O", "N2"]
DEFAULT_MW = [16.04, 32.00, 44.01, 18.02, 28.01]
DEFAULT_FRACTIONS = [0.0, 0.233, 0.0, 0.0, 0.767]   # 入口空气（0.767 氮气）

# 组分边界类型
SP_ZERO_GRAD = 0    # 零梯度：∂Y/∂n = 0（出口/壁面缺省）
SP_FIXED = 1        # 定值：Dirichlet，Y_w = value


# ---------------------------------------------------------------------------
# 组分换算 / 理想气体密度
# ---------------------------------------------------------------------------
def mass_to_mole_frac(Y, Mw):
    """质量分数矩阵（n_cells, n_sp）→ 摩尔分数矩阵（n_cells, n_sp）。

    X_i = (Y_i/Mw_i) / Σ_j (Y_j/Mw_j)。Y 全零时返回全 0 行。
    """
    Y = np.asarray(Y, float)
    Mw = np.asarray(Mw, float)
    inv = Y / np.maximum(Mw[None, :], EPS)
    denom = inv.sum(axis=1, keepdims=True)
    out = np.zeros_like(Y)
    nz = denom[:, 0] > EPS
    if nz.any():
        out[nz] = inv[nz] / denom[nz]
    return out


def mole_to_mass_frac(X, Mw):
    """摩尔分数矩阵（n_cells, n_sp）→ 质量分数矩阵（n_cells, n_sp）。

    Y_i = (X_i Mw_i) / Σ_j (X_j Mw_j)。X 全零时返回全 0 行。
    """
    X = np.asarray(X, float)
    Mw = np.asarray(Mw, float)
    num = X * Mw[None, :]
    denom = num.sum(axis=1, keepdims=True)
    out = np.zeros_like(X)
    nz = denom[:, 0] > EPS
    if nz.any():
        out[nz] = num[nz] / denom[nz]
    return out


def mixture_molar_mass(Y, Mw):
    """混合摩尔质量 W_mix（n_cells, kg/kmol）= 1 / Σ_j (Y_j/Mw_j)。"""
    Y = np.asarray(Y, float)
    Mw = np.asarray(Mw, float)
    s = (Y / np.maximum(Mw[None, :], EPS)).sum(axis=1)
    return 1.0 / np.maximum(s, EPS)


def gas_density(W_mix, T, P=DEFAULT_P_REF, R=R_UNIV):
    """理想气体密度 rho = P W_mix / (R T)（kg/m^3）。标量或数组皆可。"""
    T = np.asarray(T, float)
    W = np.asarray(W_mix, float)
    return (float(P) * W) / (float(R) * np.maximum(T, EPS))


# ---------------------------------------------------------------------------
# 面有效扩散系数
# ---------------------------------------------------------------------------
def _face_gamma(fv, gamma_cell):
    """单元有效质量扩散系数（rho D_eff, kg/(m s)）→ 面值（均值为内部面，owner 为边界面）。"""
    gamma_cell = np.asarray(gamma_cell, float)
    is_int = fv.neighbor >= 0
    nbr = np.where(is_int, fv.neighbor, fv.owner)
    return 0.5 * (gamma_cell[fv.owner] + gamma_cell[nbr])


# ---------------------------------------------------------------------------
# 组分输运装配 / 求解
# ---------------------------------------------------------------------------
def assemble_species_transport(fv, mdot, gamma_cell, Y, source,
                               btype=None, bval=None, outlet_faces=None):
    """组装单组分质量分数输运方程（上风对流 + 中心扩散 + 源项 + 边界）。

    PDE（质量分数守恒）：div(rho u Y) - div(rho D_eff grad Y) = S。
      - `mdot` 为面质量通量 (n_faces, kg/s)；`gamma_cell` 为逐单元有效质量扩散
        系数 rho·D_eff（kg/(m s)）；`source` 为逐单元体积源 (n_cells, kg/(m^3 s))。
      - 边界类型见 `SP_*` 常量：零梯度（缺省，出口/壁面外推）与定值（Dirichlet，
        入口注入 Y_w = bval）。
      - `outlet_faces` 为出流面索引：出流 m>0 时按上风外推（仅对流对角加 m），
        不再附加扩散 Dirichlet。
    返回 (rows, cols, vals, rhs, ap)：未松弛系数 + 已并边界/源项的 rhs + 无松弛对角 ap。
    """
    n = fv.n_cells
    rows_l, cols_l, vals_l = [], [], []
    rhs = np.zeros(n, float)
    is_int = fv.neighbor >= 0
    o = fv.owner[is_int]
    nb = fv.neighbor[is_int]
    m = np.asarray(mdot, float)[is_int]
    gamma_cell = np.asarray(gamma_cell, float)
    gf = _face_gamma(fv, gamma_cell)
    D = (gf[is_int] * fv.face_area[is_int]
         / np.maximum(fv._d_n[is_int], EPS))
    pos = m >= 0.0
    rows_l.extend(o[pos]); cols_l.extend(o[pos]); vals_l.extend(m[pos].astype(float))
    rows_l.extend(nb[pos]); cols_l.extend(o[pos]); vals_l.extend((-m[pos]).astype(float))
    neg = ~pos
    rows_l.extend(o[neg]); cols_l.extend(nb[neg]); vals_l.extend(m[neg].astype(float))
    rows_l.extend(nb[neg]); cols_l.extend(nb[neg]); vals_l.extend((-m[neg]).astype(float))
    rows_l.extend(o); cols_l.extend(o); vals_l.extend(D)
    rows_l.extend(nb); cols_l.extend(nb); vals_l.extend(D)
    rows_l.extend(o); cols_l.extend(nb); vals_l.extend(-D)
    rows_l.extend(nb); cols_l.extend(o); vals_l.extend(-D)

    # 边界面：遍历分类（零梯度外推 / 定值 Dirichlet + 对流一致性）
    bnd_face = np.where(~is_int)[0]
    bo = fv.owner[~is_int]
    mb = mdot[~is_int]
    Db = (gf[~is_int] * fv.face_area[~is_int]
          / np.maximum(fv._d_n[~is_int], EPS))
    btype_arr = (np.full(fv.n_faces, SP_ZERO_GRAD, int) if btype is None
                 else np.asarray(btype, int).astype(int))
    bval_arr = (np.zeros(fv.n_faces, float) if bval is None
                else np.asarray(bval, float))
    outlet = np.zeros(len(bo), bool)
    if outlet_faces is not None and len(outlet_faces):
        outlet = np.isin(bnd_face, outlet_faces, assume_unique=False)

    for jj in range(len(bo)):
        gf_i = int(bnd_face[jj])
        oi = int(bo[jj])
        m_i = float(mb[jj])
        d_i = float(Db[jj])
        bt = int(btype_arr[gf_i])
        v = float(bval_arr[gf_i])
        if outlet[jj]:
            if m_i > 0.0:
                vals_l.append(m_i); rows_l.append(oi); cols_l.append(oi)
            continue
        if bt == SP_ZERO_GRAD:
            if m_i > 0.0:
                vals_l.append(m_i); rows_l.append(oi); cols_l.append(oi)
        elif bt == SP_FIXED:
            vals_l.append(d_i); rows_l.append(oi); cols_l.append(oi)
            rhs[oi] += d_i * v
            if m_i > 0.0:
                vals_l.append(m_i); rows_l.append(oi); cols_l.append(oi)
            else:
                rhs[oi] -= m_i * v

    rhs += np.asarray(source, float) * fv.volumes
    rows = np.array(rows_l, np.int64)
    cols = np.array(cols_l, np.int64)
    vals = np.array(vals_l, float)
    ap = np.zeros(n, float)
    d_idx = rows == cols
    np.add.at(ap, rows[d_idx], vals[d_idx])
    return rows, cols, vals, rhs, ap


def solve_species(fv, mdot, gamma_cell, Y, source, btype=None, bval=None,
                  outlet_faces=None, relax=0.7, tol=1e-9, maxit=6000):
    """求解一次单组分质量分数方程（欠松弛），返回 (Y_new, ap)。"""
    rows, cols, vals, rhs, ap = assemble_species_transport(
        fv, mdot, gamma_cell, Y, source, btype, bval, outlet_faces)
    n = fv.n_cells
    use = np.asarray(Y, float)
    keep = rows != cols
    rows = rows[keep]; cols = cols[keep]; vals = vals[keep]
    aP_relaxed = ap / float(relax)
    rows = np.concatenate([rows, np.arange(n, dtype=np.int64)])
    cols = np.concatenate([cols, np.arange(n, dtype=np.int64)])
    vals = np.concatenate([vals, aP_relaxed])
    rhs = rhs + ((1.0 - relax) / relax) * ap * use
    Y_new = solve_linear(rows, cols, vals, rhs, n, tol=tol, maxit=maxit)
    return Y_new, ap


# ---------------------------------------------------------------------------
# 组分边界分类
# ---------------------------------------------------------------------------
def classify_species(fv, flow_axis=0, inlet_side="min", outlet_side="max"):
    """按流向坐标划分入口/出口/壁面边界面（全覆盖不重叠）。"""
    ax = fv.face_centroid[:, flow_axis]
    cut_min = float(ax.min())
    cut_max = float(ax.max())
    bnd = fv.is_boundary.copy()
    inlet = bnd.copy()
    if inlet_side == "min":
        inlet &= ax <= cut_min + 1e-10
    else:
        inlet &= ax >= cut_max - 1e-10
    outlet = bnd.copy()
    if outlet_side == "max":
        outlet &= ax >= cut_max - 1e-10
    else:
        outlet &= ax <= cut_min + 1e-10
    outlet &= ~inlet
    wall = bnd & ~inlet & ~outlet
    return np.where(inlet)[0], np.where(outlet)[0], np.where(wall)[0], \
        np.where(bnd)[0]


# ---------------------------------------------------------------------------
# 组分输运求解器
# ---------------------------------------------------------------------------
class SpeciesSolver:
    """多组分质量分数输运求解器：N-1 个输运方程 + 补齐组分 + 有界投影。

    `update(u, v, w, mdot, source)` 由给定混合流场与（可选）化学反应源推进各组分
    质量分数，返回归一化残差。`Y` / `mass_fractions` 暴露质量分数矩阵（n_cells, n_sp）；
    `X` / `mole_fractions` 为摩尔分数矩阵；`W_mix` 为混合摩尔质量（kg/kmol）；`rho`
    为按理想气体密度公式（由当前 W_mix、T 与参考压力）得到的逐单元密度（诊断/后处理用，
    不影响 SIMPLE 单相参考密度）。
    """

    def __init__(self, fv, names=None, Mw=None, D_mol=DEFAULT_SCALAR_DIFF,
                 sc_t=DEFAULT_SC_TURB, rho=None, P_ref=DEFAULT_P_REF,
                 T_ref=DEFAULT_T_REF, flow_axis=0, inlet_side="min",
                 outlet_side="max", inlet_fractions=None, Y0=None,
                 relax=0.7, max_iter=50):
        self.fv = fv
        if names is None:
            names = list(DEFAULT_NAMES)
        if Mw is None:
            Mw = list(DEFAULT_MW)
        names = list(names)
        Mw = np.asarray(Mw, float)
        if len(names) != len(Mw):
            raise ValueError("Species names/Mw 长度不一致：%d vs %d"
                             % (len(names), len(Mw)))
        if len(Mw) < 2:
            raise ValueError("Species 至少需 2 类（含补齐组分）")
        self._names = names
        self._Mw = Mw
        self.d_mol = float(D_mol)
        self.sc_t = float(sc_t)
        self._rho0 = None if rho is None else float(rho)
        self.P_ref = float(P_ref)
        self.T_ref = float(T_ref)
        self.relax = float(relax)
        self.max_iter = int(max_iter)
        self.flow_axis = int(flow_axis)
        self.inlet_side = inlet_side
        self.outlet_side = outlet_side
        n = fv.n_cells
        ns = len(Mw)
        # 入口质量分数（缺省为空气），长度 ns-1（补齐组分不设入口）
        iy = np.zeros(ns, float)
        if inlet_fractions is None:
            iy[:] = np.asarray(DEFAULT_FRACTIONS, float)[:ns]
        else:
            iy = np.asarray(inlet_fractions, float).ravel()
            if len(iy) != ns:
                raise ValueError("Species 入口质量分数应长度 %d" % ns)
        iy = np.clip(iy, 0.0, 1.0)
        s = iy.sum()
        if s > 1.0:
            iy = iy / s
        elif iy[ns - 1] <= 0.0:
            iy[ns - 1] = 1.0 - iy[:ns - 1].sum()
        self.inlet_fractions = iy
        # 初值质量分数（缺省入口质量分数）
        y0 = np.zeros((n, ns), float)
        if Y0 is None:
            y0[:] = self.inlet_fractions[None, :]
        else:
            y0 = np.asarray(Y0, float)
            if y0.shape != (n, ns):
                raise ValueError("Species 初值质量分数应形状 (%d, %d)" % (n, ns))
            y0 = np.clip(y0, 0.0, 1.0)
            s0 = y0.sum(axis=1, keepdims=True)
            y0 = y0 / np.maximum(s0, EPS)
        self._Y = y0
        self._inlet_faces, self._outlet_faces, self._wall_faces, self._bnd = \
            classify_species(fv, flow_axis, inlet_side, outlet_side)
        self._gamma = np.full(n, self._rho() * self.d_mol, float)
        self._residual = 0.0
        self.iteration = 0
        self._u = np.zeros(n, float)
        self._v = np.zeros(n, float)
        self._w = np.zeros(n, float)
        self._mdot = None
        self._nu_t = None

    # -- 场属性 ------------------------------------------------
    @property
    def names(self):
        return list(self._names)

    @property
    def Mw(self):
        return self._Mw

    @property
    def Y(self):
        return self._Y

    @property
    def mass_fractions(self):
        return self._Y

    @property
    def X(self):
        return mass_to_mole_frac(self._Y, self._Mw)

    @property
    def mole_fractions(self):
        return self.X

    @property
    def W_mix(self):
        return mixture_molar_mass(self._Y, self._Mw)

    @property
    def n_species(self):
        return len(self._Mw)

    @property
    def rho(self):
        """理想气体混合密度（n_cells, kg/m^3）——诊断/后处理用。"""
        return self._rho()

    def _rho(self):
        if self._rho0 is not None:
            return np.full(self.fv.n_cells, self._rho0, float)
        return gas_density(self.W_mix, self.T_ref, self.P_ref)

    # -- 边界 / 扩散 --------------------------------------------
    def set_velocities(self, u, v, w, mdot=None, nu_t=None):
        self._u = np.asarray(u, float).ravel()
        self._v = np.asarray(v, float).ravel()
        self._w = np.asarray(w, float).ravel()
        self._mdot = None if mdot is None else np.asarray(mdot, float)
        self._nu_t = None if nu_t is None else np.asarray(nu_t, float)

    def set_inlet_fractions(self, inlet_fractions):
        iy = np.asarray(inlet_fractions, float).ravel()
        if len(iy) != self.n_species:
            raise ValueError("Species 入口质量分数应长度 %d" % self.n_species)
        iy = np.clip(iy, 0.0, 1.0)
        s = iy.sum()
        if s > 1.0:
            iy = iy / s
        self.inlet_fractions = iy
        return self

    def _effective_gamma(self, nu_t):
        gamma = self._rho() * self.d_mol
        if nu_t is None or np.asarray(nu_t).size == 0:
            self._gamma[:] = gamma
            return self._gamma
        nu_t = np.asarray(nu_t, float).ravel()
        return gamma + self._rho() * nu_t / max(self.sc_t, EPS)

    def _boundary_species(self, k):
        fv = self.fv
        btype = np.full(fv.n_faces, SP_ZERO_GRAD, int)
        bval = np.zeros(fv.n_faces, float)
        if len(self._inlet_faces):
            btype[self._inlet_faces] = SP_FIXED
            bval[self._inlet_faces] = float(self.inlet_fractions[k])
        return btype, bval

    def update(self, u, v, w, mdot=None, source=None, nu_t=None):
        """由混合流场推进一次各组分质量分数（N-1 输运 + 补齐），返回归一化残差。

        `source` 为逐单元化学反应源（n_cells, n_sp, kg/(m^3 s)）；缺省为 0 的纯输运。
        """
        fv = self.fv
        y_old = np.asarray(self._Y, float)
        n = fv.n_cells
        ns = self.n_species
        transported = ns - 1
        if mdot is None:
            mdot = volume_mass(fv, u, v, w, self._rho())
        mdot = np.asarray(mdot, float)
        gamma_eff = self._effective_gamma(nu_t)
        y_new = np.zeros((n, ns), float)
        for k in range(transported):
            btype, bval = self._boundary_species(k)
            if source is None:
                s_k = np.zeros(n, float)
            else:
                s_k = np.asarray(source, float)[:, k]
            yk, _ap = solve_species(fv, mdot, gamma_eff, y_old[:, k], s_k,
                                    btype=btype, bval=bval,
                                    outlet_faces=self._outlet_faces,
                                    relax=self.relax)
            y_new[:, k] = np.clip(yk, 0.0, 1.0)
        # 补齐组分 + 有界投影保 Σ Y = 1
        y_new[:, transported] = 1.0 - np.sum(y_new[:, :transported], axis=1)
        y_new = np.clip(y_new, 0.0, 1.0)
        s = np.sum(y_new, axis=1, keepdims=True)
        y_new = y_new / np.maximum(s, EPS)
        denom = max(float(_safe_norm(y_new)), EPS)
        resid = float(_safe_norm(y_new - y_old)) / denom
        self._Y = y_new
        self.iteration += 1
        self._residual = resid
        self._update_rho()
        return resid

    def _update_rho(self):
        self._gamma[:] = self._rho() * self.d_mol

    # -- P10 后端兼容门面 ----------------------------------------
    def _initialize_field(self):
        self._Y[:] = self.inlet_fractions[None, :]
        self._residual = 0.0
        self.iteration = 0
        self._update_rho()
        return dict(n_species=self.n_species,
                    y_min=float(self._Y.min()), y_max=float(self._Y.max()),
                    w_mix=float(self.W_mix.mean()))

    def step(self):
        resid = float(self.update(self._u, self._v, self._w, mdot=self._mdot,
                                  nu_t=self._nu_t))
        self._residual = resid
        return dict(residual=resid, n_species=self.n_species,
                    y_min=float(self._Y.min()), y_max=float(self._Y.max()),
                    w_mix=float(self.W_mix.mean()))

    def residual(self):
        return self._residual

    def monitor_payload(self):
        return dict(n_species=self.n_species,
                    y_min=float(self._Y.min()), y_max=float(self._Y.max()),
                    sum_min=float(self._Y.sum(axis=1).min()),
                    sum_max=float(self._Y.sum(axis=1).max()),
                    w_mix_min=float(self.W_mix.min()),
                    w_mix_max=float(self.W_mix.max()),
                    iteration=self.iteration, residual=self._residual)


def volume_mass(fv, u, v, w, rho):
    """由速度场与密度场重构面质量通量 mdot = rho * u·n * A（n_faces,）。"""
    u = np.asarray(u, float).ravel()
    v = np.asarray(v, float).ravel()
    w = np.asarray(w, float).ravel()
    rho = np.asarray(rho, float).ravel()
    normals = fv.face_normal
    area = fv.face_area
    fu = fv.face_value(u)
    fv_c = fv.face_value(v)
    fw = fv.face_value(w)
    fr = fv.face_value(rho)
    un = fu * normals[:, 0] + fv_c * normals[:, 1] + fw * normals[:, 2]
    return fr * un * area


# ---------------------------------------------------------------------------
# 工厂注册表
# ---------------------------------------------------------------------------
_SPECIES_MODELS = {
    "species": SpeciesSolver,
    "multispecies": SpeciesSolver,
    "mass_fraction": SpeciesSolver,
}


def make_species(fv, model="species", **kwargs):
    """按字符串名实例化组分输运模型（大小写/下划线不敏感）。"""
    key = str(model).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _SPECIES_MODELS:
        names = ", ".join(sorted(set(_SPECIES_MODELS.keys())))
        raise ValueError("未知组分输运模型 %r（支持：%s）" % (model, names))
    return _SPECIES_MODELS[key](fv, **kwargs)

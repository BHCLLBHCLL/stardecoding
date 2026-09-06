# -*- coding: utf-8 -*-
"""P 波 P6：湍流族 —— Spalart-Allmaras → k-ε → k-ω SST → LES 子格子 + 壁面处理。

纯 numpy 实现（occ / scdm 两环境皆可用），建立在 P4 有限体积离散核心
（`fvm_core.FVM`）与 P5 压力基求解器（`pressure_solver.solve_linear`）之上，
为动量方程提供**湍流粘性** ν_t 场，并可在 SIMPLE 外环内求解湍流输运方程。

核心能力：

  1) 壁面处理：`wall_distance`、`y_plus`（无量纲壁面距离）、`wall_function`
     （对数律/粘性底层光滑混合）、`wall_shear`（壁面剪切应力）。
  2) 湍流模型：SA 一方程、k-ε 两方程、k-ω SST（Menter F1/F2 混合）、LES Smagorinsky
     子格子。所有模型通过 `update(...)` 返回输运方程残差，并暴露 `nu_t` 场。
  3) 标量输运装配：`assemble_scalar_transport` + `solve_scalar` 将湍流输运方程
     隐式离散为稀疏三线性（上风对流 + 中心扩散 + 源项），复用 `solve_linear`。
  4) P10 后端兼容：`TurbulenceSolver` 提供 `_initialize_field() / step() /
     residual() / monitor_payload()`，可被 `solver_run.SolverBackend` 驱动。

纯 numpy 约束：所有范数走 `solver_run._safe_norm`，不依赖 scipy / np.linalg.norm。
"""
import numpy as np

from fvm_core import FVM
from pressure_solver import solve_linear
from solver_run import _safe_norm


# ---------------------------------------------------------------------------
# 湍流常数 / 壁面常数
# ---------------------------------------------------------------------------
WALL_KAPPA = 0.41    # 冯·卡门常数
WALL_E = 9.8         # 对数律粗糙度光滑参数
_YPLUS_LAM = 11.06   # 粘性底层厚度（对数律起点）

EPS = 1e-12


# ---------------------------------------------------------------------------
# 壁面处理：壁面距离 / 无量纲壁面距离 / 壁面函数 / 壁面剪切
# ---------------------------------------------------------------------------
def wall_distance(fv, wall_faces=None):
    """每个单元质心到最近壁面 (边界面质心) 的距离。

    `wall_faces` 为可选壁面边界面的全局面索引数组；缺省用全部边界面。
    返回长度 (n_cells,) 非负数组。无壁面时返回全 1（避免除零）。
    """
    bnd = np.where(fv.is_boundary)[0]
    if wall_faces is not None:
        bnd = bnd[np.isin(bnd, wall_faces, assume_unique=False)]
    if len(bnd) == 0:
        return np.full(fv.n_cells, 1.0, float)
    fc = fv.face_centroid[bnd]
    P = fv.centroids[:, None, :] - fc[None, :, :]
    d = _safe_norm(P, axis=2)
    return d.min(axis=1)


def y_plus(d, u_tau, nu):
    """无量纲壁面距离 y+ = d * u_tau / nu（d/u_tau/nu 均可为标量或数组）。

    返回与输入广播一致的数组。nu 取最小截断 1e-14 防除零。
    """
    d = np.asarray(d, float)
    u_tau = np.asarray(u_tau, float)
    nu = np.maximum(np.asarray(nu, float), 1e-14)
    return d * u_tau / nu


def wall_function(yplus, kappa=WALL_KAPPA, E=WALL_E):
    """壁面函数：粘性底层 u+=y+ 到对数律 u+=ln(E·y+)/κ 的**光滑混合**。

    以 `_YPLUS_LAM` 为过渡中心做线性插值带混叠，单调且连续，便于测试：
      - y+ < y+_lam  → 近粘性底层 (u+ 略偏线性)；
      - y+ > y+_lam  → 近对数律。
    返回与输入广播一致的数组。
    """
    yplus = np.maximum(np.asarray(yplus, float), 1e-12)
    uplus_vis = yplus
    uplus_log = np.log(np.maximum(E * yplus, 1e-12)) / kappa
    span = max(_YPLUS_LAM, 1e-12)
    t = np.clip((yplus - _YPLUS_LAM) / span, 0.0, 1.0)
    return (1.0 - t) * uplus_vis + t * uplus_log


def wall_shear(u_tau, rho):
    """壁面剪切应力 tau_w = rho * u_tau^2（u_tau/rho 可为标量或数组）。"""
    u_tau = np.asarray(u_tau, float)
    rho = np.asarray(rho, float)
    return rho * u_tau * u_tau


# ---------------------------------------------------------------------------
# 边界分类 / 面通量 / 应变率
# ---------------------------------------------------------------------------
def classify_boundaries(fv, flow_axis=0, inlet_side="min", outlet_side="max"):
    """按流向坐标把边界面分为 入口 / 出口 / 壁面。

    判据：流向坐标靠近整体最小（或最大，取决于 inlet_side）的面为入口，
    靠近另一端为出口，其余边界为壁面。返回
    (inlet, outlet, wall, bnd) —— 四者均为全局面索引数组。
    """
    bnd = np.where(fv.is_boundary)[0]
    if len(bnd) == 0:
        return bnd, np.array([], np.int64), np.array([], np.int64), bnd
    fc = fv.face_centroid[bnd]
    coord = fc[:, flow_axis]
    lo = float(coord.min())
    hi = float(coord.max())
    tol = 1e-9 * max(1.0, abs(hi - lo))
    if inlet_side == "min":
        inlet_mask = np.abs(coord - lo) <= tol
        outlet_mask = np.abs(coord - hi) <= tol
    else:
        inlet_mask = np.abs(coord - hi) <= tol
        outlet_mask = np.abs(coord - lo) <= tol
    inlet = bnd[inlet_mask]
    outlet = bnd[outlet_mask]
    wall = bnd[~(inlet_mask | outlet_mask)]
    return inlet, outlet, wall, bnd


def face_volume_flux(fv, u, v, w):
    """面体积通量 q = (u_f·n_f) * A_f（对速度场做线性面插值）。

    u/v/w 为单元速度分量场，返回长度 (n_faces,) 的体通量（m^3/s，正值沿
    owner → neighbor 法向）。
    """
    uf = fv.face_value(np.asarray(u, float))
    vf = fv.face_value(np.asarray(v, float))
    wf = fv.face_value(np.asarray(w, float))
    vn = (uf * fv.face_normal[:, 0] + vf * fv.face_normal[:, 1]
          + wf * fv.face_normal[:, 2])
    return vn * fv.face_area


def strain_magnitude(fv, u, v, w):
    """应变率幅值 |S| = sqrt(2 S_ij S_ij)，用 Green-Gauss 梯度构造对称张量。"""
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    w = np.asarray(w, float)
    gu = fv.grad_gauss(u)
    gv = fv.grad_gauss(v)
    gw = fv.grad_gauss(w)
    S11 = gu[:, 0]
    S22 = gv[:, 1]
    S33 = gw[:, 2]
    S12 = 0.5 * (gu[:, 1] + gv[:, 0])
    S13 = 0.5 * (gu[:, 2] + gw[:, 0])
    S23 = 0.5 * (gv[:, 2] + gw[:, 1])
    return np.sqrt(2.0 * (S11 * S11 + S22 * S22 + S33 * S33
                          + 2.0 * (S12 * S12 + S13 * S13 + S23 * S23)))


# ---------------------------------------------------------------------------
# 标量输运装配 / 求解（湍流输运方程的中心离散）
# ---------------------------------------------------------------------------
def _face_diffusivity(fv, nu_cell):
    """单元有效扩散系数 -> 面有效扩散系数（内部面取均值，边界面取 owner 值）。"""
    nu_cell = np.asarray(nu_cell, float)
    is_int = fv.neighbor >= 0
    nbr = np.where(is_int, fv.neighbor, fv.owner)
    return 0.5 * (nu_cell[fv.owner] + nu_cell[nbr])


def assemble_scalar_transport(fv, qdot, nu_cell, prod, diss, bval,
                              outlet_faces=None, source=None):
    """组装保守标量输运方程（上风对流 + 中心扩散 + 源/汇），返回 COO 三线性。

    PDE（运动学形式）：div(phi q) - div(nu_eff grad phi) = prod - diss*phi + source，
      其中 q 为面体积通量 (n_faces,)、nu_eff 为面向有效运动扩散系数、
      prod/diss/source 均为逐单元数组。diss 以对角衰减形式隐式化。

    返回 (rows, cols, vals, rhs, ap)：rows/cols/vals 为未松弛系数阵列，
    rhs 已并入生产/边界贡献，ap 为含扩散汇的无松弛对角。
    """
    n = fv.n_cells
    rows_l, cols_l, vals_l = [], [], []
    rhs = np.zeros(n, float)
    is_int = fv.neighbor >= 0
    o = fv.owner[is_int]
    nb = fv.neighbor[is_int]
    m = np.asarray(qdot, float)[is_int]
    nu_f = _face_diffusivity(fv, nu_cell)
    D = (nu_f[is_int] * fv.face_area[is_int]
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
    bo = fv.owner[~is_int]
    rows_l.extend(nb); cols_l.extend(o); vals_l.extend(-D)
    bnd_face = np.where(~is_int)[0]
    mb = np.asarray(qdot, float)[~is_int]
    Db = (nu_f[~is_int] * fv.face_area[~is_int]
          / np.maximum(fv._d_n[~is_int], EPS))
    bvals = np.asarray(bval, float)[~is_int]
    outlet = np.zeros(len(bo), bool)
    if outlet_faces is not None and len(outlet_faces):
        outlet = np.isin(bnd_face, outlet_faces, assume_unique=False)
    out_conv = outlet & (mb >= 0.0)
    rows_l.extend(bo[out_conv]); cols_l.extend(bo[out_conv])
    vals_l.extend(mb[out_conv].astype(float))
    conv_rhs = ~out_conv
    idx = np.where(conv_rhs)[0]
    rhs[bo[idx]] -= mb[idx] * bvals[idx]
    diff_b = ~outlet
    rows_l.extend(bo[diff_b]); cols_l.extend(bo[diff_b]); vals_l.extend(Db[diff_b])
    idx2 = np.where(diff_b)[0]
    rhs[bo[idx2]] += Db[idx2] * bvals[idx2]
    rhs += np.asarray(prod, float) * fv.volumes
    rows = np.array(rows_l, np.int64)
    cols = np.array(cols_l, np.int64)
    vals = np.array(vals_l, float)
    ap = np.zeros(n, float)
    d_idx = rows == cols
    np.add.at(ap, rows[d_idx], vals[d_idx])
    sink = np.clip(np.asarray(diss, float), 0.0, None) * fv.volumes
    ap = ap + sink
    if source is not None:
        rhs += np.asarray(source, float) * fv.volumes
    return rows, cols, vals, rhs, ap


def solve_scalar(fv, qdot, nu_cell, phi, prod, diss, bval, outlet_faces=None,
                 source=None, relax=0.7, tol=1e-9, maxit=6000):
    """求解一次标量输运方程（欠松弛），返回 (phi_new, ap)。"""
    rows, cols, vals, rhs, ap = assemble_scalar_transport(
        fv, qdot, nu_cell, prod, diss, bval, outlet_faces, source)
    n = fv.n_cells
    use = np.asarray(phi, float)
    keep = rows != cols
    rows = rows[keep]; cols = cols[keep]; vals = vals[keep]
    aP_relaxed = ap / float(relax)
    rows = np.concatenate([rows, np.arange(n, dtype=np.int64)])
    cols = np.concatenate([cols, np.arange(n, dtype=np.int64)])
    vals = np.concatenate([vals, aP_relaxed])
    rhs = rhs + ((1.0 - relax) / relax) * ap * use
    phi_new = solve_linear(rows, cols, vals, rhs, n, tol=tol, maxit=maxit)
    return phi_new, ap
# ---------------------------------------------------------------------------
# 湍流模型基类
# ---------------------------------------------------------------------------
class TurbulenceModelBase:
    """湍流模型基类：壁面距离、摩擦速度、nu_t 场与边界值抽象。

    子类至少实现 `reset()`（初始化输运场）与 `update(u, v, w)`（一次湍流
    迭代，返回归一化残差）。`nu_t` 为湍流运动粘性场（(n_cells,)，>=0）。
    """

    def __init__(self, fv, rho=1.0, mu=1e-3, flow_axis=0,
                 inlet_side="min", outlet_side="max",
                 turbulence_intensity=0.05, length_scale=0.1, u_ref=1.0):
        self.fv = fv
        self.rho = float(rho)
        self.nu = float(mu) / max(float(rho), 1e-14)
        self.flow_axis = int(flow_axis)
        self.u_ref = float(u_ref)
        self.turbulence_intensity = float(turbulence_intensity)
        self.length_scale = float(length_scale)
        self.inlet_faces, self.outlet_faces, self.wall_faces, self.bnd = \
            classify_boundaries(fv, self.flow_axis, inlet_side, outlet_side)
        self._d = wall_distance(fv, self.wall_faces)
        if self._d.min() <= 0.0:
            self._d = np.maximum(self._d, EPS)
        self._nu_t = np.zeros(fv.n_cells, float)
        self.iteration = 0
        self._last_residual = 0.0
        self.reset()

    def reset(self):
        """初始化输运场。子类覆写。"""
        raise NotImplementedError

    def estimate_u_tau(self):
        """摩擦速度估计 u_tau ~ 0.05 * u_ref（湍流边界层典型量级），供 y+ 计算。"""
        return float(0.05 * self.u_ref)

    def yplus(self):
        """逐单元 y+（基于壁面距离与估计摩擦速度）。"""
        return y_plus(self._d, self.estimate_u_tau(), self.nu)

    @property
    def nu_t(self):
        return self._nu_t

    def update(self, u, v, w):
        """一次湍流迭代，返回归一化残差。子类覆写。"""
        raise NotImplementedError

    def _inlet_value(self):
        """入口边界的湍流量量级（由湍流强度与长度尺度给出）。"""
        return float(self.turbulence_intensity * self.u_ref * self.length_scale)

    def _boundary_value(self, value):
        """构造全场边界值数组：入口取给定值，壁面取 0（边界层贴合）。"""
        bval = np.zeros(self.fv.n_faces, float)
        if len(self.inlet_faces):
            bval[self.inlet_faces] = float(np.asarray(value, float).ravel().mean())
        return bval

    def _make_bval(self, value):
        return self._boundary_value(value)
# ---------------------------------------------------------------------------
# SA：Spalart-Allmaras 一方程模型
# ---------------------------------------------------------------------------
class SpalartAllmarasSolver(TurbulenceModelBase):
    """Spalart-Allmaras 一方程：输运修正涡粘 nu_tilde。

    nu_t = nu_tilde * f_v1, f_v1 = chi^3/(chi^3 + c_v1^3), chi = nu_tilde/nu。
    生产 P = c_b1 S_tilde nu_tilde；破坏 D (对角) = c_w1 f_w nu_tilde / d^2。
    常数取经典取值，c_w1 由 c_w1 = c_b1/kappa^2 + (1+c_b2)/sigma 导出。
    """
    c_b1 = 0.1355
    c_b2 = 0.622
    sigma = 2.0 / 3.0
    c_w2 = 0.3
    c_w3 = 2.0
    kappa = WALL_KAPPA
    c_v1 = 7.1
    c_w1 = c_b1 / (WALL_KAPPA ** 2) + (1.0 + c_b2) / sigma

    def reset(self):
        v0 = max(self._inlet_value(), EPS)
        self._nu_tilde = np.full(self.fv.n_cells, v0, float)
        self._nu_t = self._f_v1(self._nu_tilde / self.nu) * self._nu_tilde
        self.iteration = 0
        self._last_residual = 0.0

    def _f_v1(self, chi):
        chi = np.maximum(np.asarray(chi, float), EPS)
        c3 = self.c_v1 ** 3
        return chi ** 3 / (chi ** 3 + c3)

    def _f_v2(self, chi):
        chi = np.maximum(np.asarray(chi, float), EPS)
        return 1.0 - chi / (1.0 + chi * self._f_v1(chi))

    def _f_w(self, g):
        c3w = self.c_w3 ** 6
        g = np.maximum(np.asarray(g, float), 1e-12)
        g = np.minimum(g, 1e6)
        return g * ((1.0 + c3w) / (g ** 6 + c3w)) ** (1.0 / 6.0)
    def update(self, u, v, w):
        fv = self.fv
        qdot = face_volume_flux(fv, u, v, w)
        S = strain_magnitude(fv, u, v, w)
        nu_tilde = self._nu_tilde
        nu = self.nu
        chi = np.maximum(nu_tilde / nu, EPS)
        nu_eff_cell = nu + np.maximum(nu_tilde, 0.0)
        k2d2 = self.kappa ** 2 * np.maximum(self._d, EPS) ** 2
        S_tilde = np.maximum(S + nu_tilde / k2d2 * self._f_v2(chi), EPS)
        prod = self.c_b1 * S_tilde * nu_tilde
        r = nu_tilde / (k2d2 * S_tilde)
        g = r + self.c_w2 * (r ** 6 - r)
        fw = self._f_w(g)
        diss = self.c_w1 * fw * nu_tilde / np.maximum(self._d, EPS) ** 2
        diss = np.clip(diss, 0.0, None)
        bval = self._boundary_value(self._inlet_value())
        nu_tilde_new, _ = solve_scalar(
            fv, qdot, nu_eff_cell, nu_tilde, prod, diss, bval,
            self.outlet_faces, relax=0.6)
        nu_tilde_new = np.maximum(nu_tilde_new, 0.0)
        denom = max(float(_safe_norm(nu_tilde_new)), EPS)
        resid = float(_safe_norm(nu_tilde_new - nu_tilde)) / denom
        self._nu_tilde = nu_tilde_new
        self._nu_t = self._f_v1(nu_tilde_new / nu) * nu_tilde_new
        self.iteration += 1
        self._last_residual = resid
        return resid
# ---------------------------------------------------------------------------
# k-epsilon：两方程模型
# ---------------------------------------------------------------------------
class KEpsilonSolver(TurbulenceModelBase):
    """k-ε 两方程：输运湍动能 k 与耗散率 ε。

    nu_t = C_mu k^2 / eps；P_k = nu_t |S|^2。
    k 方程：prod = P_k, diss = eps/k（对角）；
    ε 方程：prod = C_1 (eps/k) P_k, diss = C_2 (eps/k)（对角）。
    """
    C_mu = 0.09
    C_1 = 1.44
    C_2 = 1.92
    sigma_k = 1.0
    sigma_eps = 1.3

    def reset(self):
        k_in = self._k_inlet()
        e_in = self._eps_inlet(k_in)
        self._k = np.full(self.fv.n_cells, k_in, float)
        self._eps = np.full(self.fv.n_cells, e_in, float)
        self._nu_t = (self.C_mu * self._k ** 2
                      / np.maximum(self._eps, EPS))
        self.iteration = 0
        self._last_residual = 0.0

    def _k_inlet(self):
        return 1.5 * (self.turbulence_intensity * self.u_ref) ** 2

    def _eps_inlet(self, k):
        return self.C_mu ** 0.75 * np.maximum(k, EPS) ** 1.5 / max(self.length_scale, EPS)

    @property
    def k(self):
        return self._k

    @property
    def epsilon(self):
        return self._eps

    def update(self, u, v, w):
        fv = self.fv
        qdot = face_volume_flux(fv, u, v, w)
        S = strain_magnitude(fv, u, v, w)
        k = np.maximum(self._k, EPS)
        eps = np.maximum(self._eps, EPS)
        nu_t = self.C_mu * k ** 2 / np.maximum(eps, EPS)
        Pk = nu_t * S ** 2
        bval_k = self._boundary_value(self._k_inlet())
        nu_k_eff = self.nu + nu_t / self.sigma_k
        k_new, _ = solve_scalar(fv, qdot, nu_k_eff, k, Pk, eps / np.maximum(k, EPS),
                                bval_k, self.outlet_faces, relax=0.6)
        k_new = np.maximum(k_new, EPS)
        nu_t = self.C_mu * k_new ** 2 / np.maximum(eps, EPS)
        Pk = nu_t * S ** 2
        e_prod = self.C_1 * (eps / np.maximum(k, EPS)) * Pk
        e_diss = self.C_2 * (eps / np.maximum(k, EPS))
        bval_e = self._boundary_value(self._eps_inlet(self._k_inlet()))
        nu_e_eff = self.nu + nu_t / self.sigma_eps
        eps_new, _ = solve_scalar(fv, qdot, nu_e_eff, eps, e_prod, e_diss,
                                  bval_e, self.outlet_faces, relax=0.6)
        eps_new = np.maximum(eps_new, EPS)
        self._k = k_new
        self._eps = eps_new
        self._nu_t = self.C_mu * k_new ** 2 / np.maximum(eps_new, EPS)
        denom = max(float(_safe_norm(k_new)) + float(_safe_norm(eps_new)), EPS)
        resid = (float(_safe_norm(k_new - k)) + float(_safe_norm(eps_new - eps))) / denom
        self.iteration += 1
        self._last_residual = resid
        return resid# ---------------------------------------------------------------------------
# k-omega SST：Menter 两方程（F1/F2 混合）
# ---------------------------------------------------------------------------
class KOmegaSSTSolver(TurbulenceModelBase):
    """k-ω SST（Menter）：近壁 ω 方程、远场 ε 方程经 F1 切换的剪切应力输运。

    nu_t = a1 k / max(a1 omega, S F2)；k 方程生产 P_k = min(nu_t S^2, 10 beta* k omega)，
    ω 方程生产 gamma P_k / nu_t，ω 方程额外含交叉扩散 CD_kw = 2 sigma_w2 (1-F1) grad k·grad omega / omega。
    """
    sigma_k1 = 0.85
    sigma_k2 = 1.0
    sigma_w1 = 0.5
    sigma_w2 = 0.856
    beta_star = 0.09
    beta1 = 0.075
    beta2 = 0.0828
    a1 = 0.31
    gamma1 = 5.0 / 9.0
    gamma2 = 0.44

    def reset(self):
        k_in = self._k_inlet()
        w_in = self._omega_inlet(k_in)
        self._k = np.full(self.fv.n_cells, k_in, float)
        self._omega = np.full(self.fv.n_cells, w_in, float)
        self._nu_t = (self.a1 * self._k
                      / np.maximum(self.a1 * self._omega, EPS))
        self.iteration = 0
        self._last_residual = 0.0

    def _k_inlet(self):
        return 1.5 * (self.turbulence_intensity * self.u_ref) ** 2

    def _omega_inlet(self, k):
        return np.sqrt(np.maximum(k, EPS)) / (
            self.beta_star ** 0.25 * max(self.length_scale, EPS))

    def update(self, u, v, w):
        fv = self.fv
        qdot = face_volume_flux(fv, u, v, w)
        S = strain_magnitude(fv, u, v, w)
        k = np.maximum(self._k, EPS)
        om = np.maximum(self._omega, EPS)
        d = np.maximum(self._d, EPS)
        nu = self.nu
        bs = self.beta_star
        bval_k0 = self._boundary_value(self._k_inlet())
        bval_w0 = self._boundary_value(self._omega_inlet(self._k_inlet()))
        grad_k0 = fv.grad_gauss(k, boundary=bval_k0)
        grad_w0 = fv.grad_gauss(om, boundary=bval_w0)
        cd_kw = np.maximum(2.0 * self.sigma_w2
                           * np.sum(grad_k0 * grad_w0, axis=1)
                           / np.maximum(om, EPS), 1e-10)
        arg1a = np.sqrt(k) / (bs * om * d)
        arg1b = 500.0 * nu / (om * d * d)
        arg1c = 4.0 * self.sigma_w2 * k / (cd_kw * d * d)
        F1 = np.tanh(np.minimum(np.maximum(arg1a, arg1b), arg1c) ** 4)
        arg2 = np.maximum(2.0 * np.sqrt(k) / (bs * om * d),
                          500.0 * nu / (om * d * d))
        F2 = np.tanh(arg2 ** 2)
        sigma_k = F1 * self.sigma_k1 + (1 - F1) * self.sigma_k2
        sigma_w = F1 * self.sigma_w1 + (1 - F1) * self.sigma_w2
        beta = F1 * self.beta1 + (1 - F1) * self.beta2
        gamma = F1 * self.gamma1 + (1 - F1) * self.gamma2
        nu_t = self.a1 * k / np.maximum(self.a1 * om, S * F2 + EPS)
        Pk = np.minimum(nu_t * S * S, 10.0 * bs * k * om)
        nu_k_eff = nu + sigma_k * nu_t
        k_new, _ = solve_scalar(fv, qdot, nu_k_eff, k, Pk, bs * om,
                                bval_k0, self.outlet_faces, relax=0.5)
        k_new = np.maximum(k_new, EPS)
        grad_k1 = fv.grad_gauss(k, boundary=bval_k0)
        cd_kw = np.maximum(2.0 * self.sigma_w2
                           * np.sum(grad_k1 * grad_w0, axis=1)
                           / np.maximum(om, EPS), 1e-10)
        arg1c = 4.0 * self.sigma_w2 * k_new / (cd_kw * d * d)
        F1 = np.tanh(np.minimum(np.maximum(arg1a, arg1b), arg1c) ** 4)
        sigma_w = F1 * self.sigma_w1 + (1 - F1) * self.sigma_w2
        beta = F1 * self.beta1 + (1 - F1) * self.beta2
        gamma = F1 * self.gamma1 + (1 - F1) * self.gamma2
        nu_t = self.a1 * k_new / np.maximum(self.a1 * om, S * F2 + EPS)
        Pk = np.minimum(nu_t * S * S, 10.0 * bs * k_new * om)
        prod_w = gamma * Pk / np.maximum(nu_t, EPS)
        bval_w1 = self._boundary_value(self._omega_inlet(self._k_inlet()))
        nu_w_eff = nu + sigma_w * nu_t
        w_new, _ = solve_scalar(fv, qdot, nu_w_eff, om, prod_w, beta * om,
                                bval_w1, self.outlet_faces, source=cd_kw,
                                relax=0.5)
        w_new = np.maximum(w_new, EPS)
        self._k = k_new
        self._omega = w_new
        self._nu_t = self.a1 * k_new / np.maximum(self.a1 * w_new, S * F2 + EPS)
        denom = max(float(_safe_norm(k_new)) + float(_safe_norm(w_new)), EPS)
        resid = (float(_safe_norm(k_new - k)) + float(_safe_norm(w_new - om))) / denom
        self.iteration += 1
        self._last_residual = resid
        return resid

    @property
    def k(self):
        return self._k

    @property
    def omega(self):
        return self._omega# ---------------------------------------------------------------------------
# LES：Smagorinsky 子格子模型
# ---------------------------------------------------------------------------
class LESSmagorinskySolver(TurbulenceModelBase):
    """LES Smagorinsky 子格子：nu_sgs = (C_s Δ)^2 |S|（无输运方程）。

    |S| = sqrt(2 S_ij S_ij)；Δ 取单元体积立方根（可显式传入 filter_scale）。
    可选 Van Driest 近壁阻尼 [1-exp(-y+/25)]^2。update() 返回相邻迭代 nu_sgs
    的相对变化作为“残差”。
    """
    C_s = 0.1

    def __init__(self, fv, rho=1.0, mu=1e-3, flow_axis=0,
                 inlet_side="min", outlet_side="max",
                 turbulence_intensity=0.05, length_scale=0.1, u_ref=1.0,
                 c_s=None, filter_scale=None, van_driest=True):
        super().__init__(fv, rho, mu, flow_axis, inlet_side, outlet_side,
                         turbulence_intensity, length_scale, u_ref)
        if c_s is not None:
            self.C_s = float(c_s)
        self.van_driest = bool(van_driest)
        self._filter = self._filter_width(filter_scale)

    def _filter_width(self, filter_scale):
        fv = self.fv
        if filter_scale is not None:
            return float(filter_scale)
        return np.maximum(np.asarray(fv.volumes, float), EPS) ** (1.0 / 3.0)

    def reset(self):
        self._nu_t = np.zeros(self.fv.n_cells, float)
        self.iteration = 0
        self._last_residual = 0.0

    def update(self, u, v, w):
        S = strain_magnitude(self.fv, u, v, w)
        prev = self._nu_t
        nu_sgs = (self.C_s * self._filter) ** 2 * S
        if self.van_driest:
            yp = self.yplus()
            nu_sgs = nu_sgs * (1.0 - np.exp(-yp / 25.0)) ** 2
        self._nu_t = np.maximum(nu_sgs, 0.0)
        denom = max(float(_safe_norm(self._nu_t)), EPS)
        resid = float(_safe_norm(self._nu_t - prev)) / denom
        self.iteration += 1
        self._last_residual = resid
        return resid# ---------------------------------------------------------------------------
# 模型注册表 / 工厂
# ---------------------------------------------------------------------------
_MODELS = {
    "sa": SpalartAllmarasSolver,
    "spalart_allmaras": SpalartAllmarasSolver,
    "spalart-allmaras": SpalartAllmarasSolver,
    "k-epsilon": KEpsilonSolver,
    "k_epsilon": KEpsilonSolver,
    "ke": KEpsilonSolver,
    "k-omega-sst": KOmegaSSTSolver,
    "k_omega_sst": KOmegaSSTSolver,
    "komega_sst": KOmegaSSTSolver,
    "sst": KOmegaSSTSolver,
    "les": LESSmagorinskySolver,
    "smagorinsky": LESSmagorinskySolver,
    "les_smagorinsky": LESSmagorinskySolver,
}


def make_model(name, fv, **kwargs):
    """按字符串名实例化湍流模型（大小写/下划线/连字符不敏感）。"""
    key = str(name).strip().lower()
    if key not in _MODELS:
        names = ", ".join(sorted(set(_MODELS.keys())))
        raise ValueError("未知湍流模型 %r（支持：%s）" % (name, names))
    return _MODELS[key](fv, **kwargs)# ---------------------------------------------------------------------------
# P10 后端兼容门面：TurbulenceSolver
# ---------------------------------------------------------------------------
class TurbulenceSolver:
    """把湍流模型包装为 P10 `SolverBackend` 兼容接口。

    提供 `_initialize_field() / step() / residual() / monitor_payload()`，
    可由 `solver_run.SolverBackend` 驱动：step() 用最近一次注入的流场
    （`set_velocities`）推进湍流输运方程，返回归一化残差与流场统计。
    """

    def __init__(self, fv, model="sa", rho=1.0, mu=1e-3, flow_axis=0,
                 inlet_side="min", outlet_side="max",
                 turbulence_intensity=0.05, length_scale=0.1, u_ref=1.0,
                 **model_kwargs):
        self.fv = fv
        self.model_name = str(model).strip().lower()
        self.rho = float(rho)
        self.mu = float(mu)
        self._model = make_model(
            model, fv, rho=rho, mu=mu, flow_axis=flow_axis,
            inlet_side=inlet_side, outlet_side=outlet_side,
            turbulence_intensity=turbulence_intensity,
            length_scale=length_scale, u_ref=u_ref, **model_kwargs)
        n = fv.n_cells
        self._u = np.zeros(n, float)
        self._v = np.zeros(n, float)
        self._w = np.zeros(n, float)
        self._residual = 0.0

    @property
    def model(self):
        return self._model

    @property
    def nu_t(self):
        return self._model.nu_t

    def set_velocities(self, u, v, w):
        self._u = np.asarray(u, float)
        self._v = np.asarray(v, float)
        self._w = np.asarray(w, float)

    def _initialize_field(self):
        self._model.reset()
        return dict(model=self.model_name,
                    nu_t_max=float(self._model.nu_t.max()))

    def step(self):
        resid = float(self._model.update(self._u, self._v, self._w))
        self._residual = resid
        return dict(residual=resid,
                    u_min=float(self._u.min()), u_max=float(self._u.max()),
                    u_mean=float(self._u.mean()))

    def residual(self):
        return self._residual

    def monitor_payload(self):
        return dict(model=self.model_name,
                    nu_t_max=float(self._model.nu_t.max()),
                    iteration=self._model.iteration,
                    residual=self._residual,
                    model_residual=float(getattr(
                        self._model, "_last_residual", 0.0)))

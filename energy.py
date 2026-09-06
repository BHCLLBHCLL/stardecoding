# -*- coding: utf-8 -*-
"""P 波 P7：能量/传热 —— 对流-扩散能量方程 + 共轭传热 + 简化辐射谱系。

纯 numpy 实现（occ / scdm 两环境皆可用），建立在 P4 有限体积离散核心
（`fvm_core.FVM`）与 P5 压力基求解器（`pressure_solver.solve_linear`）之上，
为温度 T 提供能量输运求解，并可与 SIMPLE 外环 + 湍流 κ_t 耦合。

物理方程（保守形式，稳态）：
    div(rho*Cp*u*T) - div(kappa_eff*grad T) = S_T
其中 kappa_eff = kappa + kappa_t = kappa + Cp*mu_t/Pr_t（湍流导热率），
mu_t = rho*nu_t，Pr_t 为湍流普朗特数。S_T 为体积源（W/m^3），
可含简化辐射净发射、热生成等。

核心能力：

  1) 能量输运装配：`assemble_energy_transport` 组装温度标量输运方程
     （上风对流 + 中心扩散 + 源项），支持多类型**传热边界**：
     绝热 / 定温 / 热流密度 / 第三类(Robin 对流换热) / 简化辐射。
  2) 共轭传热 (CHT)：材料按单元分区（流体 / 固体），固体区速度=0 只解纯导热，
     经界面连续性耦合，`EnergySolver` 经 `set_materials` 启用。
  3) 简化辐射：Stefan-Boltzmann ε·σ·T^4 线性化的体辐射源项与壁面辐射边界，
     `RadiationModel` 提供净发射与辐射传热系数。
  4) 传热求解器：`EnergySolver` 提供 `update(u,v,w,mdot)` 导入流场求解温度，
     并暴露 `T`、`dT`（温度梯度）、`heat_flux`（壁面热流）；
     P10 后端兼容门面 `_initialize_field() / step() / residual() / monitor_payload()`。

纯 numpy 约束：所有范数走 `solver_run._safe_norm`，不依赖 scipy / np.linalg.norm。
"""
import numpy as np

from fvm_core import FVM
from pressure_solver import solve_linear
from solver_run import _safe_norm


# ---------------------------------------------------------------------------
# 能量 / 传热常数
# ---------------------------------------------------------------------------
SIGMA_STEFAN = 5.67e-8   # Stefan-Boltzmann 常数（W/(m^2 K^4)）
PR_T = 0.85              # 湍流普朗特数（k-ε / k-ω 常用值）
DEFAULT_RHO = 1.18       # 空气密度（kg/m^3）
DEFAULT_CP = 1005.0      # 空气比热（J/(kg K)）
DEFAULT_KAPPA = 0.0257   # 空气导热率（W/(m K)）
DOMINANT_TEMP = 300.0    # 线性化参考温度（未指定时的稳健初值）
EPS = 1e-12


# ---------------------------------------------------------------------------
# 传热边界类型
# ---------------------------------------------------------------------------
BND_ADIABATIC = 0     # 绝热：零梯度 (∂T/∂n = 0)
BND_FIXED_TEMP = 1    # 定温：Dirichlet，T_w = value
BND_FIXED_FLUX = 2    # 热流密度：Neumann，q_w = value (W/m^2，入为正)
BND_ROBIN = 3         # 第三类：q_w = h (T_ref - T_w)
BND_RADIATION = 4     # 简化辐射：q_w = ε σ (T_surr^4 - T_w^4)


# ---------------------------------------------------------------------------
# 面扩散 / 材料辅助
# ---------------------------------------------------------------------------
def _face_kappa(fv, kappa_cell):
    """单元有效导热率 -> 面有效导热率（内部面取均值，边界面取 owner 值）。"""
    kappa_cell = np.asarray(kappa_cell, float)
    is_int = fv.neighbor >= 0
    nbr = np.where(is_int, fv.neighbor, fv.owner)
    return 0.5 * (kappa_cell[fv.owner] + kappa_cell[nbr])


def assemble_energy_transport(fv, mdot, kappa_cell, cp_cell, T, source,
                              btype=None, bval=None, h_face=None, emis_face=None,
                              outlet_faces=None, prod=None, diss=None):
    """组装温度标量输运方程（上风对流 + 中心扩散 + 源项 + 多类型传热边界）。

    PDE（保守形式）：div(rho*cp*u*T) - div(kappa_eff*grad T) = S_T。
      - `mdot` 为面质量通量 (n_faces,)；`kappa_cell` / `cp_cell` 为逐单元有效
        导热率与比热；`source` 为逐单元体积源 (n_cells, W/m^3)。
      - 传热边界类型见 `BND_*` 常量；`bval` 语义随类型而变：
          定温 -> T_w；热流 -> q_w (W/m^2, 入为正)；Robin/辐射 -> 参考温度。
      - `h_face` / `emis_face` 为面向第 3/4 类边界的对流换热系数与发射率；缺省 0。
      - `prod` / `diss` 为逐单元线性化源项（如简化辐射净发射）：
          生产 prod 加入 rhs，衰减 diss 以对角形式隐式化。
    返回 (rows, cols, vals, rhs, ap)：未松弛系数 + 已并边界/源项的 rhs + 无松弛对角 ap。
    """
    n = fv.n_cells
    rows_l, cols_l, vals_l = [], [], []
    rhs = np.zeros(n, float)
    is_int = fv.neighbor >= 0
    o = fv.owner[is_int]
    nb = fv.neighbor[is_int]
    cp_cell = np.asarray(cp_cell, float)
    kappa_cell = np.asarray(kappa_cell, float)
    cm = 0.5 * (cp_cell[o] + cp_cell[nb]) * mdot[is_int]
    kf = _face_kappa(fv, kappa_cell)
    D = (kf[is_int] * fv.face_area[is_int]
         / np.maximum(fv._d_n[is_int], EPS))
    pos = cm >= 0.0
    rows_l.extend(o[pos]); cols_l.extend(o[pos]); vals_l.extend(cm[pos].astype(float))
    rows_l.extend(nb[pos]); cols_l.extend(o[pos]); vals_l.extend((-cm[pos]).astype(float))
    neg = ~pos
    rows_l.extend(o[neg]); cols_l.extend(nb[neg]); vals_l.extend(cm[neg].astype(float))
    rows_l.extend(nb[neg]); cols_l.extend(nb[neg]); vals_l.extend((-cm[neg]).astype(float))
    rows_l.extend(o); cols_l.extend(o); vals_l.extend(D)
    rows_l.extend(nb); cols_l.extend(nb); vals_l.extend(D)
    rows_l.extend(o); cols_l.extend(nb); vals_l.extend(-D)
    rows_l.extend(nb); cols_l.extend(o); vals_l.extend(-D)

    # 边界面：遍历分类处理（绝热 / 定温 / 热流 / Robin / 辐射）
    bnd_face = np.where(~is_int)[0]
    bo = fv.owner[~is_int]
    mb = mdot[~is_int]
    cb = cp_cell[bo] * mb
    Db = (kf[~is_int] * fv.face_area[~is_int]
          / np.maximum(fv._d_n[~is_int], EPS))
    if btype is None:
        btype_arr = np.full(fv.n_faces, BND_ADIABATIC, int)
    else:
        btype_arr = np.asarray(btype, int).astype(int)
    bval_arr = np.zeros(fv.n_faces, float)
    if bval is not None:
        bval_arr = np.asarray(bval, float)
    if h_face is None:
        h_arr = np.zeros(fv.n_faces, float)
    else:
        h_arr = np.asarray(h_face, float)
    if emis_face is None:
        em_arr = np.zeros(fv.n_faces, float)
    else:
        em_arr = np.asarray(emis_face, float)
    outlet = np.zeros(len(bo), bool)
    if outlet_faces is not None and len(outlet_faces):
        outlet = np.isin(bnd_face, outlet_faces, assume_unique=False)

    for jj in range(len(bo)):
        gf = int(bnd_face[jj])
        oi = int(bo[jj])
        m = float(mb[jj])
        c = float(cb[jj])
        d = float(Db[jj])
        bt = int(btype_arr[gf])
        v = float(bval_arr[gf])
        A = float(fv.face_area[gf])
        if outlet[jj]:
            if m > 0.0:
                vals_l.append(c); rows_l.append(oi); cols_l.append(oi)
            continue
        if bt == BND_ADIABATIC:
            vals_l.append(c); rows_l.append(oi); cols_l.append(oi)
        elif bt == BND_FIXED_TEMP:
            vals_l.append(d); rows_l.append(oi); cols_l.append(oi)
            rhs[oi] += d * v
            if m > 0.0:
                vals_l.append(c); rows_l.append(oi); cols_l.append(oi)
            else:
                rhs[oi] -= c * v
        elif bt == BND_FIXED_FLUX:
            rhs[oi] += v * A
            if m > 0.0:
                vals_l.append(c); rows_l.append(oi); cols_l.append(oi)
            else:
                rhs[oi] -= c * v
        elif bt == BND_ROBIN:
            G = (d / A) if A > EPS else 0.0
            H = float(h_arr[gf])
            Eff = (G * H / (G + H)) if (G + H) > EPS else 0.0
            FluxA = Eff * A
            vals_l.append(FluxA); rows_l.append(oi); cols_l.append(oi)
            rhs[oi] += FluxA * v
            if m > 0.0:
                vals_l.append(c); rows_l.append(oi); cols_l.append(oi)
            else:
                rhs[oi] -= c * v
        elif bt == BND_RADIATION:
            em = float(np.clip(em_arr[gf], 0.0, 1.0))
            Tw = float(T[oi]) if T is not None else v
            Tsurr = v
            h_rad = em * SIGMA_STEFAN * (Tsurr + Tw) * (Tsurr * Tsurr + Tw * Tw)
            G = (d / A) if A > EPS else 0.0
            Eff = (G * h_rad / (G + h_rad)) if (G + h_rad) > EPS else 0.0
            FluxA = Eff * A
            vals_l.append(FluxA); rows_l.append(oi); cols_l.append(oi)
            rhs[oi] += FluxA * Tsurr
            if m > 0.0:
                vals_l.append(c); rows_l.append(oi); cols_l.append(oi)
            else:
                rhs[oi] -= c * v

    rhs += np.asarray(source, float) * fv.volumes
    if prod is not None:
        rhs += np.asarray(prod, float) * fv.volumes
    rows = np.array(rows_l, np.int64)
    cols = np.array(cols_l, np.int64)
    vals = np.array(vals_l, float)
    ap = np.zeros(n, float)
    d_idx = rows == cols
    np.add.at(ap, rows[d_idx], vals[d_idx])
    if diss is not None:
        sink = np.clip(np.asarray(diss, float), 0.0, None) * fv.volumes
        ap = ap + sink
    return rows, cols, vals, rhs, ap


def solve_energy(fv, mdot, kappa_cell, cp_cell, T, source, btype=None, bval=None,
                 h_face=None, emis_face=None, outlet_faces=None, prod=None,
                 diss=None, relax=0.7, tol=1e-9, maxit=6000):
    """求解一次能量方程（欠松弛），返回 (T_new, ap)。"""
    rows, cols, vals, rhs, ap = assemble_energy_transport(
        fv, mdot, kappa_cell, cp_cell, T, source, btype, bval, h_face,
        emis_face, outlet_faces, prod, diss)
    n = fv.n_cells
    use = np.asarray(T, float)
    keep = rows != cols
    rows = rows[keep]; cols = cols[keep]; vals = vals[keep]
    aP_relaxed = ap / float(relax)
    rows = np.concatenate([rows, np.arange(n, dtype=np.int64)])
    cols = np.concatenate([cols, np.arange(n, dtype=np.int64)])
    vals = np.concatenate([vals, aP_relaxed])
    rhs = rhs + ((1.0 - relax) / relax) * ap * use
    T_new = solve_linear(rows, cols, vals, rhs, n, tol=tol, maxit=maxit)
    return T_new, ap

# ---------------------------------------------------------------------------
# 传热边界分类 / 辅助
# ---------------------------------------------------------------------------
def classify_thermal(fv, flow_axis=0, inlet_side="min", outlet_side="max"):
    """按流向坐标把边界面分为 入口 / 出口 / 壁面（返回全局面索引数组）。

    判据与 `turbulence.classify_boundaries` 一致：流向坐标靠近整体最小（或
    最大，取决于 inlet_side）的面为入口，另一端为出口，其余边界为壁面。
    """
    bnd = np.where(fv.is_boundary)[0]
    if len(bnd) == 0:
        return bnd, np.array([], np.int64), np.array([], np.int64), bnd
    coord = fv.face_centroid[bnd][:, flow_axis]
    lo = float(coord.min())
    hi = float(coord.max())
    tol = 1e-9 * max(1.0, abs(hi - lo))
    if inlet_side == "min":
        in_mask = np.abs(coord - lo) <= tol
        out_mask = np.abs(coord - hi) <= tol
    else:
        in_mask = np.abs(coord - hi) <= tol
        out_mask = np.abs(coord - lo) <= tol
    inlet = bnd[in_mask]
    outlet = bnd[out_mask]
    wall = bnd[~(in_mask | out_mask)]
    return inlet, outlet, wall, bnd


def thermal_btypes(fv, inlet_faces, outlet_faces, wall_faces, inlet_temp=None,
                   wall_btype=BND_ADIABATIC, wall_bval=0.0, wall_h=0.0,
                   wall_emis=0.0, wall_tref=DOMINANT_TEMP):
    """构造全场传热边界类型 / 值数组。

    入口：定温（默认 inlet_temp，缺省绝热）；出口：绝热零梯度；壁面：按
    `wall_btype`（绝热/定温/热流/Robin/辐射）设定。返回长度 (n_faces,) 的
    (btype, bval, h_face, emis_face) 四元组。
    """
    nf = fv.n_faces
    btype = np.full(nf, BND_ADIABATIC, int)
    bval = np.zeros(nf, float)
    h_face = np.zeros(nf, float)
    emis_face = np.zeros(nf, float)
    if inlet_temp is not None and len(inlet_faces):
        btype[inlet_faces] = BND_FIXED_TEMP
        bval[inlet_faces] = float(inlet_temp)
    if len(wall_faces):
        btype[wall_faces] = int(wall_btype)
        if wall_btype == BND_FIXED_TEMP:
            bval[wall_faces] = float(wall_bval)
        elif wall_btype == BND_FIXED_FLUX:
            bval[wall_faces] = float(wall_bval)
        elif wall_btype == BND_ROBIN:
            bval[wall_faces] = float(wall_tref)
            h_face[wall_faces] = float(wall_h)
        elif wall_btype == BND_RADIATION:
            bval[wall_faces] = float(wall_tref)
            emis_face[wall_faces] = float(np.clip(wall_emis, 0.0, 1.0))
    return btype, bval, h_face, emis_face


def volume_to_mass(fv, u, v, w, rho):
    """由速度分量重构面质量通量 mdot = rho_face * (u·n) * A（n 朝 owner 外侧）。

    `rho` 为逐单元密度；面密度取 owner/neighbor 均值（边界面取 owner 值）。
    """
    uf = fv.face_value(np.asarray(u, float))
    vf = fv.face_value(np.asarray(v, float))
    wf = fv.face_value(np.asarray(w, float))
    vn = (uf * fv.face_normal[:, 0] + vf * fv.face_normal[:, 1]
          + wf * fv.face_normal[:, 2])
    rho_f = 0.5 * (rho[fv.owner] + rho[np.where(fv.neighbor >= 0,
                                                fv.neighbor, fv.owner)])
    return rho_f * vn * fv.face_area


# ---------------------------------------------------------------------------
# 传热求解器：EnergySolver
# ---------------------------------------------------------------------------
class EnergySolver:
    """传热求解器：对流-扩散能量方程 + 材料分区（共轭传热）+ 简化辐射。

    `update(u, v, w, mdot, nu_t)` 由给定流场求解温度场 T，返回归一化残差。
    材料属性按单元可分区（流体 / 固体）：固体区速度视为 0（纯导热），从而
    支持固-液共轭传热。`nu_t` 为湍流运动粘性场，用于湍流导热率
    kappa_t = Cp * rho * nu_t / Pr_t。
    """

    def __init__(self, fv, rho=DEFAULT_RHO, cp=DEFAULT_CP, kappa=DEFAULT_KAPPA,
                 flow_axis=0, inlet_side="min", outlet_side="max",
                 inlet_temp=300.0, t_ref=300.0, pr_t=PR_T,
                 wall_btype=BND_ADIABATIC, wall_bval=0.0, wall_h=0.0,
                 wall_emis=0.0, wall_tref=None, relax=0.7, max_iter=50):
        self.fv = fv
        n = fv.n_cells
        self.rho_fluid = float(rho)
        self.cp_fluid = float(cp)
        self.kappa_fluid = float(kappa)
        self.pr_t = float(pr_t)
        self.inlet_temp = float(inlet_temp)
        self.t_ref = float(t_ref)
        self.wall_btype = int(wall_btype)
        self.wall_bval = float(wall_bval)
        self.wall_h = float(wall_h)
        self.wall_emis = float(wall_emis)
        self.wall_tref = float(wall_tref) if wall_tref is not None else float(t_ref)
        self.relax = float(relax)
        self.max_iter = int(max_iter)
        self.flow_axis = int(flow_axis)
        self.inlet_faces, self.outlet_faces, self.wall_faces, self._bnd = \
            classify_thermal(fv, flow_axis, inlet_side, outlet_side)
        self._phase = np.zeros(n, int)
        self._rho = np.full(n, float(rho), float)
        self._cp = np.full(n, float(cp), float)
        self._kappa = np.full(n, float(kappa), float)
        self._kappa_t = np.zeros(n, float)
        self._T = np.full(n, float(t_ref), float)
        self._residual = 0.0
        self.iteration = 0
        self._wall_heat_flux = np.zeros(len(self.wall_faces), float)
        self._last_Nusselt = 0.0
        self._u = np.zeros(n, float)
        self._v = np.zeros(n, float)
        self._w = np.zeros(n, float)
        self._mdot = None
        self._nu_t = None
        self._rad_vol_emis = 0.0
        self._rad_vol_tref = float(t_ref)

    # -- 场属性 ------------------------------------------------
    @property
    def T(self):
        return self._T

    @property
    def temperature(self):
        return self._T

    @property
    def phase(self):
        return self._phase

    @property
    def heat_flux(self):
        """壁面热流（W/m^2，入域为正），长度 = 壁面面数。"""
        return self._wall_heat_flux

    @property
    def nusselt(self):
        return self._last_Nusselt

    @property
    def gradient(self):
        """温度梯度 grad(T)（n_cells,3），用 Green-Gauss 重构。"""
        return self.fv.grad_gauss(self._T, boundary=self._T_boundary())

    @property
    def dT(self):
        return self.gradient

    # -- 材质 / 共轭 --------------------------------------------
    def set_materials(self, cell_phase=None, rho_s=None, cp_s=None, kappa_s=None,
                      fluid_rho=None, fluid_cp=None, fluid_kappa=None):
        """设置材料属性与单元分区（0=流体，1=固体），启用共轭传热。

        `cell_phase` 为 (n_cells,) 类别数组（非 0 视为固体）；未指定的相属性
        沿用当前值。返回 self 以便链式调用。
        """
        n = self.fv.n_cells
        if cell_phase is not None:
            ph = np.asarray(cell_phase)
            if ph.shape != (n,):
                raise ValueError("cell_phase 需长度 (%d,)，得 %s" % (n, ph.shape))
            self._phase = np.where(ph != 0, 1, 0).astype(int)
        solid = self._phase == 1
        if fluid_rho is not None:
            self._rho[~solid] = float(fluid_rho)
        if fluid_cp is not None:
            self._cp[~solid] = float(fluid_cp)
        if fluid_kappa is not None:
            self._kappa[~solid] = float(fluid_kappa)
        if rho_s is not None:
            self._rho[solid] = float(rho_s)
        if cp_s is not None:
            self._cp[solid] = float(cp_s)
        if kappa_s is not None:
            self._kappa[solid] = float(kappa_s)
        return self

    def set_radiation_volume(self, emissivity, tref):
        """设置简化体辐射源（均匀参与介质）：S_rad = ε_v σ (Tref^4 - T^4)。"""
        self._rad_vol_emis = float(np.clip(emissivity, 0.0, None))
        self._rad_vol_tref = float(tref)
        return self

    def set_velocities(self, u, v, w, mdot=None, nu_t=None):
        """注入当前流场与面质量通量，供 `step()` 推进能量方程。"""
        self._u = np.asarray(u, float).ravel()
        self._v = np.asarray(v, float).ravel()
        self._w = np.asarray(w, float).ravel()
        self._mdot = None if mdot is None else np.asarray(mdot, float)
        self._nu_t = None if nu_t is None else np.asarray(nu_t, float)

    # -- 内部辅助 ------------------------------------------------
    def _effective_kappa(self, nu_t):
        kappa = np.asarray(self._kappa, float)
        if nu_t is None or np.asarray(nu_t).size == 0:
            self._kappa_t[:] = 0.0
            return kappa
        nu_t = np.asarray(nu_t, float).ravel()
        mu_t = self._rho * nu_t
        self._kappa_t[:] = self._cp * mu_t / max(self.pr_t, EPS)
        return kappa + self._kappa_t

    def _thermal_fields(self):
        return thermal_btypes(self.fv, self.inlet_faces, self.outlet_faces,
                              self.wall_faces, inlet_temp=self.inlet_temp,
                              wall_btype=self.wall_btype,
                              wall_bval=self.wall_bval, wall_h=self.wall_h,
                              wall_emis=self.wall_emis, wall_tref=self.wall_tref)

    def _T_boundary(self):
        fv = self.fv
        b = np.zeros(fv.n_faces, float)
        isbnd = fv.is_boundary
        if isbnd.any():
            b[isbnd] = self._T[fv.owner[isbnd]]
        btype, bval, _h, _e = self._thermal_fields()
        fxt = btype == BND_FIXED_TEMP
        b[fxt] = bval[fxt]
        return b

    def _update_wall_heat_flux(self):
        fv = self.fv
        wf = np.asarray(self.wall_faces, np.int64)
        if wf.size == 0:
            self._wall_heat_flux = np.zeros(0, float)
            self._last_Nusselt = 0.0
            return
        T = self._T
        kappa_eff = self._kappa + self._kappa_t
        kf_wall = kappa_eff[fv.owner[wf]]
        dn_f = np.maximum(fv._d_n[wf], EPS)
        btype, bval, h_face, emis_face = self._thermal_fields()
        bt = btype[wf]
        v = bval[wf]
        T_cell = T[fv.owner[wf]]
        q = np.zeros(wf.size, float)
        for i, gf in enumerate(wf):
            t = int(bt[i])
            val = float(v[i])
            Tc = float(T_cell[i])
            if t == BND_ADIABATIC:
                q[i] = 0.0
            elif t == BND_FIXED_TEMP:
                q[i] = kf_wall[i] * (val - Tc) / dn_f[i]
            elif t == BND_FIXED_FLUX:
                q[i] = val
            elif t == BND_ROBIN:
                G = kf_wall[i] / dn_f[i]
                H = float(h_face[wf[i]])
                Eff = (G * H / (G + H)) if (G + H) > EPS else 0.0
                q[i] = Eff * (val - Tc)
            elif t == BND_RADIATION:
                em = float(np.clip(emis_face[wf[i]], 0.0, 1.0))
                Tsurr = val
                Tw = Tc
                h_rad = em * SIGMA_STEFAN * (Tsurr + Tw) * (Tsurr * Tsurr + Tw * Tw)
                G = kf_wall[i] / dn_f[i]
                Eff = (G * h_rad / (G + h_rad)) if (G + h_rad) > EPS else 0.0
                q[i] = Eff * (val - Tc)
        self._wall_heat_flux = q
        self._last_Nusselt = 0.0

    def update(self, u, v, w, mdot=None, nu_t=None):
        """由流场求解一次能量方程，更新温度场，返回归一化残差。"""
        fv = self.fv
        T = np.asarray(self._T, float)
        n = fv.n_cells
        if mdot is None:
            mdot = volume_to_mass(fv, u, v, w, self._rho)
        else:
            mdot = np.asarray(mdot, float)
        solid = self._phase == 1
        if solid.any():
            is_int = fv.neighbor >= 0
            o = fv.owner
            nb = np.where(is_int, fv.neighbor, o)
            face_solid = solid[o] | solid[nb]
            mdot = mdot.copy()
            mdot[face_solid] = 0.0
        kappa_eff = self._effective_kappa(nu_t)
        btype, bval, h_face, emis_face = self._thermal_fields()
        prod = None
        diss = None
        if self._rad_vol_emis > 0.0:
            Tsurr = self._rad_vol_tref
            Tw = np.maximum(T, EPS)
            h_rad = self._rad_vol_emis * SIGMA_STEFAN * (Tsurr + Tw) * (
                Tsurr * Tsurr + Tw * Tw)
            prod = h_rad * Tsurr
            diss = h_rad
        T_new, _ap = solve_energy(fv, mdot, kappa_eff, self._cp, T,
                                  np.zeros(n, float), btype=btype, bval=bval,
                                  h_face=h_face, emis_face=emis_face,
                                  outlet_faces=self.outlet_faces, prod=prod,
                                  diss=diss, relax=self.relax)
        denom = max(float(_safe_norm(T_new)), EPS)
        resid = float(_safe_norm(T_new - T)) / denom
        self._T = T_new
        self.iteration += 1
        self._residual = resid
        self._update_wall_heat_flux()
        return resid

    # -- P10 后端兼容门面 ----------------------------------------
    def _initialize_field(self):
        self._T[:] = float(self.t_ref)
        self._residual = 0.0
        self.iteration = 0
        return dict(temp_min=float(self._T.min()), temp_max=float(self._T.max()),
                    temp_mean=float(self._T.mean()))

    def step(self):
        resid = float(self.update(self._u, self._v, self._w,
                                  mdot=self._mdot, nu_t=self._nu_t))
        self._residual = resid
        return dict(residual=resid,
                    temp_min=float(self._T.min()), temp_max=float(self._T.max()),
                    temp_mean=float(self._T.mean()))

    def residual(self):
        return self._residual

    def monitor_payload(self):
        q = self._wall_heat_flux
        return dict(temp_min=float(self._T.min()), temp_max=float(self._T.max()),
                    temp_mean=float(self._T.mean()),
                    iteration=self.iteration, residual=self._residual,
                    heat_flux_max=float(q.max()) if q.size else 0.0)


# ---------------------------------------------------------------------------
# 共轭传热求解器：ConjugateSolver
# ---------------------------------------------------------------------------
class ConjugateSolver(EnergySolver):
    """共轭传热 (CHT) 求解器：在 `EnergySolver` 上叠加固-液分区与扫掠收敛。

    通过 `set_solid_material(cell_mask, rho_s, cp_s, kappa_s)` 指定固体区，
    并在 `solve()` 内对能量的对流-扩散与纯导热区域做统一装配（隐式耦合），
    界面连续性由同一稀疏矩阵自动满足（面取 kappa 均值）。
    """

    def set_solid_material(self, cell_mask, rho_s, cp_s, kappa_s):
        """指定固体区材料（布尔掩码或 (n_cells,) 类别数组）。"""
        return self.set_materials(cell_phase=cell_mask, rho_s=rho_s,
                                  cp_s=cp_s, kappa_s=kappa_s)

    def solve(self, u, v, w, mdot=None, nu_t=None, max_sweep=10, tol=1e-6):
        """对能量做多次扫掠以收敛固-液界面耦合，返回最终残差。"""
        resid = 0.0
        for _ in range(int(max_sweep)):
            resid = float(self.update(u, v, w, mdot=mdot, nu_t=nu_t))
            if resid < float(tol):
                break
        return resid


# ---------------------------------------------------------------------------
# 简化辐射模型：RadiationModel
# ---------------------------------------------------------------------------
class RadiationModel:
    """简化辐射模型：Stefan-Boltzmann 灰体辐射净交换（view 因子≈1）。

    提供净发射通量 `net_emission(T)`、线性化辐射传热系数 `h_rad(T_surf, T_surr)`
    与体辐射源 `volume_source(T)`，供 `EnergySolver` 及用户脚本调用。
    """

    def __init__(self, emissivity=0.9, tref=DOMINANT_TEMP,
                 sigma=SIGMA_STEFAN, view=1.0):
        self.epsilon = float(np.clip(emissivity, 0.0, 1.0))
        self.tref = float(tref)
        self.sigma = float(sigma)
        self.view = float(view)

    def emittance(self, T):
        """黑体/灰体发射率 ε σ T^4。"""
        return self.epsilon * self.sigma * np.asarray(T, float) ** 4

    def net_emission(self, T):
        """净发射 q = ε σ (T^4 - Tref^4)（入为正）。"""
        T = np.asarray(T, float)
        return self.epsilon * self.sigma * (T ** 4 - float(self.tref) ** 4)

    def h_rad(self, T_surface, T_surr=None):
        """线性化辐射传热系数 h = ε σ (T_s + T_a)(T_s^2 + T_a^2)。"""
        Ts = np.asarray(T_surface, float)
        Ta = float(self.tref) if T_surr is None else np.asarray(T_surr, float)
        return self.epsilon * self.sigma * (Ts + Ta) * (Ts * Ts + Ta * Ta)

    def volume_source(self, T):
        """体辐射源 S = ε σ (Tref^4 - T^4)（吸收为正体积热源）。"""
        return -self.net_emission(T)

    def linear_source(self):
        """返回线性化体源 (prod, diss)，供 `solve_energy` 的 prod/diss 使用。

        S ≈ h_rad(Tref)·Tref - h_rad(Tref)·T，即 prod = h_ref*Tref, diss = h_ref。
        """
        h = float(self.epsilon * self.sigma * (float(self.tref) + float(self.tref))
                  * (float(self.tref) ** 2 + float(self.tref) ** 2))
        return h * float(self.tref), h


# ---------------------------------------------------------------------------
# 工厂注册表
# ---------------------------------------------------------------------------
_ENERGY_MODELS = {
    "energy": EnergySolver,
    "conjugate": ConjugateSolver,
    "cht": ConjugateSolver,
    "conjugate_heat_transfer": ConjugateSolver,
}


def make_energy(fv, model="energy", **kwargs):
    """按字符串名实例化能量/传热模型（大小写/下划线不敏感）。"""
    key = str(model).strip().lower()
    if key not in _ENERGY_MODELS:
        names = ", ".join(sorted(set(_ENERGY_MODELS.keys())))
        raise ValueError("未知能量模型 %r（支持：%s）" % (model, names))
    return _ENERGY_MODELS[key](fv, **kwargs)
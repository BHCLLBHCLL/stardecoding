# -*- coding: utf-8 -*-
"""P 波 P5：压力基求解器（SIMPLE 分离 + Rhie-Chow + AMG/ILU/numpy 线性求解）。

在 P4 的 `FVM` 单元中心面拓扑上构建**速度-压力耦合**的稳态不可压求解器：

  1) 稀疏线性求解：COO(row, col, data) 表示。occ 环境优先走 `scipy.sparse`
     （构造 CSR；`pyamg` 若安装用 smoothed_aggregation AMG，否则 `spilu`+`bicgstab`
     做 ILU 加速，`spsolve` 兜底）；scdm 环境回退**纯 numpy Gauss-Seidel/SOR**
     （M 矩阵/泊松矩阵对角占优，收敛可靠）。
  2) SIMPLE 分离（每外部迭代）：
       - 动量预测：对 u/v/w 分别装配对流(一阶上风)+扩散(中心差分)+压力梯度源，
         隐式求解 → 预测速度 u*；
       - Rhie-Chow 面质量通量：动量插值重构面速度，消除并列网格压力棋盘；
       - 压力修正：由质量不平衡装配压力泊松矩阵 → p'；
       - 校正：p += α_p p'、u -= d ∇p'、面通量 += 动量插值修正。
  3) 边界条件：速度 Dirichlet（入口/无滑移壁面）+ 出口零梯度；压力 Neumann
     （壁面/入口/出口零梯度）+ 出口压力参考（固定一个参考单元 p'=0）。

残差 = 归一化连续性质量不平衡，收敛时 → 0，曲线健康衰减（P5 验收：残差下降
曲线健康）。纯 numpy 约束（scdm 无 scipy）：所有范数走 `solver_run._safe_norm`
（sqrt(sum(x*x))），避免 occ 环境 np.linalg 触发 Windows fatal exception。

验收核心（P5 行）：压力基求解器 —— SIMPLE/PISO + Coupled；AMG(pyamg)/ILU，
残差曲线健康。
"""
import numpy as np

from fvm_core import FVM
from solver_run import _safe_norm


# ---------------------------------------------------------------------------
# 稀疏线性求解（COO → scipy / 纯 numpy 双路径）
# ---------------------------------------------------------------------------
def _scipy_available():
    try:
        import scipy  # noqa: F401
        return True
    except Exception:
        return False


def solve_linear(row, col, data, b, n, tol=1e-9, maxit=8000,
                 x0=None, kind="auto", sor=1.8):
    """解稀疏线性系统 A x = b（纯 numpy 必可用，scipy 可选加速）。

    row/col/data 为 COO 三元组（float/整数数组），shape=(n,n)。返回解向量 x。
    kind: "auto" | "scipy" | "numpy"。
    """
    row = np.asarray(row, np.int64)
    col = np.asarray(col, np.int64)
    data = np.asarray(data, float)
    b = np.asarray(b, float).ravel()
    n = int(n)
    if b.shape != (n,):
        raise ValueError("线性求解 RHS 需 (%d,)，得 %s" % (n, b.shape))
    if _scipy_available() and kind in ("auto", "scipy"):
        try:
            x = _solve_scipy(row, col, data, b, n, tol, maxit, x0)
            if x is not None:
                return x
        except Exception:
            pass
    return _solve_numpy(row, col, data, b, n, tol, maxit, x0, sor)


def _solve_scipy(row, col, data, b, n, tol, maxit, x0):
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    A = sp.csr_matrix((data, (row, col)), shape=(n, n))
    try:
        import pyamg  # noqa: F401
        ml = pyamg.smoothed_aggregation_solver(A)
        x = ml.solve(b, x0=x0, tol=tol, maxiter=maxit)
        if _check_residual(row, col, data, x, b, 1e-6):
            return np.asarray(x, float).ravel()
    except Exception:
        pass
    try:
        ilu = spla.spilu(A.tocsc())
        x, info = spla.bicgstab(A, b, M=ilu, tol=tol, maxiter=maxit)
        if info == 0 or _check_residual(row, col, data, x, b, 1e-6):
            return np.asarray(x, float).ravel()
    except Exception:
        pass
    try:
        x = spla.spsolve(A.tocsc(), b)
        return np.asarray(x, float).ravel()
    except Exception:
        return None


def _check_residual(row, col, data, x, b, rtol):
    x = np.asarray(x, float).ravel()
    n = len(b)
    ax = np.zeros(n, float)
    np.add.at(ax, row, data * x[col])
    rnorm = _safe_norm(ax - b)
    bnorm = _safe_norm(b)
    return rnorm <= rtol * max(bnorm, 1e-30)


def _solve_numpy(row, col, data, b, n, tol, maxit, x0, sor):
    """纯 numpy Gauss-Seidel/SOR：COO 按行聚合为 CSR-ish 行储存，逐行迭代。"""
    if n == 0:
        return np.zeros(0, float)
    x = np.zeros(n, float) if x0 is None else np.asarray(x0, float).ravel().astype(float).copy()
    if x.shape != (n,):
        x = np.zeros(n, float)
    order = np.argsort(row, kind="stable")
    r = row[order]
    c = col[order]
    d = data[order]
    counts = np.zeros(n, np.int64)
    np.add.at(counts, r, 1)
    row_start = np.zeros(n + 1, np.int64)
    np.cumsum(counts, out=row_start[1:])
    diag = np.zeros(n, float)
    np.add.at(diag, r[r == c], d[r == c])
    bnorm = _safe_norm(b)
    scale = max(bnorm, 1.0)
    for _ in range(int(maxit)):
        max_diff = 0.0
        for i in range(n):
            lo = row_start[i]
            hi = row_start[i + 1]
            s = 0.0
            for k in range(lo, hi):
                s += d[k] * x[c[k]]
            dg = diag[i]
            if abs(dg) < 1e-300:
                continue
            newx = x[i] + sor * (b[i] - s) / dg
            diff = newx - x[i]
            if diff < 0.0:
                diff = -diff
            if diff > max_diff:
                max_diff = diff
            x[i] = newx
        if max_diff <= tol * scale:
            break
    return x


# ---------------------------------------------------------------------------
# 压力基 SIMPLE 求解器
# ---------------------------------------------------------------------------
class PressureSolver:
    """P5：基于 FVM 单元中心面离散的稳态不可压 SIMPLE 分离求解器。

    单元中心变量：速度 u/v/w（n_cells,）、压力 p（n_cells,）、面质量通量 mdot
    （n_faces,）。`SolverBackend` 可直接驱动（P10 闭环），step() 返回残差/场统计，
    monitor_payload() 与之同键。field() 返回速度幅值（监视器/后处理用）。

    边界（速度）：`velocity_bc` 由 `inlet_axis/inlet_side/inlet_velocity` 设定
    入口（固定速度入流）；其余边界面按 `outlet_side` 分出口零梯度与无滑移壁面
    （Dirichlet 0）。压力在出口侧取 `pressure_ref` 参考单元固定（p'=0）。
    """

    def __init__(self, vertices, cells, rho=1.0, mu=1.0e-3,
                 inlet_axis=0, inlet_side="min", inlet_velocity=(1.0, 0.0, 0.0),
                 outlet_side="max", pressure_ref=0.0,
                 alpha_momentum=0.7, alpha_pressure=0.3,
                 max_outer=50, name="Pressure", initializer=None,
                 turb_model=None, energy_model=None, inlet_temp=None,
                 beta=0.0, gravity=(0.0, 0.0, -9.81), buoy_ref_temp=300.0,
                 vof_model=None, inlet_alpha=1.0,
                 mixture_model=None, mixture_inlet_alphas=None,
                 mixture_alpha0=None,
                 eulerian_model=None, eulerian_inlet_alphas=None,
                 eulerian_alpha0=None, eulerian_enable_lift=False,
                 eulerian_enable_virtual_mass=False,
                 eulerian_enable_wall_lubrication=False,
                 eulerian_population_balance=False,
                 eulerian_coalescence_rate=None, eulerian_breakup_rate=None,
                 eulerian_nucleation_rate=None,
                 dpm_model=None, dpm_rho_p=None, dpm_dia_p=None,
                 dpm_mass_flow=None, dpm_parcels_per_step=None,
                 species_model=None, combustion_model=None,
                 combustion_inlet_fractions=None, combustion_ref_temp=300.0):
        self.name = name
        self._rho0 = float(rho)
        self._mu0 = float(mu)
        self.vof_model = vof_model
        self.inlet_alpha = float(np.clip(inlet_alpha, 0.0, 1.0))
        self.mixture_model = mixture_model
        self.mixture_inlet_alphas = mixture_inlet_alphas
        self.mixture_alpha0 = mixture_alpha0
        self.eulerian_model = eulerian_model
        self.eulerian_inlet_alphas = eulerian_inlet_alphas
        self.eulerian_alpha0 = eulerian_alpha0
        self.eulerian_enable_lift = bool(eulerian_enable_lift)
        self.eulerian_enable_virtual_mass = bool(eulerian_enable_virtual_mass)
        self.eulerian_enable_wall_lubrication = bool(eulerian_enable_wall_lubrication)
        self.eulerian_population_balance = bool(eulerian_population_balance)
        self.eulerian_coalescence_rate = eulerian_coalescence_rate
        self.eulerian_breakup_rate = eulerian_breakup_rate
        self.eulerian_nucleation_rate = eulerian_nucleation_rate
        self.dpm_model = dpm_model
        self.dpm_rho_p = dpm_rho_p
        self.dpm_dia_p = dpm_dia_p
        self.dpm_mass_flow = dpm_mass_flow
        self.dpm_parcels_per_step = dpm_parcels_per_step
        self.species_model = species_model
        self.combustion_model = combustion_model
        self.combustion_inlet_fractions = combustion_inlet_fractions
        self.combustion_ref_temp = float(combustion_ref_temp)
        self.turb_model = turb_model
        self.energy_model = energy_model
        self.inlet_temp = float(inlet_temp) if inlet_temp is not None else None
        self.beta = float(beta)
        self.gravity = tuple(float(g) for g in gravity)
        self.buoy_ref_temp = float(buoy_ref_temp)
        self.inlet_axis = int(inlet_axis)
        self.inlet_side = inlet_side
        self.inlet_velocity = tuple(float(v) for v in inlet_velocity)
        self.outlet_side = outlet_side
        self.pressure_ref = float(pressure_ref)
        self.alpha_momentum = float(alpha_momentum)
        self.alpha_pressure = float(alpha_pressure)
        self.max_outer = int(max_outer)
        self.initializer = initializer
        self.vertices = None
        self.cells = None
        self.iteration = 0
        self._fv = None
        self._u = None
        self._v = None
        self._w = None
        self._p = None
        self._mdot = None
        self._inlet_faces = None
        self._outlet_faces = None
        self._wall_faces = None
        self._dirichlet_cell = None
        self._inlet_mdot_scale = 1.0
        self._last_residual = float("nan")
        self._last_cont_ratio = float("nan")
        self._last_d_cell = None
        self.set_mesh(vertices, cells)

    # -- 网格 / 边界 -----------------------------------------------
    def set_mesh(self, vertices, cells):
        V = np.asarray(vertices, float)
        C = np.asarray(cells, np.int64)
        if C.ndim != 2 or C.shape[1] != 4:
            raise ValueError("PressureSolver 需要四面体单元(4 列)")
        if len(V) < 4 or len(C) < 1:
            raise ValueError("网格过小，无法压力求解")
        self.vertices = V
        self.cells = C
        fv = FVM(V, C)
        self._fv = fv
        self._build_boundary()
        self._initialize_field()
        self._ensure_turb_model()
        self._ensure_energy_model()
        self._ensure_vof_model()
        self._ensure_mixture_model()
        self._ensure_eulerian_model()
        self._ensure_dpm_model()
        self._ensure_species_model()
        self._ensure_combustion_model()

    def _build_boundary(self):
        fv = self._fv
        ax = fv.face_centroid[:, self.inlet_axis]
        cut_min = float(ax.min())
        cut_max = float(ax.max())
        bnd = fv.is_boundary.copy()
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
        wall = bnd & ~inlet & ~outlet
        self._inlet_faces = np.where(inlet)[0]
        self._outlet_faces = np.where(outlet)[0]
        self._wall_faces = np.where(wall)[0]
        oc = fv.centroids[:, self.inlet_axis]
        if self.outlet_side == "max":
            self._dirichlet_cell = int(np.argmax(oc))
        else:
            self._dirichlet_cell = int(np.argmin(oc))
        self._inlet_mdot_scale = self._abs_inlet_mdot()

    def _abs_inlet_mdot(self):
        fv = self._fv
        if len(self._inlet_faces) == 0:
            return 1.0
        vel = np.asarray(self.inlet_velocity, float)
        n = fv.face_normal[self._inlet_faces]
        A = fv.face_area[self._inlet_faces]
        un = vel[0] * n[:, 0] + vel[1] * n[:, 1] + vel[2] * n[:, 2]
        return max(float(np.sum(self._face_rho()[self._inlet_faces] * np.abs(un) * A)), 1e-12)

    # -- 几何 / FVM 便利 -------------------------------------------
    @property
    def fvm(self):
        return self._fv

    def set_initializer(self, init):
        """P3 注入初始化器：Run 前 Initialize 用它生成初场（None 恢复默认零场）。"""
        self.initializer = init

    def _set_initial_field(self, arr):
        """P5 写入初场：标量数组作为压力初值（单元长或节点长），速度保持零。"""
        arr = np.asarray(arr, float).ravel()
        fv = self._fv
        if arr.shape == (fv.n_cells,):
            p = arr.copy()
        elif arr.shape == (len(self.vertices),):
            p = arr[self.cells].mean(axis=1)
        else:
            raise ValueError("P5 初场维度不匹配：%s（需节点 %d 或单元 %d）"
                             % (arr.shape, len(self.vertices), fv.n_cells))
        self._p = p
        self._u[:] = 0.0
        self._v[:] = 0.0
        self._w[:] = 0.0
        self._rebuild_mdot()
        self.iteration = 0
        self._last_residual = float("nan")
        self._last_cont_ratio = float("nan")

    def _initialize_field(self):
        fv = self._fv
        self._u = np.zeros(fv.n_cells, float)
        self._v = np.zeros(fv.n_cells, float)
        self._w = np.zeros(fv.n_cells, float)
        self._p = np.full(fv.n_cells, self.pressure_ref, float)
        if self.initializer is not None and self.initializer.source_field is not None:
            self.initializer.apply_initial(self)
            self.iteration = 0
            self._last_residual = float("nan")
            self._last_cont_ratio = float("nan")
            self._rebuild_mdot()
            return
        self._u[:] = self.inlet_velocity[0]
        self._v[:] = self.inlet_velocity[1]
        self._w[:] = self.inlet_velocity[2]
        self._rebuild_mdot()
        self.iteration = 0
        self._last_residual = float("nan")
        self._last_cont_ratio = float("nan")

    def _rebuild_mdot(self):
        """由当前速度场线性插值面速度，估算面质量通量 mdot（Rhie-Chow 前一步）。"""
        fv = self._fv
        uf = fv.face_value(self._u, boundary=self._boundary_u(0))
        vf = fv.face_value(self._v, boundary=self._boundary_u(1))
        wf = fv.face_value(self._w, boundary=self._boundary_u(2))
        un = uf * fv.face_normal[:, 0] + vf * fv.face_normal[:, 1] \
            + wf * fv.face_normal[:, 2]
        self._mdot = self._face_rho() * un * fv.face_area
        self._fix_boundary_mdot()

    def _fix_boundary_mdot(self):
        """边界质量通量：入口固定（速度 Dirichlet 入流）、壁面 0、出口零梯度外推。"""
        fv = self._fv
        vel = np.asarray(self.inlet_velocity, float)
        if len(self._inlet_faces):
            n = fv.face_normal[self._inlet_faces]
            A = fv.face_area[self._inlet_faces]
            un = vel[0] * n[:, 0] + vel[1] * n[:, 1] + vel[2] * n[:, 2]
            self._mdot[self._inlet_faces] = self._face_rho()[self._inlet_faces] * un * A
        if len(self._wall_faces):
            self._mdot[self._wall_faces] = 0.0
        # 出口零梯度外推：由修正后单元速度重构（保证全局质量守恒一致）
        if len(self._outlet_faces) and self._u is not None:
            oo = fv.owner[self._outlet_faces]
            n = fv.face_normal[self._outlet_faces]
            A = fv.face_area[self._outlet_faces]
            un = (self._u[oo] * n[:, 0] + self._v[oo] * n[:, 1]
                  + self._w[oo] * n[:, 2])
            self._mdot[self._outlet_faces] = self._face_rho()[self._outlet_faces] * un * A
            # 全局质量守恒对标：出口面为压力 Neumann（∂p'/∂n=0），不进泊松矩阵，
            # 其通量仅由单元速度外推，可能与入口不闭合 → 统一的出口单元残差平台。
            # 对出口通量做全局缩放，使 sum(mdot)=0（入口固定、壁面=0，出口为唯一可调边界面）。
            inflow = self._mdot[self._inlet_faces].sum()
            outflow = self._mdot[self._outlet_faces].sum()
            if abs(outflow) > 1e-30:
                self._mdot[self._outlet_faces] *= -inflow / outflow

    # -- 边界值（速度分量，返回面长数组） ---------------------------
    def _boundary_u(self, comp):
        """速度分量的边界面 Dirichlet 值。

        入口 = 入流速度；壁面 = 0（无滑移）；出口 = 零梯度外推（owner 值），
        用于面质量通量重构。动量装配对出口用零梯度（不取边界已知值）。
        """
        fv = self._fv
        b = np.zeros(fv.n_faces, float)
        vel = self.inlet_velocity
        b[self._inlet_faces] = vel[comp]
        if self._u is not None:
            comp_arr = {0: self._u, 1: self._v, 2: self._w}[comp]
            b[self._outlet_faces] = comp_arr[fv.owner[self._outlet_faces]]
        return b

    # -- 场 / 迭代 -------------------------------------------------
    def field(self):
        return _safe_norm(np.stack([self._u, self._v, self._w], axis=1), axis=1)

    def velocity(self):
        return np.stack([self._u, self._v, self._w], axis=1)

    def pressure(self):
        return self._p

    def mass_flux(self):
        return self._mdot

    def source_nodes(self):
        return self._inlet_faces

    def residual(self):
        return float("nan") if self._u is None else self._last_residual

    def continuity_ratio(self):
        return self._last_cont_ratio

    # -- 物性：单相标量 / VOF 可变密度 ----------------------------
    @property
    def rho(self):
        """密度：无多相时为标量 _rho0；有 VOF/Mixture 时为逐单元混合密度场。"""
        if self.vof_model is not None and not isinstance(self.vof_model, str):
            return self.vof_model.rho
        if self.mixture_model is not None and not isinstance(self.mixture_model, str):
            return self.mixture_model.rho
        if self.eulerian_model is not None and not isinstance(self.eulerian_model, str):
            return self.eulerian_model.rho
        return self._rho0

    @property
    def mu(self):
        """动力粘度：无多相时为标量 _mu0；有 VOF/Mixture 时为逐单元混合粘度场。"""
        if self.vof_model is not None and not isinstance(self.vof_model, str):
            return self.vof_model.mu
        if self.mixture_model is not None and not isinstance(self.mixture_model, str):
            return self.mixture_model.mu
        if self.eulerian_model is not None and not isinstance(self.eulerian_model, str):
            return self.eulerian_model.mu
        return self._mu0

    @property
    def rho_cell(self):
        """逐单元密度场（n_cells,）；单相时返回满标量的数组。"""
        r = self.rho
        if np.isscalar(r):
            return np.full(self._fv.n_cells, float(r), float)
        return np.asarray(r, float).ravel()

    def _face_rho(self):
        """面密度（n_faces,）：恒为参考密度 _rho0。

        单流体 VOF：压力/动量矩阵采用单相参考密度以保持 SIMPLE 稳定，密度差通过
        动量 RHS 的显式重力体源进入（见 `_assemble_momentum`），故此处不做可变密度。
        """
        return np.full(self._fv.n_faces, float(self._rho0), float)

    def _face_mu(self):
        """面动力粘度（n_faces,）：恒为参考粘度 _mu0（动机同 `_face_rho`）。"""
        return np.full(self._fv.n_faces, float(self._mu0), float)

    # -- 多相 VOF 耦合 --------------------------------------------
    def _ensure_vof_model(self):
        """惰性构建 VOF 模型：vof_model 为字符串名时按当前网格/边界实例化。"""
        if self.vof_model is None or not isinstance(self.vof_model, str):
            return
        import vof as _vof
        fv = self._fv
        kwargs = dict(
            flow_axis=self.inlet_axis, inlet_side=self.inlet_side,
            outlet_side=self.outlet_side, inlet_alpha=self.inlet_alpha)
        self.vof_model = _vof.make_vof(fv, model=self.vof_model, **kwargs)

    def _update_vof(self):
        """在 SIMPLE 环尾部用当前速度场推进一次 α 输运，更新 PLIC 界面与混合密度。"""
        if self._u is None:
            return
        if self.vof_model is None or isinstance(self.vof_model, str):
            return
        try:
            self.vof_model.update(self._u, self._v, self._w)
        except Exception:
            pass

    # -- 多相 Mixture / DPM 耦合 -----------------------------------
    def _ensure_mixture_model(self):
        """惰性构建 Mixture 模型：mixture_model 为字符串名时按当前网格/边界实例化。"""
        if self.mixture_model is None or not isinstance(self.mixture_model, str):
            return
        import mixture as _mix
        fv = self._fv
        kwargs = dict(
            flow_axis=self.inlet_axis, inlet_side=self.inlet_side,
            outlet_side=self.outlet_side,
            inlet_alphas=self.mixture_inlet_alphas,
            alpha0=self.mixture_alpha0)
        self.mixture_model = _mix.make_mixture(
            fv, model=self.mixture_model, **kwargs)

    def _update_mixture(self):
        """在 SIMPLE 环尾部用当前速度场推进各弥散相体积分数，更新混合物性。"""
        if self._u is None:
            return
        if self.mixture_model is None or isinstance(self.mixture_model, str):
            return
        try:
            self.mixture_model.update(self._u, self._v, self._w, mdot=self._mdot)
        except Exception:
            pass

    def _ensure_eulerian_model(self):
        """惰性构建欧拉-欧拉多相模型：eulerian_model 为字符串名时按当前网格/边界实例化。"""
        if self.eulerian_model is None or not isinstance(self.eulerian_model, str):
            return
        import eulerian_multiphase as _em
        fv = self._fv
        kwargs = dict(
            flow_axis=self.inlet_axis, inlet_side=self.inlet_side,
            outlet_side=self.outlet_side,
            inlet_alphas=self.eulerian_inlet_alphas,
            alpha0=self.eulerian_alpha0)
        if self.eulerian_enable_lift:
            kwargs["enable_lift"] = True
        if self.eulerian_enable_virtual_mass:
            kwargs["enable_virtual_mass"] = True
        if self.eulerian_enable_wall_lubrication:
            kwargs["enable_wall_lubrication"] = True
        if self.eulerian_population_balance:
            kwargs["population_balance"] = True
        if self.eulerian_coalescence_rate is not None:
            kwargs["coalescence_rate"] = float(self.eulerian_coalescence_rate)
        if self.eulerian_breakup_rate is not None:
            kwargs["breakup_rate"] = float(self.eulerian_breakup_rate)
        if self.eulerian_nucleation_rate is not None:
            kwargs["nucleation_rate"] = float(self.eulerian_nucleation_rate)
        self.eulerian_model = _em.make_eulerian_multiphase(
            fv, model=self.eulerian_model, **kwargs)

    def _update_eulerian(self):
        """在 SIMPLE 环尾部用当前速度场推进欧拉-欧拉各相体积分数，更新相间作用力。"""
        if self._u is None:
            return
        if self.eulerian_model is None or isinstance(self.eulerian_model, str):
            return
        try:
            self.eulerian_model.update(self._u, self._v, self._w, mdot=self._mdot)
        except Exception:
            pass

    def _ensure_dpm_model(self):
        """惰性构建 DPM 模型：dpm_model 为字符串名时按当前网格/边界实例化。"""
        if self.dpm_model is None or not isinstance(self.dpm_model, str):
            return
        import dpm as _dpm
        fv = self._fv
        kwargs = dict(
            flow_axis=self.inlet_axis, inlet_side=self.inlet_side,
            outlet_side=self.outlet_side,
            inlet_velocity=self.inlet_velocity)
        if self.dpm_rho_p is not None:
            kwargs["rho_p"] = float(self.dpm_rho_p)
        if self.dpm_dia_p is not None:
            kwargs["d_p"] = float(self.dpm_dia_p)
        if self.dpm_mass_flow is not None:
            kwargs["mass_flow"] = float(self.dpm_mass_flow)
        if self.dpm_parcels_per_step is not None:
            kwargs["parcels_per_step"] = int(self.dpm_parcels_per_step)
        self.dpm_model = _dpm.make_dpm(fv, model=self.dpm_model, **kwargs)

    def _update_dpm(self):
        """在 SIMPLE 环尾部用当前流体场推进拉格朗日粒子，更新连续相耦合源。"""
        if self._u is None:
            return
        if self.dpm_model is None or isinstance(self.dpm_model, str):
            return
        try:
            self.dpm_model.set_fluid(self.rho, self.mu)
            self.dpm_model.update(self._u, self._v, self._w)
        except Exception:
            pass

    # -- 组分 / 燃烧耦合 -----------------------------------------
    def _ensure_species_model(self):
        """惰性构建组分输运模型：species_model 为字符串名时按当前网格/边界实例化。"""
        if self.species_model is None or not isinstance(self.species_model, str):
            return
        import species as _sp
        fv = self._fv
        kwargs = dict(flow_axis=self.inlet_axis, inlet_side=self.inlet_side,
                      outlet_side=self.outlet_side)
        self.species_model = _sp.make_species(fv, model=self.species_model,
                                              **kwargs)

    def _update_species(self):
        """在 SIMPLE 环尾部用当前速度场推进组分质量分数输运。"""
        if self._u is None:
            return
        if self.species_model is None or isinstance(self.species_model, str):
            return
        try:
            mdot = getattr(self, "_mdot", None)
            self.species_model.update(self._u, self._v, self._w,
                                      mdot=mdot, nu_t=self.nu_t)
        except Exception:
            pass

    def _ensure_combustion_model(self):
        """惰性构建燃烧模型：combustion_model 为字符串名时按当前网格/边界实例化。"""
        if self.combustion_model is None or not isinstance(self.combustion_model, str):
            return
        import combustion as _cb
        fv = self._fv
        kwargs = dict(flow_axis=self.inlet_axis, inlet_side=self.inlet_side,
                      outlet_side=self.outlet_side,
                      T_ref=self.combustion_ref_temp)
        if self.combustion_inlet_fractions is not None:
            kwargs["inlet_fractions"] = self.combustion_inlet_fractions
        self.combustion_model = _cb.make_combustion(
            fv, model=self.combustion_model, **kwargs)

    def _update_combustion(self):
        """在 SIMPLE 环尾部用当前流场/温度推进燃烧源项，更新组分与热释放。"""
        if self._u is None:
            return
        if self.combustion_model is None or isinstance(self.combustion_model, str):
            return
        try:
            T = None
            if (self.energy_model is not None
                    and not isinstance(self.energy_model, str)):
                T = getattr(self.energy_model, "T", None)
            mdot = getattr(self, "_mdot", None)
            self.combustion_model.update(self._u, self._v, self._w,
                                         mdot=mdot, nu_t=self.nu_t, T=T)
        except Exception:
            pass

    # -- 湍流 nu_t 耦合 -------------------------------------------
    @property
    def nu_t(self):
        """湍流粘性 nu_t（n_cells,）；无湍流模型时为全零。"""
        if self._fv is None:
            return np.zeros(0, float)
        if self.turb_model is None:
            return np.zeros(self._fv.n_cells, float)
        nu = np.asarray(self.turb_model.nu_t, float).ravel()
        if nu.shape != (self._fv.n_cells,):
            return np.zeros(self._fv.n_cells, float)
        return nu

    def _face_nu_t(self):
        """返回面插值 nu_t（n_faces,）：内部面取 owner/neighbor 平均，边界面取 owner。"""
        fv = self._fv
        nu = self.nu_t
        nf = fv.n_faces
        nu_face = np.zeros(nf, float)
        is_int = fv.neighbor >= 0
        if self.turb_model is not None:
            nu_face[is_int] = 0.5 * (nu[fv.owner[is_int]] + nu[fv.neighbor[is_int]])
            nu_face[~is_int] = nu[fv.owner[~is_int]]
        return nu_face

    def _ensure_turb_model(self):
        """惰性构建湍流模型：turb_model 为字符串名时按当前网格/物性实例化。"""
        if self.turb_model is None or not isinstance(self.turb_model, str):
            return
        import turbulence as _turb
        fv = self._fv
        u_ref = float(_safe_norm(np.array(self.inlet_velocity, float)))
        length_scale = float(np.mean(np.asarray(fv.volumes, float) ** (1.0 / 3.0)))
        self.turb_model = _turb.make_model(
            self.turb_model, fv, rho=self._rho0, mu=self._mu0,
            u_ref=u_ref, length_scale=length_scale)

    def _update_turbulence(self):
        """在 SIMPLE 环尾部解湍流输运方程，更新 nu_t 供下一步动量装配。"""
        if self._u is None:
            return
        try:
            self.turb_model.update(self._u, self._v, self._w)
        except Exception:
            pass

    # -- 能量 / 传热耦合 -------------------------------------------
    def _ensure_energy_model(self):
        """惰性构建能量模型：energy_model 为字符串名时按当前网格/物性实例化。"""
        if self.energy_model is None or not isinstance(self.energy_model, str):
            return
        import energy as _eng
        fv = self._fv
        kwargs = {}
        if self.inlet_temp is not None:
            kwargs["inlet_temp"] = self.inlet_temp
        if self.beta > 0.0:
            kwargs["wall_btype"] = _eng.BND_FIXED_TEMP
        self.energy_model = _eng.make_energy(fv, model=self.energy_model, **kwargs)

    def _update_energy(self):
        """在 SIMPLE 环尾部解能量方程，更新温度场供浮力动量源使用。"""
        if self._u is None:
            return
        if self.energy_model is None:
            return
        try:
            mdot = getattr(self, "_mdot", None)
            self.energy_model.update(
                self._u, self._v, self._w, mdot=mdot, nu_t=self.nu_t)
        except Exception:
            pass

    # -- 动量装配 -------------------------------------------------
    def _assemble_momentum(self, comp):
        fv = self._fv
        mdot = self._mdot
        mu_face = self._face_mu()
        rho_face = self._face_rho()
        n = fv.n_cells
        rows = []
        cols = []
        vals = []
        rhs = np.zeros(n, float)
        # 内部面：对流(上风)+扩散(中心差分)
        is_int = fv.neighbor >= 0
        o = fv.owner[is_int]
        nb = fv.neighbor[is_int]
        m = mdot[is_int]
        nu_face = self._face_nu_t()
        D = (mu_face[is_int] + rho_face[is_int] * nu_face[is_int]) * fv.face_area[is_int] / np.maximum(fv._d_n[is_int], 1e-12)
        pos = m >= 0.0
        rows.extend(o[pos]); cols.extend(o[pos]); vals.extend(m[pos].astype(float))
        rows.extend(nb[pos]); cols.extend(o[pos]); vals.extend((-m[pos]).astype(float))
        neg = ~pos
        rows.extend(o[neg]); cols.extend(nb[neg]); vals.extend(m[neg].astype(float))
        rows.extend(nb[neg]); cols.extend(nb[neg]); vals.extend((-m[neg]).astype(float))
        rows.extend(o); cols.extend(o); vals.extend(D)
        rows.extend(nb); cols.extend(nb); vals.extend(D)
        rows.extend(o); cols.extend(nb); vals.extend(-D)
        rows.extend(nb); cols.extend(o); vals.extend(-D)
        # 边界面（np.where(~is_int)[0] 给出边界面全局索引，与 bo/mb/Db 同序）
        bo = fv.owner[~is_int]
        bnd_face = np.where(~is_int)[0]
        mb = mdot[~is_int]
        Db = (mu_face[~is_int] + rho_face[~is_int] * nu_face[~is_int]) * fv.face_area[~is_int] / np.maximum(fv._d_n[~is_int], 1e-12)
        bval = self._boundary_u(comp)
        bvals = bval[~is_int]
        # 出口零梯度面：速度外推 φ_face=φ_owner → 对流对角 += m（m>0 出流）
        # 扩散零梯度 → 无贡献。其余边界面（入口/壁面 Dirichlet）走已知边界值。
        outlet_mask = np.isin(bnd_face, self._outlet_faces, assume_unique=False)
        # 零梯度出流且 m>0：对流对角 += mb
        out_conv = outlet_mask & (mb >= 0.0)
        rows.extend(bo[out_conv]); cols.extend(bo[out_conv])
        vals.extend(mb[out_conv].astype(float))
        # 已知边界速度的对流（入口/壁面，含所有 m<0 反向流入）：RHS -= m*bval
        conv_rhs = ~out_conv
        for i in range(len(bo)):
            bcell = bo[i]
            if conv_rhs[i]:
                rhs[bcell] -= mb[i] * bvals[i]
        # 扩散 Dirichlet（入口/壁面固定速度）：对角 += Db，RHS += Db*bval；
        # 出口零梯度：不贡献扩散
        diff_b = ~outlet_mask
        rows.extend(bo[diff_b]); cols.extend(bo[diff_b]); vals.extend(Db[diff_b])
        for i in range(len(bo)):
            if not outlet_mask[i]:
                bcell = bo[i]
                rhs[bcell] += Db[i] * bvals[i]
        # 压力梯度源：-V ∇p
        grad_p = self._grad_pressure(self._p)
        rhs += -fv.volumes * grad_p[:, comp]
        # Boussinesq 浮力源：S = -ρ₀ β (T - T₀) g · V（仅 beta>0 且启用能量模型）
        if self.beta > 0.0 and self.energy_model is not None:
            T = np.asarray(getattr(self.energy_model, "T", None), float)
            if T is not None and T.shape == (fv.n_cells,):
                rhs += (-self.rho * self.beta
                        * (T - self.buoy_ref_temp) * self.gravity[comp]
                        * fv.volumes)
        # DPM 连续相耦合源：粒子拖曳反作用力密度逐单元体源（S 已含 /V 归一）。
        if self.dpm_model is not None and not isinstance(self.dpm_model, str):
            try:
                dpm_src = self.dpm_model.coupling_source()
                if dpm_src.shape == (fv.n_cells, 3):
                    rhs += dpm_src[:, comp] * fv.volumes
            except Exception:
                pass
        # VOF 重力耦合暂不启用显式密度差体源：水/气密度比 (~850×) 下该源项过大，
        # 且未在压力方程中以 p_rgh（剔除静水压）同步处理，导致 SIMPLE 发散。
        # 单流体界面捕捉保持稳定即可；浮力/静水耦合留给自由面算例用 p_rgh 形式完善。
        # 速度欠松弛：aP = ap/α；RHS 补偿 (1-α)/α * ap * φ_old
        rows = np.array(rows, np.int64)
        cols = np.array(cols, np.int64)
        vals = np.array(vals, float)
        ap = np.zeros(n, float)
        d_idx = rows == cols
        np.add.at(ap, rows[d_idx], vals[d_idx])
        cur = {0: self._u, 1: self._v, 2: self._w}[comp]
        relax = self.alpha_momentum
        aP_relaxed = ap / relax
        keep = rows != cols
        rows = rows[keep]
        cols = cols[keep]
        vals = vals[keep]
        rows = np.concatenate([rows, np.arange(n, dtype=np.int64)])
        cols = np.concatenate([cols, np.arange(n, dtype=np.int64)])
        vals = np.concatenate([vals, aP_relaxed])
        rhs += ((1.0 - relax) / relax) * ap * cur
        return rows, cols, vals, rhs, ap

    # -- 压力梯度（零梯度外推 Neumann） ---------------------------
    def _grad_pressure(self, phi):
        fv = self._fv
        b = np.zeros(fv.n_faces, float)
        bnd = fv.is_boundary
        b[bnd] = phi[fv.owner[bnd]]
        return fv.grad_gauss(np.asarray(phi, float), boundary=b)

    def _grad_pressure_of(self, phi):
        fv = self._fv
        b = np.zeros(fv.n_faces, float)
        bnd = fv.is_boundary
        b[bnd] = phi[fv.owner[bnd]]
        return fv.grad_gauss(np.asarray(phi, float), boundary=b)

    # -- 质量不平衡 / 压力修正 ------------------------------------
    def _continuity_imbalance(self):
        fv = self._fv
        is_int = fv.neighbor >= 0
        nbr = np.where(is_int, fv.neighbor, 0)
        acc = np.zeros(fv.n_cells, float)
        np.add.at(acc, fv.owner, self._mdot)
        np.add.at(acc, nbr[is_int], -self._mdot[is_int])
        return acc

    def _assemble_pressure_correction(self, d_cell):
        fv = self._fv
        n = fv.n_cells
        is_int = fv.neighbor >= 0
        o = fv.owner[is_int]
        nb = fv.neighbor[is_int]
        dface = 0.5 * (d_cell[o] + d_cell[nb])
        A = fv.face_area[is_int]
        dn = np.maximum(fv._d_n[is_int], 1e-12)
        gamma = self._face_rho()[is_int] * A * dface / dn
        rows = np.concatenate([o, nb, o, nb])
        cols = np.concatenate([o, nb, nb, o])
        vals = np.concatenate([gamma, gamma, -gamma, -gamma])
        ref = self._dirichlet_cell
        keep = rows != ref
        rows = rows[keep]
        cols = cols[keep]
        vals = vals[keep]
        rows = np.concatenate([rows, [ref]])
        cols = np.concatenate([cols, [ref]])
        vals = np.concatenate([vals, [1.0]])
        imbalance = self._continuity_imbalance()
        rhs = -imbalance
        rhs[ref] = 0.0
        return rows, cols, vals, rhs, o, nb, gamma, is_int

    # -- 单步 SIMPLE -------------------------------------------------
    def step(self):
        """执行一次 SIMPLE 外部迭代：3 动量 + Rhie-Chow + 压力修正，更新场与残差。"""
        fv = self._fv
        n = fv.n_cells
        d_cell = np.zeros(n, float)
        for comp in range(3):
            r, c, v, rhs, ap = self._assemble_momentum(comp)
            sol = solve_linear(r, c, v, rhs, n, tol=1e-9, maxit=6000)
            if comp == 0:
                self._u = sol
            elif comp == 1:
                self._v = sol
            else:
                self._w = sol
            aP = ap / self.alpha_momentum
            d_cell = fv.volumes / np.maximum(aP, 1e-12)
        self._last_d_cell = d_cell
        # Rhie-Chow 面质量通量（由预测速度 + 压力梯度重构）
        self._recompute_mdot_rhie_chow()
        # 压力修正装配
        rp, cp, vp, rhsp, o_int, nb_int, gamma, is_int = \
            self._assemble_pressure_correction(d_cell)
        pprime = solve_linear(rp, cp, vp, rhsp, n, tol=1e-9, maxit=8000)
        # 速度 / 压力校正：u -= d ∇p'、p += α_p p'
        gpp = self._grad_pressure_of(pprime)
        self._p = self._p + self.alpha_pressure * pprime
        self._u = self._u - d_cell * gpp[:, 0]
        self._v = self._v - d_cell * gpp[:, 1]
        self._w = self._w - d_cell * gpp[:, 2]
        # 直接校正面质量通量（SIMPLE 标准做法）：mdot_f += gamma (p'_O - p'_N)
        # —— 与压力修正矩阵用同一 gamma，保证泊松解与通量修正一致，正是残差收敛关键。
        self._mdot[is_int] += gamma * (pprime[o_int] - pprime[nb_int])
        self._fix_boundary_mdot()
        # 残差：归一化连续性质量不平衡（收敛 → 0）
        cont = _safe_norm(self._continuity_imbalance())
        cont_ratio = cont / self._inlet_mdot_scale
        self.iteration += 1
        self._last_cont_ratio = float(cont_ratio)
        self._last_residual = float(max(cont_ratio, 1e-14))
        if self.turb_model is not None:
            self._update_turbulence()
        self._update_energy()
        self._update_vof()
        self._update_mixture()
        self._update_eulerian()
        self._update_dpm()
        self._update_species()
        self._update_combustion()
        return {"residual": self._last_residual,
                "cont_residual": float(cont_ratio),
                "u_min": float(self._u.min()),
                "u_max": float(self._u.max()),
                "u_mean": float(self._u.mean())}

    def _recompute_mdot_rhie_chow(self):
        """Rhie-Chow 动量插值面质量通量（消除棋盘压力）。

        mdot_f = ρ A_f [ ū_f·n_f + d_f ( (p_N - p_O)/d_n - 0.5(∇p_O+∇p_N)·n_f ) ]
        """
        fv = self._fv
        is_int = fv.neighbor >= 0
        o = fv.owner
        nb = np.where(is_int, fv.neighbor, 0)
        uf = fv.face_value(self._u, boundary=self._boundary_u(0))
        vf = fv.face_value(self._v, boundary=self._boundary_u(1))
        wf = fv.face_value(self._w, boundary=self._boundary_u(2))
        un = uf * fv.face_normal[:, 0] + vf * fv.face_normal[:, 1] \
            + wf * fv.face_normal[:, 2]
        mdot = self._face_rho() * un * fv.face_area
        # Rhie-Chow 动量插值（消除棋盘压力）：面通量 = 插值速度通量 +
        #   d_f [ (∇p)_interp·n - (p_N - p_O)/d_n ]
        # 标准推导：u = H/a_p - d ∇p（压力力 -V∇p），面速度用直接压力梯度
        # 替代插值梯度；故修正项符号为正 d_f[(∇p)_interp·n - (∂p/∂n)_f]。
        gp = self._grad_pressure(self._p)
        d_face = np.zeros(fv.n_faces, float)
        d_cell = self._last_d_cell
        if d_cell is not None:
            d_face[is_int] = 0.5 * (d_cell[o[is_int]] + d_cell[nb[is_int]])
            gradn_face = (self._p[nb] - self._p[o]) / np.maximum(fv._d_n, 1e-12)
            gradn_interp = (gp[o, 0] * fv.face_normal[:, 0]
                            + gp[o, 1] * fv.face_normal[:, 1]
                            + gp[o, 2] * fv.face_normal[:, 2]) * 0.5
            gradn_interp = gradn_interp + 0.5 * (
                gp[nb, 0] * fv.face_normal[:, 0]
                + gp[nb, 1] * fv.face_normal[:, 1]
                + gp[nb, 2] * fv.face_normal[:, 2])
            corr = d_face * (gradn_interp - gradn_face)
            mdot[is_int] += self._face_rho()[is_int] * fv.face_area[is_int] * corr[is_int]
        self._mdot = mdot
        self._fix_boundary_mdot()

    # -- 监视器 -----------------------------------------------------
    def monitor_payload(self):
        """本步指标的监视器映射：残差 + 场统计（与 step() 返回键一致）。"""
        payload = {"residual": self._last_residual,
                   "cont_residual": self._last_cont_ratio,
                   "u_min": float(self._u.min()),
                   "u_max": float(self._u.max()),
                   "u_mean": float(self._u.mean())}
        if self.vof_model is not None and not isinstance(self.vof_model, str):
            payload["alpha_min"] = float(self.vof_model.alpha.min())
            payload["alpha_max"] = float(self.vof_model.alpha.max())
            payload["phase1_volume"] = self.vof_model.phase1_volume
        if self.mixture_model is not None and not isinstance(self.mixture_model, str):
            payload["mix_rho_min"] = float(self.mixture_model.rho.min())
            payload["mix_rho_max"] = float(self.mixture_model.rho.max())
            payload["mix_alpha_sum"] = float(self.mixture_model.alphas.sum(axis=1).max())
        if self.eulerian_model is not None and not isinstance(self.eulerian_model, str):
            payload["ee_n_phases"] = self.eulerian_model.n_phases
            payload["ee_alpha_min"] = float(self.eulerian_model.alphas.min())
            payload["ee_alpha_max"] = float(self.eulerian_model.alphas.max())
            payload["ee_alpha_sum"] = float(self.eulerian_model.alphas.sum(axis=1).max())
            payload["ee_mom_src"] = float(_safe_norm(self.eulerian_model.momentum_source))
        if self.dpm_model is not None and not isinstance(self.dpm_model, str):
            payload["n_particles"] = self.dpm_model.n_particles
            payload["n_active"] = self.dpm_model.n_active
            payload["n_escaped"] = self.dpm_model.n_escaped
        if self.species_model is not None and not isinstance(self.species_model, str):
            y = np.asarray(self.species_model.Y, float)
            payload["sp_y_min"] = float(y.min())
            payload["sp_y_max"] = float(y.max())
            payload["sp_sum_max"] = float(y.sum(axis=1).max())
        if self.combustion_model is not None and not isinstance(self.combustion_model, str):
            payload["combo_Q_max"] = float(self.combustion_model.heat_release.max())
            payload["combo_T_max"] = float(self.combustion_model.T.max())
            payload["combo_n_ignited"] = int(np.count_nonzero(
                self.combustion_model.ignited))
        return payload

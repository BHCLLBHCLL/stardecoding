# -*- coding: utf-8 -*-
"""P 波 P5：压力基求解器（SIMPLE 分离 + Rhie-Chow + 稀疏线性求解，occ/scdm 两环境皆可用）。

覆盖：
  PressureSolver：网格/边界初始化，field/velocity/pressure/mass_flux/residual、
                  连续性不守恒的求解器（可在 SolverBackend 下驱动，P10 闭环）
  step() 残差下降：P5 验收 —— 残差曲线健康（启动期小振荡后单调衰减至 ~1e-8）
  全局质量守恒：sum(mdot)~0、出口对入口、内部散度归一化范数 <1e-6
  物理解：速度场有限、入口速度量级合理
  solve_linear：COO 稀疏求解，numpy / scipy / auto 三路径同一精确解；
                RHS 维度不匹配报错
  错误路径：非四面体 / 过小网格报 ValueError

验收核心（P5 行）：压力基求解器 —— SIMPLE/PISO + Coupled，AMG(pyamg)/ILU，
残差下降曲线健康。
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

from fvm_core import cube_tet_mesh
from pressure_solver import PressureSolver, solve_linear
from solver_run import SolverBackend, StopCriterion, _safe_norm


# ---------------------------------------------------------------- 工具
def _make(nx=2, **kw):
    V, C = cube_tet_mesh(nx)
    kw.setdefault("mu", 1e-3)
    kw.setdefault("inlet_velocity", (1.0, 0.0, 0.0))
    kw.setdefault("alpha_momentum", 0.7)
    kw.setdefault("alpha_pressure", 0.3)
    return PressureSolver(V, C, **kw)


def _run(s, n=40):
    hist = []
    for _ in range(n):
        hist.append(float(s.step()["residual"]))
    return hist


# ---------------------------------------------------------------- 初始化 / API
def test_pressure_solver_initialization():
    s = _make(nx=2)
    n_cells = len(s.cells)
    assert s.iteration == 0
    assert s.field().shape == (n_cells,)
    assert s.velocity().shape == (n_cells, 3)
    assert s.pressure().shape == (n_cells,)
    assert s.mass_flux().shape == (s._fv.n_faces,)
    assert np.isnan(s.residual())            # 未 step 前无残差
    assert s._inlet_faces.size > 0 and s._outlet_faces.size > 0


def test_pressure_solver_boundary_partition():
    s = _make(nx=2)
    fv = s._fv
    all_boundary = np.arange(fv.n_faces)[fv.is_boundary]
    union = np.concatenate([s._inlet_faces, s._outlet_faces, s._wall_faces])
    assert all_boundary.size == union.size            # 边界全覆盖且不重叠
    assert len(set(union.tolist())) == union.size


def test_pressure_solver_step_returns_keys():
    s = _make(nx=2)
    p = s.step()
    assert set(p) == {"residual", "cont_residual", "u_min", "u_max", "u_mean"}
    assert np.isfinite(p["residual"])
    assert s.monitor_payload()["residual"] == pytest.approx(p["residual"])
    assert s.iteration == 1


# ---------------------------------------------------------------- P5 验收：残差下降曲线健康
def test_pressure_solver_residual_curve_healthy_nx2():
    s = _make(nx=2)
    hist = _run(s, n=40)
    first = hist[:20]
    tail = hist[20:]
    assert hist[-1] < 1e-5                        # 收敛到低残差
    assert max(tail) < max(first) * 0.1           # 曲线后半程整体远低于前半程（健康下降）
    assert hist[-1] < hist[0]                     # 相对初值显著下降


def test_pressure_solver_residual_curve_healthy_nx3():
    s = _make(nx=3)
    hist = _run(s, n=40)
    assert hist[-1] < 1e-5
    assert max(hist[20:]) < max(hist[:20]) * 0.1


def test_pressure_solver_residual_net_decay():
    # 启动期允许振荡，但末态必须低于中程：曲线整体向零收敛
    s = _make(nx=2, mu=1e-2)
    hist = _run(s, n=40)
    assert hist[-1] < hist[20]


# ---------------------------------------------------------------- 守恒性
def test_pressure_solver_global_mass_conservation():
    s = _make(nx=2)
    _run(s, n=40)
    mdot = s.mass_flux()
    # 全局质量守恒 = 边界面净通量 sum(imb)=0（内部面对 global sum 贡献 0）
    assert abs(s._continuity_imbalance().sum()) < 1e-8
    inflow = mdot[s._inlet_faces].sum()
    outflow = mdot[s._outlet_faces].sum()
    assert abs(inflow + outflow) < 1e-8                    # 出口对入口闭合


def test_pressure_solver_interior_divergence_free():
    s = _make(nx=2)
    _run(s, n=40)
    imb = s._continuity_imbalance()
    assert _safe_norm(imb) < 1e-6                          # 内部散度趋零
    assert abs(imb.sum()) < 1e-9                           # 全局不平衡趋零


def test_pressure_solver_conservation_nx3():
    s = _make(nx=3)
    _run(s, n=40)
    mdot = s.mass_flux()
    assert abs(s._continuity_imbalance().sum()) < 1e-8      # 边界面净通量守恒
    inflow = mdot[s._inlet_faces].sum()
    outflow = mdot[s._outlet_faces].sum()
    assert abs(inflow + outflow) < 1e-8
    assert _safe_norm(s._continuity_imbalance()) < 1e-6


# ---------------------------------------------------------------- 物理解
def test_pressure_solver_velocity_field_sane():
    s = _make(nx=2)
    _run(s, n=40)
    u = s.velocity()
    assert np.isfinite(u).all()
    assert s.field().min() >= 0.0                            # 速度幅值非负
    mean_u = s.velocity()[:, 0].mean()
    assert 0.0 < mean_u <= 1.0 + 1e-6                        # 入口速度 1.0，无滑移壁面均值略低


# ---------------------------------------------------------------- 稀疏线性求解
def _tridiag(n):
    row, col, data = [], [], []
    for i in range(n):
        row.append(i); col.append(i); data.append(2.0)
        if i > 0:
            row.append(i); col.append(i - 1); data.append(-1.0)
            row.append(i - 1); col.append(i); data.append(-1.0)
    return np.array(row, np.int64), np.array(col, np.int64), np.array(data, float)


@pytest.mark.parametrize("kind", ["numpy", "scipy", "auto"])
def test_solve_linear_exact(kind):
    n = 6
    row, col, data = _tridiag(n)
    b = np.ones(n)
    x = solve_linear(row, col, data, b, n, tol=1e-12, maxit=8000, kind=kind)
    ax = np.zeros(n)
    np.add.at(ax, row, data * x[col])
    assert _safe_norm(ax - b) < 1e-8


def test_solve_linear_numpy_no_scipy_path():
    # 强制 numpy 路径：M 矩阵精确解
    n = 5
    row, col, data = _tridiag(n)
    x = solve_linear(row, col, data, np.ones(n), n, tol=1e-12, maxit=8000,
                     kind="numpy")
    assert np.isfinite(x).all() and _safe_norm(x) > 0.0


def test_solve_linear_rejects_bad_rhs():
    n = 4
    row, col, data = _tridiag(n)
    with pytest.raises(ValueError):
        solve_linear(row, col, data, np.ones(n + 1), n)


def test_solve_linear_consistent_diagonal_poisson_like():
    # 类泊松（对角占优）矩阵：numpy 路径也收敛到唯一解
    n = 8
    row, col, data = _tridiag(n)
    b = np.linspace(0.0, 1.0, n)
    x = solve_linear(row, col, data, b, n, tol=1e-12, maxit=12000, kind="numpy")
    ax = np.zeros(n)
    np.add.at(ax, row, data * x[col])
    assert _safe_norm(ax - b) < 1e-8


# ---------------------------------------------------------------- 错误路径
def test_pressure_solver_rejects_non_tet():
    V, C = cube_tet_mesh(1)
    with pytest.raises(ValueError):
        PressureSolver(V, C[:, :3])


def test_pressure_solver_rejects_small_mesh():
    with pytest.raises(ValueError):
        PressureSolver(np.zeros((3, 3)), np.zeros((0, 4), np.int64))


def test_pressure_solver_set_mesh_rebuilds():
    s = _make(nx=2)
    n0 = len(s.cells)
    V, C = cube_tet_mesh(3)
    s.set_mesh(V, C)
    assert len(s.cells) != n0
    assert s.field().shape == (len(C),)
    assert s.iteration == 0


# ---------------------------------------------------------------- P10 闭环
def test_pressure_solver_solverbackend_closed_loop():
    """P5 接入 solver_run：PressureSolver 可直接被 SolverBackend 驱动（P10 闭环）。

    默认监视器（residual/u_min/u_max/u_mean）随 step() 采样；run_loop 至步数判停
    COMPLETED；残差曲线健康（终值远低于峰值，净衰减）；曲线元组/报告可产出。
    """
    s = _make(nx=2)
    crit = StopCriterion(max_iterations=25, name="p5_closed")
    be = SolverBackend(solver=s, stop_criteria=[crit])
    be.initialize()
    res = be.run_loop()
    assert res["state"] == "COMPLETED"
    assert res["iteration"] == 25
    assert "最大迭代" in res["reason"]
    m = be.metrics()
    assert m["state"] == "COMPLETED"
    rows = m["monitors"]
    names = {r["name"] for r in rows}
    assert {"residual", "u_min", "u_max", "u_mean"} <= names
    rmon = next(r for r in rows if r["name"] == "residual")
    assert rmon["n"] == 25
    assert rmon["last"] < 1e-4, "P10 闭环残差应收敛: %.2e" % rmon["last"]
    assert rmon["last"] < rmon["y_max"], "P10 残差曲线应净衰减"
    assert rmon["y_min"] >= 0.0
    # GUI SeriesCanvas 兼容曲线 + 文本报告（P10 运行时产物）
    items = be.curve_items()
    assert items and items[0][0] == "residual"
    assert any("迭代" in ln for ln in be.report_lines())

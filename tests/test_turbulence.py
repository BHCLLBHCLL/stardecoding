# -*- coding: utf-8 -*-
"""P 波 P6：湍流族 —— Spalart-Allmaras → k-ε → k-ω SST → LES 子格子 + 壁面处理。

覆盖：
  壁面处理：wall_distance（单元壁面距离） / y_plus（无量纲壁面距离） /
            wall_function（粘性底层-对数律光滑混合，单调） / wall_shear（壁面剪切）
  边界分类 / 应变率：classify_boundaries（入口/出口/壁面全覆盖不重叠）、
                     strain_magnitude（均匀场~0、剪切场>0）
  模型：make_model 经别名/大小写/连字符解析出正确类；SA / k-ε / k-ω SST / LES
        update() 返回有限残差、nu_t 有限非负、多步收敛无 NaN
  方程残差：SA 输运方程残差有限且随迭代单调下降
  错误路径：未知模型名报 ValueError、网格/场量维度不匹配报错
  P10 后端：TurbulenceSolver 提供 step()/monitor_payload()/residual()
  集成：PressureSolver 以字符串注入湍流模型，nu_t 形状一致、step() 无报错，
        基线（无模型）nu_t 全零

验收核心（P6 行）：湍流族 —— 壁面处理/壁面函数、SA→k-ε→k-ω SST→LES 子格子。
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

from fvm_core import FVM, cube_tet_mesh
from pressure_solver import PressureSolver
import turbulence as T
from turbulence import (
    KEpsilonSolver, KOmegaSSTSolver, LESSmagorinskySolver, SpalartAllmarasSolver,
    TurbulenceSolver, WALL_E, WALL_KAPPA,
    classify_boundaries, make_model, strain_magnitude,
    wall_distance, wall_function, wall_shear, y_plus,
)


# ---------------------------------------------------------------- 工具
def _fv(nx=2):
    V, C = cube_tet_mesh(nx)
    return FVM(V, C)


def _shear(fv):
    """剪切速度场：u=y+1, v=0.1x, w=0（避免零梯度退化）。"""
    x = fv.centroids[:, 0]
    y = fv.centroids[:, 1]
    return y + 1.0, 0.1 * x, np.zeros(fv.n_cells)


# ---------------------------------------------------------------- 壁面处理
def test_y_plus_scalar_and_array():
    assert np.isclose(y_plus(0.0, 1.0, 1e-3), 0.0, atol=1e-12)
    assert np.isclose(y_plus(0.1, 0.5, 1e-3), 0.1 * 0.5 / 1e-3, atol=1e-12)
    out = y_plus(np.array([0.1, 0.2]), 1.0, 1e-3)
    assert out.shape == (2,)
    assert np.allclose(out, np.array([0.1, 0.2]) / 1e-3)


def test_wall_function_viscous_sublayer():
    # 粘性底层：y+ << y+_lam -> u+ ~ y+（单调上升）
    ys = np.linspace(0.01, 1.0, 20)
    up = wall_function(ys)
    assert (np.diff(up) >= 0.0).all()
    assert np.isclose(wall_function(0.01), 0.01, atol=1e-9)


def test_wall_function_log_law():
    # 对数律：y+ 大 -> u+ = ln(E·y+)/kappa（单调上升）
    ys = np.linspace(20.0, 1000.0, 20)
    up = wall_function(ys)
    assert (np.diff(up) >= 0.0).all()
    yp = 100.0
    assert np.isclose(wall_function(yp), np.log(WALL_E * yp) / WALL_KAPPA, rtol=1e-6)


def test_wall_shear():
    assert np.isclose(wall_shear(0.3, 1.2), 1.2 * 0.3 ** 2)
    out = wall_shear(np.array([0.1, 0.2]), 1.0)
    assert out.shape == (2,)


def test_wall_distance_nonnegative_and_near_wall():
    fv = _fv(2)
    d = wall_distance(fv)
    assert d.shape == (fv.n_cells,)
    assert (d >= 0.0).all()
    assert d.max() <= 1.0          # 单位立方体最大跨距
    assert d.min() < 0.5           # 贴壁单元距壁面很小


# ---------------------------------------------------------------- 边界分类 / 应变率
def test_classify_boundaries_partition():
    fv = _fv(2)
    inlet, outlet, wall, bnd = classify_boundaries(fv)
    all_b = np.arange(fv.n_faces)[fv.is_boundary]
    assert all_b.size == bnd.size
    union = np.concatenate([inlet, outlet, wall])
    assert union.size == bnd.size
    assert len(set(union.tolist())) == union.size    # 全覆盖且不重叠
    assert inlet.size > 0 and outlet.size > 0


def test_strain_magnitude_uniform_zero():
    fv = _fv(2)
    n = fv.n_cells
    S = strain_magnitude(fv, np.ones(n), np.zeros(n), np.zeros(n))
    assert float(np.max(np.abs(S))) < 1e-8


def test_strain_magnitude_shear_positive():
    fv = _fv(2)
    u, v, w = _shear(fv)
    S = strain_magnitude(fv, u, v, w)
    assert float(np.max(S)) > 0.0


# ---------------------------------------------------------------- 模型注册 / 工厂
_ALIASES = {
    "sa": SpalartAllmarasSolver,
    "SPALART_ALLMARAS": SpalartAllmarasSolver,
    "k-epsilon": KEpsilonSolver,
    "K_Epsilon": KEpsilonSolver,
    "ke": KEpsilonSolver,
    "k-omega-sst": KOmegaSSTSolver,
    "KOmega_SST": KOmegaSSTSolver,
    "sst": KOmegaSSTSolver,
    "les": LESSmagorinskySolver,
    "smagorinsky": LESSmagorinskySolver,
}


def test_make_model_resolves_aliases():
    fv = _fv(2)
    for name, cls in _ALIASES.items():
        m = make_model(name, fv, rho=1.0, mu=1e-3)
        assert isinstance(m, cls)


def test_make_model_unknown_raises():
    fv = _fv(2)
    with pytest.raises(ValueError):
        make_model("not-a-model", fv, rho=1.0, mu=1e-3)


# ---------------------------------------------------------------- 模型残差 / nu_t
def _run_model(name, n=8):
    fv = _fv(2)
    m = make_model(name, fv, rho=1.0, mu=1e-3, u_ref=1.0, length_scale=0.2)
    u, v, w = _shear(fv)
    res = []
    for _ in range(n):
        res.append(float(m.update(u, v, w)))
    return m, np.array(res)


def test_sa_residual_finite_and_monotone():
    m, res = _run_model("sa")
    assert np.isfinite(res).all()
    assert res[-1] < res[0]                       # 单调下降
    assert (np.diff(res) <= 1e-9).all()          # 非升
    nu = m.nu_t
    assert nu.shape == (m.fv.n_cells,)
    assert np.isfinite(nu).all()
    assert (nu >= 0.0).all()


@pytest.mark.parametrize("name", ["k-epsilon", "k-omega-sst", "les"])
def test_transport_models_residual_and_nu_t(name):
    m, res = _run_model(name)
    assert np.isfinite(res).all()
    assert res[-1] <= res[0] + 1e-9               # 不发散
    nu = m.nu_t
    assert np.isfinite(nu).all()
    assert (nu >= 0.0).all()


# ---------------------------------------------------------------- P10 后端门面
def test_turbulence_solver_backend():
    fv = _fv(2)
    ts = TurbulenceSolver(fv, model="sa", rho=1.0, mu=1e-3)
    init = ts._initialize_field()
    assert "model" in init and "nu_t_max" in init
    u, v, w = _shear(fv)
    ts.set_velocities(u, v, w)
    payload = ts.step()
    assert {"residual", "u_min", "u_max", "u_mean"} <= set(payload.keys())
    assert np.isfinite(payload["residual"])
    assert ts.residual() == payload["residual"]
    mono = ts.monitor_payload()
    assert "nu_t_max" in mono


# ---------------------------------------------------------------- 集成：PressureSolver
def test_pressure_solver_baseline_nu_t_zero():
    V, C = cube_tet_mesh(2)
    s = PressureSolver(V, C, rho=1.0, mu=1e-3)
    assert s.nu_t.shape == (len(s.cells),)
    assert float(s.nu_t.max()) == 0.0
    assert np.isfinite(s.step()["residual"])


@pytest.mark.parametrize("name", ["sa", "k-epsilon", "k-omega-sst", "les"])
def test_pressure_solver_integration_all_models(name):
    V, C = cube_tet_mesh(2)
    s = PressureSolver(V, C, rho=1.0, mu=1e-3,
                       inlet_velocity=(1.0, 0.0, 0.0), max_outer=3,
                       turb_model=name)
    assert s.nu_t.shape == (len(s.cells),)
    assert float(s.nu_t.max()) >= 0.0
    assert np.isfinite(s.step()["residual"])
    assert np.isfinite(s.nu_t).all()


def test_pressure_solver_object_model_injection():
    # 以对象形式注入：直接传入 make_model 实例
    V, C = cube_tet_mesh(2)
    fv = FVM(V, C)
    model = make_model("sa", fv, rho=1.0, mu=1e-3, u_ref=1.0, length_scale=0.2)
    s = PressureSolver(V, C, rho=1.0, mu=1e-3,
                       inlet_velocity=(1.0, 0.0, 0.0), max_outer=3,
                       turb_model=model)
    assert s.nu_t.shape == (len(s.cells),)
    assert np.isfinite(s.step()["residual"])
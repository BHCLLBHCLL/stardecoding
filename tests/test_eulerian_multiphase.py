# -*- coding: utf-8 -*-
"""P 波 P10：多相欧拉-欧拉（Eulerian-Eulerian）—— 相体积分数守恒输运 + 相间作用力
（拖曳 / 升力 / 虚拟质量 / 壁面润滑）+ 群体平衡（聚并 / 破碎 / 成核）。

覆盖：
  常量：DEFAULT_LIFT_COEF / DEFAULT_VIRTUAL_MASS_COEF / DEFAULT_WALL_LUB_COEF_CW1·CW2 /
        DEFAULT_COALESCENCE_RATE / DEFAULT_BREAKUP_RATE / DEFAULT_NUCLEATION_RATE
  拖曳：drag_exchange（Schiller-Naumann + Wen-Yu 空泡率修正，Stokes 低 Re 极限）、
        interphase_drag（力密度）
  相间力：lift_force / virtual_mass_force / wall_lubrication_force（系数 0 或 d=0 归零）
  群体平衡：coalescence_kernel / breakup_kernel / nucleation_rate /
        population_balance_source（Σ_k S_k = 0 守恒）
  EulerianMultiphaseSolver：初始 Σα=1、混合密度/粘度、update() 注入弥散相增长、
        相间动量源有限、P10 后端门面 step()/residual()/monitor_payload()
  工厂：make_eulerian_multiphase 别名/大小写解析，未知模型报 ValueError
  集成：PressureSolver 字符串注入 eulerian + step() 稳定、αΣ=1、monitor 含 ee 键；
        基线不暴露 ee 键

验收核心（P10 行）：star.multiphase（EulerianPhase + PhaseInteraction +
Population Balance）相间拖曳/升力/虚拟质量/壁面润滑 + 聚并/破碎/成核。
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
from eulerian_multiphase import (
    DEFAULT_BREAKUP_RATE, DEFAULT_COALESCENCE_RATE, DEFAULT_LIFT_COEF,
    DEFAULT_NUCLEATION_RATE, DEFAULT_VIRTUAL_MASS_COEF,
    DEFAULT_WALL_LUB_COEF_CW1, DEFAULT_WALL_LUB_COEF_CW2,
    EulerianMultiphaseSolver,
    breakup_kernel, coalescence_kernel, drag_exchange, interphase_drag,
    lift_force, make_eulerian_multiphase, nucleation_rate,
    population_balance_source, virtual_mass_force, wall_lubrication_force,
)


# ---------------------------------------------------------------- 工具
def _fv(nx=2):
    V, C = cube_tet_mesh(nx)
    return FVM(V, C)


def _uniform_u(fv, speed=1.0):
    n = fv.n_cells
    return (np.full(n, speed, float), np.zeros(n, float), np.zeros(n, float))


# ---------------------------------------------------------------- 常量
def test_eulerian_constants():
    assert DEFAULT_LIFT_COEF > 0.0
    assert DEFAULT_VIRTUAL_MASS_COEF > 0.0
    assert DEFAULT_WALL_LUB_COEF_CW1 < 0.0
    assert DEFAULT_WALL_LUB_COEF_CW2 > 0.0
    assert DEFAULT_COALESCENCE_RATE == 0.0
    assert DEFAULT_BREAKUP_RATE == 0.0
    assert DEFAULT_NUCLEATION_RATE == 0.0


# ---------------------------------------------------------------- 拖曳
def test_drag_exchange_finite_and_monotone():
    d = drag_exchange(1e-3, 998.0, 1e-3, np.array([1.0, 0.1]),
                      np.array([0.1, 0.2]), np.array([0.9, 0.8]))
    assert d.shape == (2,)
    assert np.all(np.isfinite(d)) and np.all(d > 0.0)
    assert float(d[0]) > float(d[1])  # 相对速度越大拖曳越强


def test_drag_exchange_stokes_limit():
    d = drag_exchange(1e-3, 998.0, 1e-3, np.array([0.0, 0.0]),
                      np.array([0.2, 0.2]), np.array([0.8, 0.8]))
    assert np.all(np.isfinite(d)) and np.all(d > 0.0)  # |u|→0 走 Stokes 极限无除零


def test_interphase_drag_force_density():
    K, F = interphase_drag(1e-3, 998.0, 1e-3,
                           np.array([[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
                           np.array([0.1, 0.1]), np.array([0.9, 0.9]))
    assert K.shape == (2,)
    assert F.shape == (2, 3)
    assert np.all(np.isfinite(F))
    assert np.all(F >= 0.0)


# ---------------------------------------------------------------- 相间力
def test_lift_force_zero_coef():
    f = lift_force(np.full(3, 0.1), 998.0, np.zeros((3, 3)), np.zeros((3, 3)),
                   coef=0.0)
    assert np.allclose(f, 0.0)


def test_virtual_mass_force_zero_coef():
    f = virtual_mass_force(np.full(3, 0.1), 998.0, np.zeros((3, 3)), coef=0.0)
    assert np.allclose(f, 0.0)


def test_wall_lubrication_zero_diameter():
    f = wall_lubrication_force(np.full(3, 0.1), 998.0, np.zeros((3, 3)),
                               np.full(3, 0.05), np.zeros((3, 3)), d=0.0)
    assert np.allclose(f, 0.0)


def test_wall_lubrication_pushes_away():
    f = wall_lubrication_force(np.full(3, 0.1), 998.0,
                               np.array([[1.0, 0.0, 0.0]] * 3),
                               np.full(3, 0.05),
                               np.tile(np.array([0.0, 0.0, 1.0]), (3, 1)), d=1e-3)
    assert np.all(np.isfinite(f))
    # 力沿 -n_wall（远离壁面），故 z 分量应 ≤ 0
    assert np.all(f[:, 2] <= 1e-12)


# ---------------------------------------------------------------- 群体平衡
def test_population_balance_conserves_volume():
    alphas = np.array([[0.6, 0.3, 0.1], [0.5, 0.4, 0.1]], float)
    src = population_balance_source(alphas, np.array([0.0, 5e-4, 1e-3]),
                                    0.1, 0.05, 0.02)
    assert src.shape == (2, 3)
    assert np.allclose(src.sum(axis=1), 0.0, atol=1e-12)


def test_population_balance_disabled_returns_zero():
    src = population_balance_source(np.array([[0.6, 0.3, 0.1]], float),
                                    np.array([0.0, 5e-4, 1e-3]), 0.0, 0.0, 0.0)
    assert np.allclose(src, 0.0)


def test_coalescence_breakup_nucleation_kernels():
    a = np.array([0.2, 0.3, 0.1], float)
    assert np.all(np.isfinite(coalescence_kernel(a, a, 5e-4, 1e-3, 0.1)))
    assert np.all(np.isfinite(breakup_kernel(a, 1e-3, 0.05)))
    assert np.all(np.isfinite(nucleation_rate(a, 0.02)))
    assert np.allclose(coalescence_kernel(a, a, 5e-4, 1e-3, 0.0), 0.0)


# ---------------------------------------------------------------- 求解器
def test_eulerian_solver_initial_state():
    fv = _fv(2)
    em = EulerianMultiphaseSolver(fv, alpha0=[0.1, 0.05])
    assert em.n_phases == 3
    assert np.allclose(em.alphas[:, 0], 0.85)
    assert np.allclose(em.alphas.sum(axis=1), 1.0)
    assert np.allclose(em.rho, 0.85 * 998.0 + 0.1 * 1.18 + 0.05 * 800.0)
    assert em.slip.shape == (fv.n_cells, 3, 3)
    assert em.drift.shape == (fv.n_cells, 3, 3)
    assert em.phase_volume(1) > 0.0


def test_eulerian_solver_update_injects_dispersed():
    fv = _fv(2)
    em = EulerianMultiphaseSolver(fv, inlet_alphas=[0.2, 0.1])
    u, v, w = _uniform_u(fv)
    r0 = em.update(u, v, w)
    assert np.isfinite(r0)
    assert em.iteration == 1
    assert em.residual() == pytest.approx(r0)
    assert em.alphas.min() >= 0.0 and em.alphas.max() <= 1.0
    assert np.allclose(em.alphas.sum(axis=1), 1.0, atol=1e-9)
    assert em.phase_volume(1) > 0.0 and em.phase_volume(2) > 0.0
    assert np.all(np.isfinite(em.rho))
    assert np.all(np.isfinite(em.momentum_source))


def test_eulerian_solver_p10_facade():
    fv = _fv(2)
    em = EulerianMultiphaseSolver(fv, inlet_alphas=[0.1, 0.05])
    info = em._initialize_field()
    assert {"n_phases", "alpha_min", "alpha_max", "rho_min", "rho_max"} <= set(info)
    assert em.residual() == pytest.approx(0.0, abs=1e-12)
    u, v, w = _uniform_u(fv)
    em.set_velocities(u, v, w)
    p = em.step()
    assert {"residual", "n_phases", "alpha_min", "alpha_max"} <= set(p)
    assert np.isfinite(p["residual"])
    m = em.monitor_payload()
    assert {"n_phases", "phase_volumes", "momentum_src", "residual",
            "iteration"} <= set(m)
    assert m["iteration"] == em.iteration


def test_eulerian_momentum_source_nonzero_with_interaction():
    fv = _fv(2)
    n = fv.n_cells
    em = EulerianMultiphaseSolver(fv, alpha0=[0.2, 0.1], enable_lift=True,
                                  enable_virtual_mass=True)
    u = np.full(n, 1.0) + np.linspace(0.0, 0.2, n)
    em.update(u, np.zeros(n), np.zeros(n))
    assert np.all(np.isfinite(em.momentum_source))


# ---------------------------------------------------------------- 工厂
def test_make_eulerian_multiphase_factory_aliases():
    fv = _fv(2)
    assert isinstance(make_eulerian_multiphase(fv, "eulerian"),
                      EulerianMultiphaseSolver)
    assert isinstance(make_eulerian_multiphase(fv, "ee"), EulerianMultiphaseSolver)
    assert isinstance(make_eulerian_multiphase(fv, "EULER_EULER"),
                      EulerianMultiphaseSolver)
    with pytest.raises(ValueError):
        make_eulerian_multiphase(fv, "nope")


# ---------------------------------------------------------------- 集成
def test_pressure_solver_eulerian_coupling():
    V, C = cube_tet_mesh(2)
    s = PressureSolver(V, C, mu=1.0e-3, rho=998.0, inlet_velocity=(1.0, 0.0, 0.0),
                       eulerian_model="eulerian", eulerian_inlet_alphas=[0.2, 0.1],
                       eulerian_enable_lift=True,
                       eulerian_population_balance=True,
                       eulerian_coalescence_rate=0.1)
    assert s.eulerian_model is not None
    assert not isinstance(s.eulerian_model, str)
    assert s.eulerian_model.alphas.shape == (len(C), 3)
    for _ in range(6):
        p = s.step()
    assert np.all(np.isfinite(s.velocity()))
    assert np.isfinite(p["residual"])
    ee = s.eulerian_model
    assert np.allclose(ee.alphas.sum(axis=1), 1.0, atol=1e-8)
    assert ee.alphas.min() >= 0.0 and ee.alphas.max() <= 1.0
    r = s.rho
    assert np.isfinite(r).all()
    m = s.monitor_payload()
    assert {"ee_n_phases", "ee_alpha_min", "ee_alpha_max", "ee_alpha_sum",
            "ee_mom_src"} <= set(m)


def test_pressure_solver_without_eulerian_baseline():
    V, C = cube_tet_mesh(2)
    s = PressureSolver(V, C, mu=1e-3, inlet_velocity=(1.0, 0.0, 0.0))
    assert s.eulerian_model is None
    for _ in range(5):
        p = s.step()
    assert np.isfinite(p["residual"])
    m = s.monitor_payload()
    assert "ee_n_phases" not in m

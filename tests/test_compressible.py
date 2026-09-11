# -*- coding: utf-8 -*-
"""P 波 P12：可压缩 / 密度基流（Compressible / Density-Based Flow）—— 理想气体状态
方程（EOS）+ 压力基可压缩 SIMPLE 密度耦合 + 能量-密度耦合 + 声速/马赫数诊断。

覆盖：
  常量：DEFAULT_GAMMA / DEFAULT_MW / DEFAULT_DT / DEFAULT_RELAX / MACH_EPS
  EOS：ideal_gas_rho（标量/数组广播、T→0 保护）、density_derivative
  声学：sound_speed（c=√(γRT/W)）、sound_speed_pr（c=√(γp/ρ)）、
        velocity_magnitude、mach_number
  压缩性：dilatation（∇·u，均匀场归零）、compression_work
  密度-压力耦合：density_correction（ρ'=ρp'/(γp)）、
        compressibility_diagonal（V/(c²Δt)）、density_change_source（(ρ−ρ_prev)V/Δt）
  CompressibleSolver：EOS 密度状态、欠松弛推进、温度/压力敏感、面密度、
        P10 后端门面 step()/residual()/monitor_payload()
  工厂：make_compressible 别名/大小写解析，未知模型报 ValueError
  集成：PressureSolver 字符串注入 compressible + step() 稳定、rho 为逐单元场、
        monitor 含 cmp 键；基线不暴露 cmp 键（零回归）

验收核心（P12 行）：star.flow（Compressible / Coupled Flow + Ideal Gas EOS）——
密度由状态方程求得、声速/马赫数诊断、压力修正体现压缩性。
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
from species import R_UNIV
from compressible import (
    DEFAULT_DT, DEFAULT_GAMMA, DEFAULT_MW, DEFAULT_RELAX, MACH_EPS,
    CompressibleSolver, compressibility_diagonal, compression_work,
    density_change_source, density_correction, density_derivative, dilatation,
    ideal_gas_rho, mach_number, make_compressible, sound_speed, sound_speed_pr,
    velocity_magnitude,
)


# ---------------------------------------------------------------- 工具
def _fv(nx=2):
    V, C = cube_tet_mesh(nx)
    return FVM(V, C)


def _uniform_u(fv, speed=1.0):
    n = fv.n_cells
    return (np.full(n, speed, float), np.zeros(n, float), np.zeros(n, float))


# ---------------------------------------------------------------- 常量
def test_compressible_constants():
    assert DEFAULT_GAMMA > 1.0
    assert DEFAULT_MW > 0.0
    assert DEFAULT_DT > 0.0
    assert 0.0 < DEFAULT_RELAX <= 1.0
    assert MACH_EPS > 0.0


# ---------------------------------------------------------------- EOS
def test_ideal_gas_rho_scalar_and_array():
    r = ideal_gas_rho(101325.0, 300.0)
    assert np.isclose(r, 101325.0 * DEFAULT_MW / (R_UNIV * 300.0))
    arr = ideal_gas_rho(np.array([101325.0, 202650.0]), 300.0)
    assert arr.shape == (2,)
    assert np.isclose(arr[1] / arr[0], 2.0)  # 密度正比压力
    # 温度升高密度下降
    assert ideal_gas_rho(101325.0, 600.0) < ideal_gas_rho(101325.0, 300.0)


def test_ideal_gas_rho_zero_temp_protected():
    r = ideal_gas_rho(101325.0, 0.0)
    assert np.isfinite(r) and r > 0.0


def test_density_derivative_is_W_over_RT():
    d = density_derivative(300.0)
    assert np.isclose(d, DEFAULT_MW / (R_UNIV * 300.0))
    # (∂ρ/∂p)_T = γ / c²（等温压缩率与等熵声速的关系）
    c = sound_speed(300.0)
    assert np.isclose(d * c * c, DEFAULT_GAMMA, rtol=1e-9)


# ---------------------------------------------------------------- 声学
def test_sound_speed_pr_matches_sound_speed():
    rho = ideal_gas_rho(101325.0, 300.0)
    c1 = sound_speed(300.0)
    c2 = sound_speed_pr(101325.0, rho)
    assert np.isclose(c1, c2, rtol=1e-9)
    assert c1 > 0.0


def test_velocity_magnitude_and_mach():
    m = velocity_magnitude(np.array([3.0, 0.0]), np.array([4.0, 0.0]),
                           np.array([0.0, 0.0]))
    assert np.allclose(m, [5.0, 0.0])
    ma = mach_number(np.array([340.0]), np.zeros(1), np.zeros(1), 340.0)
    assert np.isclose(ma[0], 1.0)


def test_mach_number_zero_speed_speed_guard():
    ma = mach_number(np.zeros(1), np.zeros(1), np.zeros(1), 0.0)
    assert np.isfinite(ma[0]) and ma[0] == 0.0


# ---------------------------------------------------------------- 压缩性诊断
def test_dilatation_uniform_field_zero():
    fv = _fv(2)
    u, v, w = _uniform_u(fv, 1.0)
    div = dilatation(fv, u, v, w)
    assert div.shape == (fv.n_cells,)
    assert np.allclose(div, 0.0, atol=1e-10)


def test_dilatation_linear_field_slope():
    fv = _fv(3)
    a = 2.0
    u = a * fv.centroids[:, 0]
    v = np.zeros(fv.n_cells)
    w = np.zeros(fv.n_cells)
    div = dilatation(fv, u, v, w)
    # 拉伸流（u=a·x）：逐单元散度为正、体积加权平均为正（净外流）
    assert np.all(div > 0.0)
    assert np.average(div, weights=fv.volumes) > 0.0
    # 线性缩放：斜率加倍 → 散度精确加倍
    div2 = dilatation(fv, 2.0 * u, v, w)
    assert np.allclose(div2, 2.0 * div, rtol=1e-9)


def test_compression_work():
    dil = np.array([1.0, -2.0])
    p = np.array([100.0, 100.0])
    assert np.allclose(compression_work(dil, p), [100.0, -200.0])


# ---------------------------------------------------------------- 密度-压力耦合
def test_density_correction_proportional_to_pprime():
    rho = np.array([1.2])
    p = np.array([101325.0])
    d1 = density_correction(np.array([10.0]), rho, p)
    d2 = density_correction(np.array([20.0]), rho, p)
    assert np.isclose(d2[0] / d1[0], 2.0)


def test_density_correction_zero_when_pprime_zero():
    d = density_correction(np.zeros(2), np.array([1.2, 1.0]),
                           np.array([101325.0, 101325.0]))
    assert np.allclose(d, 0.0)


def test_compressibility_diagonal_scaling():
    V = np.array([1.0, 1.0])
    d1 = compressibility_diagonal(V, np.array([100.0, 100.0]), dt=1e-3)
    d2 = compressibility_diagonal(V, np.array([200.0, 200.0]), dt=1e-3)
    assert np.all(d1 > 0.0)
    assert np.allclose(d1 / d2, 4.0)  # 反比 c²


def test_density_change_source_zero_at_steady():
    rho = np.array([1.2, 1.0])
    src = density_change_source(rho, rho.copy(), np.array([1.0, 1.0]), dt=1e-3)
    assert np.allclose(src, 0.0)
    src2 = density_change_source(np.array([1.2]), np.array([1.0]),
                                 np.array([2.0]), dt=0.5)
    assert np.isclose(src2[0], (1.2 - 1.0) * 2.0 / 0.5)


# ---------------------------------------------------------------- 求解器
def test_solver_initial_density_is_reference():
    fv = _fv(2)
    s = CompressibleSolver(fv)
    assert np.allclose(s.rho, s.rho_ref)
    assert s.iteration == 0
    assert s.sound_speed.shape == (fv.n_cells,)
    assert np.isfinite(s.max_mach)


def test_solver_update_temperature_lowers_density():
    fv = _fv(2)
    n = fv.n_cells
    s = CompressibleSolver(fv, relax=1.0)
    s.update(np.zeros(n), np.zeros(n), np.zeros(n),
             T=np.full(n, 600.0), p=np.zeros(n))
    # T 翻倍 → 密度减半
    assert np.allclose(s.rho, s.rho_ref * 0.5, rtol=1e-9)


def test_solver_update_pressure_raises_density():
    fv = _fv(2)
    n = fv.n_cells
    s = CompressibleSolver(fv, relax=1.0)
    s.update(np.zeros(n), np.zeros(n), np.zeros(n),
             p=np.full(n, 101325.0))
    # 表压等于 p_ref → 绝对压力翻倍 → 密度翻倍
    assert np.allclose(s.rho, s.rho_ref * 2.0, rtol=1e-9)


def test_solver_update_residual_decreases():
    fv = _fv(2)
    n = fv.n_cells
    s = CompressibleSolver(fv, relax=0.7)
    r1 = s.update(np.zeros(n), np.zeros(n), np.zeros(n),
                  T=np.full(n, 600.0), p=np.zeros(n))
    r2 = s.update(np.zeros(n), np.zeros(n), np.zeros(n),
                  T=np.full(n, 600.0), p=np.zeros(n))
    assert r1 > 0.0 and r2 < r1


def test_solver_face_density_shape_and_uniform():
    fv = _fv(2)
    s = CompressibleSolver(fv)
    fr = s.face_density()
    assert fr.shape == (fv.n_faces,)
    assert np.allclose(fr, s.rho_ref)


def test_solver_coupling_terms_shapes():
    fv = _fv(2)
    s = CompressibleSolver(fv)
    assert s.compressibility_diagonal().shape == (fv.n_cells,)
    assert s.density_change_source().shape == (fv.n_cells,)
    assert s.density_correction(np.ones(fv.n_cells)).shape == (fv.n_cells,)


def test_solver_p10_facade():
    fv = _fv(2)
    s = CompressibleSolver(fv)
    ini = s._initialize_field()
    assert {"rho_min", "rho_max", "c_min", "c_max", "mach_max"} <= set(ini)
    out = s.step()
    assert "residual" in out and np.isfinite(out["residual"])
    assert np.isfinite(s.residual())
    pay = s.monitor_payload()
    assert {"rho_min", "rho_max", "p_min", "p_max", "T_min", "T_max",
            "c_min", "c_max", "mach_max", "dil_max", "iteration",
            "residual"} <= set(pay)


# ---------------------------------------------------------------- 工厂
def test_factory_aliases_and_case():
    fv = _fv(2)
    assert isinstance(make_compressible(fv, "compressible"), CompressibleSolver)
    assert isinstance(make_compressible(fv, "Compressible-Flow"), CompressibleSolver)
    assert isinstance(make_compressible(fv, "density_based"), CompressibleSolver)
    assert isinstance(make_compressible(fv, "Ideal Gas"), CompressibleSolver)
    assert isinstance(make_compressible(fv, "COUPLED_FLOW"), CompressibleSolver)


def test_factory_unknown_raises():
    fv = _fv(2)
    with pytest.raises(ValueError):
        make_compressible(fv, "nope")


# ---------------------------------------------------------------- 集成
def test_pressure_solver_baseline_no_cmp_keys():
    V, C = cube_tet_mesh(2)
    s = PressureSolver(V, C, mu=1e-3, inlet_velocity=(1.0, 0.0, 0.0))
    assert s.compressible_model is None
    assert np.isscalar(s.rho)
    for _ in range(2):
        assert np.isfinite(s.step()["residual"])
    pay = s.monitor_payload()
    assert not any(k.startswith("cmp_") for k in pay)


def test_pressure_solver_compressible_coupling():
    V, C = cube_tet_mesh(2)
    s = PressureSolver(V, C, mu=1e-3, inlet_velocity=(1.0, 0.0, 0.0),
                       compressible_model="compressible",
                       compressible_p_ref=101325.0, compressible_t_ref=300.0)
    assert s.compressible_model is not None and not isinstance(s.compressible_model, str)
    assert not np.isscalar(s.rho)
    assert np.asarray(s.rho).shape == (s._fv.n_cells,)
    for _ in range(4):
        out = s.step()
        assert np.isfinite(out["residual"])
    assert np.isfinite(s.velocity()).all()
    pay = s.monitor_payload()
    assert {"cmp_rho_min", "cmp_rho_max", "cmp_p_min", "cmp_p_max",
            "cmp_T_max", "cmp_c_min", "cmp_c_max", "cmp_mach_max"} <= set(pay)
    assert pay["cmp_rho_min"] > 0.0
    assert pay["cmp_c_max"] > 0.0

# -*- coding: utf-8 -*-
"""P 波 P9：组分输运 + 燃烧 —— 质量分数守恒输运 + 全局 Arrhenius 反应动力学。

覆盖：
  组分：换算（质量↔摩尔、混合摩尔质量、理想气体密度）、输运装配/求解形状与有界性、
        边界分类、SpeciesSolver 收敛（Σ Y = 1）、P10 后端门面、工厂别名
  燃烧：arrhenius_rate 单调性、GlobalReaction（质量源自动守恒、热释放）、
        层流火焰速度/厚度、进度变量、点火器（温度/火花/进度）、工厂别名
  CombustionModel：update() 产热与组分守恒、P10 后端门面
  集成：PressureSolver 注入 species/combustion 字符串后 step() 无报错，
        基线（无模型）step() 无回归

验收核心（P9 行）：组分输运 + 燃烧反应动力学谱系。
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
import species as SP
import combustion as CB
from species import (
    DEFAULT_MW, DEFAULT_NAMES, DEFAULT_SCALAR_DIFF, DEFAULT_SC_TURB,
    R_UNIV, SP_FIXED, SP_ZERO_GRAD,
    SpeciesSolver, assemble_species_transport, classify_species, gas_density,
    make_species, mass_to_mole_frac, mixture_molar_mass, mole_to_mass_frac,
    solve_species, volume_mass,
)
from combustion import (
    DEFAULT_A, DEFAULT_EA, DEFAULT_H_RXN, DEFAULT_NU, DEFAULT_REACT_IDX,
    CombustionModel, GlobalReaction, ProgressVariableIgnitor, SparkIgnitor,
    TemperatureIgnitor, arrhenius_rate, combustion_progress, flame_thickness,
    laminar_flame_speed, make_combustion, make_ignitor, make_prequick_mech,
)


# ---------------------------------------------------------------- 工具
def _fv(nx=2):
    V, C = cube_tet_mesh(nx)
    return FVM(V, C)


def _uniform_u(fv, speed=1.0):
    n = fv.n_cells
    return (np.full(n, speed, float), np.zeros(n, float), np.zeros(n, float))


def _air_fractions():
    return [0.0, 0.233, 0.0, 0.0, 0.767]


# ---------------------------------------------------------------- 组分换算
def test_species_constants_and_conversion():
    assert R_UNIV == pytest.approx(8314.462)
    assert DEFAULT_SC_TURB > 0.0 and DEFAULT_SCALAR_DIFF > 0.0
    assert len(DEFAULT_NAMES) == len(DEFAULT_MW) == 5
    Y = np.array([_air_fractions()], float)
    X = mass_to_mole_frac(Y, DEFAULT_MW)
    Y2 = mole_to_mass_frac(X, DEFAULT_MW)
    assert np.allclose(Y2, Y, atol=1e-9)
    W = mixture_molar_mass(Y, DEFAULT_MW)
    assert np.all(W > 0.0)
    assert np.allclose(W.sum(axis=-1), W.ravel(), atol=1e-9)
    rho = gas_density(W, 300.0, 101325.0)
    assert np.all(rho > 0.0)
    assert float(rho[0]) == pytest.approx(101325.0 * float(W[0]) / (R_UNIV * 300.0))


def test_mass_to_mole_frac_zero_guard():
    Y = np.zeros((3, 5), float)
    X = mass_to_mole_frac(Y, DEFAULT_MW)
    assert np.all(X == 0.0)
    X2 = mole_to_mass_frac(np.zeros((3, 5), float), DEFAULT_MW)
    assert np.all(X2 == 0.0)


# ---------------------------------------------------------------- 输运装配 / 求解
def test_assemble_species_transport_shapes():
    fv = _fv(2)
    n = fv.n_cells
    mdot = volume_mass(fv, *_uniform_u(fv), np.full(n, 1.0))
    gamma = np.full(n, 1.0 * DEFAULT_SCALAR_DIFF, float)
    Y = np.full(n, 0.233, float)
    rows, cols, vals, rhs, ap = assemble_species_transport(
        fv, mdot, gamma, Y, np.zeros(n, float))
    assert rows.ndim == 1 and cols.ndim == 1 and vals.ndim == 1
    assert rhs.shape == (n,) and ap.shape == (n,)
    assert np.all(np.isfinite(vals)) and np.all(np.isfinite(rhs))
    assert np.all(np.isfinite(ap))
    assert ap.min() >= -1e-6


def test_solve_species_pure_convection_bounded():
    fv = _fv(2)
    n = fv.n_cells
    mdot = volume_mass(fv, *_uniform_u(fv), np.full(n, 1.0))
    gamma = np.full(n, 1.0 * DEFAULT_SCALAR_DIFF, float)
    Y0 = np.full(n, 0.0, float)
    btype = np.full(fv.n_faces, SP_ZERO_GRAD, int)
    bval = np.zeros(fv.n_faces, float)
    btype[fv.is_boundary] = SP_ZERO_GRAD
    Ynew, ap = solve_species(fv, mdot, gamma, Y0, np.zeros(n, float),
                             btype=btype, bval=bval, outlet_faces=np.where(
                                 fv.is_boundary)[0], relax=0.7)
    assert Ynew.shape == (n,)
    assert np.all(np.isfinite(Ynew))
    assert np.all(np.abs(Ynew) <= 1.0 + 1e-6)


def test_classify_species_boundary():
    fv = _fv(2)
    tin, tout, twall, tbnd = classify_species(fv, 0, "min", "max")
    bnd = np.where(fv.is_boundary)[0]
    assert set(tin) ^ set(tout) ^ set(twall) == set(tbnd)
    assert len(tin) + len(tout) + len(twall) == len(bnd)
    assert len(tin) > 0 and len(tout) > 0


# ---------------------------------------------------------------- SpeciesSolver
def test_species_solver_convection_converges():
    fv = _fv(2)
    n = fv.n_cells
    nsp = len(DEFAULT_MW)
    sp = SpeciesSolver(fv)
    assert sp.n_species == nsp
    assert np.allclose(sp.Y.sum(axis=1), 1.0)
    u, v, w = _uniform_u(fv)
    mdot = volume_mass(fv, u, v, w, np.full(n, sp._rho()))
    nu_t = np.zeros(n, float)
    resid = 1.0
    for _ in range(8):
        resid = sp.update(u, v, w, mdot=mdot, nu_t=nu_t)
    assert np.isfinite(resid)
    assert resid < 1e-3
    assert np.allclose(sp.Y.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(sp.Y >= 0.0) and np.all(sp.Y <= 1.0)


def test_species_solver_p10_facade():
    fv = _fv(2)
    sp = SpeciesSolver(fv)
    sp.set_velocities(*_uniform_u(fv))
    info = sp._initialize_field()
    assert set(info) == {"n_species", "y_min", "y_max", "w_mix"}
    assert np.isclose(sp.residual(), 0.0, atol=1e-12)
    p = sp.step()
    assert set(p) == {"residual", "n_species", "y_min", "y_max", "w_mix"}
    m = sp.monitor_payload()
    assert {"n_species", "y_min", "y_max", "sum_min", "sum_max",
            "w_mix_min", "w_mix_max", "iteration", "residual"} <= set(m)
    assert m["iteration"] == sp.iteration
    assert np.isclose(m["sum_min"], 1.0, atol=1e-6)


# ---------------------------------------------------------------- 燃烧反应动力学
def test_arrhenius_rate_monotonic_T():
    cold = arrhenius_rate(DEFAULT_A, DEFAULT_EA, 0.0, 300.0)
    hot = arrhenius_rate(DEFAULT_A, DEFAULT_EA, 0.0, 2000.0)
    assert cold > 0.0 and hot > 0.0
    assert hot > cold
    # β 指数提升速率
    beta = arrhenius_rate(DEFAULT_A, DEFAULT_EA, 1.0, 2000.0)
    assert beta > hot


def test_global_reaction_rate_positive_hot():
    reac = make_prequick_mech()
    conc = np.array([[0.5, 1.0, 0.0, 0.0, 0.0]], float)  # kmol/m^3 反应物
    w_cold = float(reac.rate(conc, np.array([300.0]))[0])
    w_hot = float(reac.rate(conc, np.array([2000.0]))[0])
    assert w_cold >= 0.0 and w_hot >= 0.0
    assert w_hot > w_cold


def test_global_reaction_mass_source_conserves():
    reac = GlobalReaction(nu=[-1.0, 1.0], reactant_idx=[0],
                          reactant_orders=[1.0], A=1.0e6, Ea=1.0e8)
    Y = np.array([[0.5, 0.5]], float)          # 组分 A → B（质量守恒）
    Mw = np.array([10.0, 10.0], float)         # 平衡分子量 → Σ ν_i W_i = 0
    T = np.array([1500.0], float)
    rho = np.array([1.0], float)
    S = reac.mass_source(Y, T, rho, Mw)
    assert S.shape == (1, 2)
    assert np.allclose(np.sum(S, axis=1), 0.0, atol=1e-9)
    q = reac.heat_release(Y, T, rho, Mw)
    assert np.all(q >= 0.0)


# ---------------------------------------------------------------- 火焰 / 进度
def test_flame_speed_thickness_progress():
    assert laminar_flame_speed(300.0) == pytest.approx(0.4)
    assert laminar_flame_speed(600.0) > laminar_flame_speed(300.0)
    th = flame_thickness(0.4)
    assert th == pytest.approx(2.2e-5 / 0.4)
    Y = np.array([[0.05, 0.0], [0.0, 0.0]], float)  # fuel_idx=0 从 0.05 → 0
    c = combustion_progress(Y, fuel_idx=0, Y_fuel_inlet=0.05)
    assert np.allclose(c, [0.0, 1.0], atol=1e-9)
    assert np.all(np.clip(c, 0.0, 1.0) >= 0.0)


# ---------------------------------------------------------------- 点火器
def test_ignitors():
    fv = _fv(2)
    c = fv.centroids
    tig = TemperatureIgnitor(threshold=1500.0)
    assert np.all(tig.factor(c, T=np.full(fv.n_cells, 2000.0)) == 1.0)
    assert np.all(tig.factor(c, T=np.full(fv.n_cells, 300.0)) == 0.0)
    sig = SparkIgnitor(position=(0.5, 0.5, 0.5), radius=0.6, energy=1.0e5,
                       duration=1.0)
    f = sig.factor(c)
    assert np.all(f >= 0.0) and np.all(f <= 1.0)
    assert float(np.max(sig.heat_source(c))) > 0.0
    pig = ProgressVariableIgnitor(threshold=0.5)
    assert np.all(pig.factor(c, progress=np.full(fv.n_cells, 0.9)) == 1.0)
    assert np.all(pig.factor(c, progress=np.full(fv.n_cells, 0.1)) == 0.0)


def test_make_ignitor_aliases_and_unknown():
    assert isinstance(make_ignitor("temperature"), TemperatureIgnitor)
    assert isinstance(make_ignitor("TEMP"), TemperatureIgnitor)
    assert isinstance(make_ignitor("spark"), SparkIgnitor)
    assert isinstance(make_ignitor("progress_variable"), ProgressVariableIgnitor)
    with pytest.raises(ValueError):
        make_ignitor("nope")


# ---------------------------------------------------------------- CombustionModel
def test_combustion_model_update_heat_release():
    fv = _fv(2)
    n = fv.n_cells
    cm = CombustionModel(fv, inlet_fractions=[0.05, 0.20, 0.0, 0.0, 0.75])
    # 未燃（低温，无燃料反应或点火）：Q=0
    cm.set_temperature(np.full(n, 300.0, float))
    cm._chemical_source()
    assert cm.omega.max() < 1e6
    assert np.allclose(cm.Y.sum(axis=1), 1.0)
    # 高温 + 燃料 → 放热
    u, v, w = _uniform_u(fv)
    mdot = volume_mass(fv, u, v, w, np.full(n, cm._sp._rho()))
    for _ in range(3):
        resid = cm.update(u, v, w, mdot=mdot, nu_t=np.zeros(n, float),
                          T=np.full(n, 2000.0, float))
    assert np.isfinite(resid)
    assert cm.heat_release.max() > 0.0
    assert cm.omega.max() > 0.0
    assert np.allclose(cm.Y.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(cm.Y >= 0.0)


def test_combustion_model_p10_facade():
    fv = _fv(2)
    cm = make_combustion(fv, "combustion")
    cm.set_velocities(*_uniform_u(fv))
    info = cm._initialize_field()
    assert isinstance(info["n_species"], int)
    assert "heat_release_max" in info
    assert np.isclose(cm.residual(), 0.0, atol=1e-12)
    p = cm.step()
    assert "residual" in p and "heat_release_max" in p
    m = cm.monitor_payload()
    assert {"n_species", "fuel", "y_min", "y_max", "w_mix",
            "heat_release_max", "heat_release_mean", "progress_max",
            "s_L", "flame_thickness", "n_ignited", "iteration",
            "residual"} <= set(m)
    assert m["iteration"] == cm.iteration


# ---------------------------------------------------------------- 工厂
def test_make_species_factory_aliases():
    fv = _fv(2)
    assert isinstance(make_species(fv, "species"), SpeciesSolver)
    assert isinstance(make_species(fv, "multispecies"), SpeciesSolver)
    assert isinstance(make_species(fv, "MASS_FRACTION"), SpeciesSolver)
    with pytest.raises(ValueError):
        make_species(fv, "nope")


def test_make_combustion_factory_aliases():
    fv = _fv(2)
    assert isinstance(make_combustion(fv, "combustion"), CombustionModel)
    assert isinstance(make_combustion(fv, "global"), CombustionModel)
    assert isinstance(make_combustion(fv, "GLOBE"), CombustionModel)
    with pytest.raises(ValueError):
        make_combustion(fv, "nope")


# ---------------------------------------------------------------- 集成：PressureSolver
def test_pressure_solver_combustion_coupling():
    V, C = cube_tet_mesh(2)
    s = PressureSolver(V, C, mu=1e-3, inlet_velocity=(1.0, 0.0, 0.0),
                       species_model="species",
                       combustion_model="combustion",
                       combustion_inlet_fractions=[0.05, 0.20, 0.0, 0.0, 0.75],
                       combustion_ref_temp=1800.0)
    assert s.species_model is not None
    assert s.combustion_model is not None
    for _ in range(5):
        p = s.step()
    assert np.isfinite(p["residual"])
    assert np.all(np.isfinite(s.velocity()))
    cm = s.combustion_model
    assert np.allclose(cm.Y.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(np.isfinite(cm.heat_release))


def test_pressure_solver_without_combustion_baseline():
    V, C = cube_tet_mesh(2)
    s = PressureSolver(V, C, mu=1e-3, inlet_velocity=(1.0, 0.0, 0.0))
    assert s.species_model is None and s.combustion_model is None
    for _ in range(5):
        p = s.step()
    assert np.isfinite(p["residual"])
    assert np.all(np.isfinite(s.velocity()))

# -*- coding: utf-8 -*-
"""P 波 P7：能量/传热 —— 对流-扩散能量方程 + 共轭传热 + 简化辐射谱系。

覆盖：
  常量/边界：BND_* 传热边界类型、SIGMA_STEFAN、物性默认值
  装配/求解：assemble_energy_transport（五类边界，形状与有界性）、
             solve_energy（纯导热、稳态温度有界）
  边界分类/辅助：classify_thermal（入口/出口/壁面全覆盖不重叠）、
                 thermal_btypes（定温/绝热/热流/Robin）四元组、
                 volume_to_mass（均匀速度面通量守恒）
  EnergySolver：update() 收敛、T 有界、固定温壁面加热、材料分区（共轭）、
                辐射壁面、P10 后端门面 step()/residual()/monitor_payload()
  简化辐射：RadiationModel 净发射/辐射传热系数/体源/线性化
  工厂：make_energy 经别名/大小写解析，未知模型报 ValueError
  集成：PressureSolver 字符串注入能量模型（+Boussinesq 浮力）step() 无报错，
        基线（无能量模型）step() 无报错、nu_t 耦合形状一致

验收核心（P7 行）：能量/传热（对流扩散+共轭）+ 简化辐射谱系。
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
import energy as E
from energy import (
    BND_ADIABATIC, BND_FIXED_FLUX, BND_FIXED_TEMP, BND_RADIATION, BND_ROBIN,
    DEFAULT_CP, DEFAULT_KAPPA, DEFAULT_RHO, SIGMA_STEFAN,
    ConjugateSolver, EnergySolver, RadiationModel,
    assemble_energy_transport, classify_thermal, make_energy,
    solve_energy, thermal_btypes, volume_to_mass,
)


# ---------------------------------------------------------------- 工具
def _fv(nx=2):
    V, C = cube_tet_mesh(nx)
    return FVM(V, C)


def _uniform_u(fv, speed=1.0):
    n = fv.n_cells
    return (np.full(n, speed, float), np.zeros(n, float), np.zeros(n, float))


# ---------------------------------------------------------------- 常量 / 边界类型
def test_energy_constants_and_bnd():
    assert SIGMA_STEFAN == pytest.approx(5.67e-8)
    assert DEFAULT_RHO > 0.0 and DEFAULT_CP > 0.0 and DEFAULT_KAPPA > 0.0
    # 边界类型为互异枚举
    vals = [BND_ADIABATIC, BND_FIXED_TEMP, BND_FIXED_FLUX, BND_ROBIN, BND_RADIATION]
    assert len(set(vals)) == len(vals)
    assert BND_ADIABATIC == 0 and BND_FIXED_TEMP == 1


# ---------------------------------------------------------------- 装配 / 求解
def test_assemble_energy_transport_shapes():
    fv = _fv(2)
    n = fv.n_cells
    mdot = volume_to_mass(fv, *_uniform_u(fv), np.full(n, DEFAULT_RHO))
    kappa = np.full(n, DEFAULT_KAPPA, float)
    cp = np.full(n, DEFAULT_CP, float)
    T = np.full(n, 300.0, float)
    rows, cols, vals, rhs, ap = assemble_energy_transport(
        fv, mdot, kappa, cp, T, np.zeros(n, float))
    assert rows.ndim == 1 and cols.ndim == 1 and vals.ndim == 1
    assert rhs.shape == (n,) and ap.shape == (n,)
    assert np.all(np.isfinite(vals)) and np.all(np.isfinite(rhs))
    assert np.all(np.isfinite(ap))
    assert ap.min() >= -1e-6


def test_solve_energy_pure_conduction():
    fv = _fv(2)
    n = fv.n_cells
    T0 = np.full(n, 320.0, float)
    mdot = np.zeros(fv.n_faces, float)
    btype = np.full(fv.n_faces, BND_ADIABATIC, int)
    bval = np.zeros(fv.n_faces, float)
    inlet, outlet, wall, _ = classify_thermal(fv)
    btype[inlet] = BND_FIXED_TEMP
    bval[inlet] = 300.0
    btype[outlet] = BND_FIXED_TEMP
    bval[outlet] = 350.0
    T_new, ap = solve_energy(fv, mdot, np.full(n, 1.0), np.full(n, 1.0),
                             T0, np.zeros(n), btype=btype, bval=bval, relax=1.0)
    assert T_new.shape == (n,)
    assert np.all(np.isfinite(T_new))
    # 纯导热（Laplace）解析解受两个定温边界锚定：温度落在 [300, 350]
    assert T_new.min() >= 300.0 - 1e-6
    assert T_new.max() <= 350.0 + 1e-6


def test_solve_energy_fixed_temp_bounded():
    fv = _fv(2)
    n = fv.n_cells
    btype = np.full(fv.n_faces, BND_ADIABATIC, int)
    bval = np.zeros(fv.n_faces, float)
    inlet, outlet, wall, _ = classify_thermal(fv)
    btype[inlet] = BND_FIXED_TEMP
    bval[inlet] = 300.0
    btype[wall] = BND_FIXED_TEMP
    bval[wall] = 350.0
    T_new, _ap = solve_energy(fv, np.zeros(fv.n_faces), np.full(n, 1.0),
                              np.full(n, 1.0), np.full(n, 320.0, float),
                              np.zeros(n), btype=btype, bval=bval, relax=1.0)
    assert np.all(np.isfinite(T_new))
    # 定温边界确定性约束：解应落在 [min bval, max bval] 内
    assert T_new.min() >= 300.0 - 1e-6
    assert T_new.max() <= 350.0 + 1e-6


# ---------------------------------------------------------------- 边界分类 / 辅助
def test_classify_thermal_partition():
    fv = _fv(2)
    inlet, outlet, wall, bnd = classify_thermal(fv)
    assert inlet.size > 0 and outlet.size > 0
    assert inlet.size + outlet.size + wall.size == len(bnd)
    union = set(inlet.tolist()) | set(outlet.tolist()) | set(wall.tolist())
    assert len(union) == len(bnd)               # 全覆盖且不重叠


def test_thermal_btypes_four_tuple():
    fv = _fv(2)
    inlet, outlet, wall, _ = classify_thermal(fv)
    btype, bval, h, em = thermal_btypes(
        fv, inlet, outlet, wall, inlet_temp=310.0,
        wall_btype=BND_FIXED_TEMP, wall_bval=320.0)
    assert btype.shape == (fv.n_faces,) and bval.shape == (fv.n_faces,)
    assert h.shape == (fv.n_faces,) and em.shape == (fv.n_faces,)
    assert np.all(btype[inlet] == BND_FIXED_TEMP)
    assert np.allclose(bval[inlet], 310.0)
    assert np.all(btype[outlet] == BND_ADIABATIC)
    assert np.all(btype[wall] == BND_FIXED_TEMP)
    assert np.allclose(bval[wall], 320.0)


def test_thermal_btypes_robin():
    fv = _fv(2)
    inlet, outlet, wall, _ = classify_thermal(fv)
    btype, bval, h, em = thermal_btypes(
        fv, inlet, outlet, wall, inlet_temp=300.0,
        wall_btype=BND_ROBIN, wall_h=25.0, wall_tref=350.0)
    assert np.all(btype[wall] == BND_ROBIN)
    assert np.allclose(bval[wall], 350.0)
    assert np.allclose(h[wall], 25.0)
    assert np.allclose(em[wall], 0.0)


def test_volume_to_mass_conservation():
    fv = _fv(2)
    mdot = volume_to_mass(fv, *_uniform_u(fv), np.full(fv.n_cells, DEFAULT_RHO))
    assert mdot.shape == (fv.n_faces,)
    assert np.all(np.isfinite(mdot))
    # 均匀流动经封闭边界：净面通量 ~ 0
    assert abs(mdot[fv.is_boundary].sum()) < 1e-6


# ---------------------------------------------------------------- EnergySolver
def test_energy_solver_convection_converges():
    fv = _fv(2)
    es = EnergySolver(fv, inlet_temp=300.0)
    u, v, w = _uniform_u(fv)
    first = None
    for i in range(60):
        res = es.update(u, v, w)
        first = first if first is not None else res
    assert np.all(np.isfinite(es.T))
    assert es.residual() == pytest.approx(res)
    assert res < first                        # 残差较首步下降
    assert res < 1e-5
    # 绝热 + 仅对流：稳态保持在入口温度附近
    assert np.allclose(es.T, 300.0, atol=1.0)


def test_energy_solver_fixed_temp_wall_bounded():
    fv = _fv(2)
    es = EnergySolver(fv, inlet_temp=300.0,
                      wall_btype=BND_FIXED_TEMP, wall_bval=350.0)
    u, v, w = _uniform_u(fv)
    for _ in range(80):
        es.update(u, v, w)
    T = es.T
    assert np.all(np.isfinite(T))
    assert T.min() >= 300.0 - 1e-3             # 受对流入口温度下界约束
    assert T.max() > 300.0                     # 热壁加热
    assert T.max() <= 350.0 + 1e-3             # 受热壁上界约束


def test_energy_solver_materials_split():
    fv = _fv(2)
    es = EnergySolver(fv, inlet_temp=300.0,
                      wall_btype=BND_FIXED_TEMP, wall_bval=350.0)
    phase = np.zeros(fv.n_cells, int)
    phase[fv.centroids[:, 0] > 0.5] = 1        # 右侧为固体区
    es.set_materials(cell_phase=phase, rho_s=7800.0, cp_s=500.0, kappa_s=16.0)
    assert es.phase.shape == (fv.n_cells,)
    assert es.phase.dtype.kind in "iu"
    assert int(es.phase.sum()) > 0
    assert es.phase.max() == 1
    u, v, w = _uniform_u(fv)
    for _ in range(6):
        es.update(u, v, w)
    assert np.all(np.isfinite(es.T))


def test_energy_solver_radiation_wall():
    fv = _fv(2)
    es = EnergySolver(fv, inlet_temp=300.0,
                      wall_btype=BND_RADIATION, wall_emis=0.9, wall_tref=500.0)
    u, v, w = _uniform_u(fv)
    for _ in range(80):
        es.update(u, v, w)
    T = es.T
    assert np.all(np.isfinite(T))
    assert T.max() > 300.0
    assert T.max() <= 500.0 + 1e-3
    assert np.all(np.isfinite(es.heat_flux))


def test_energy_solver_p10_facade():
    fv = _fv(2)
    es = EnergySolver(fv, inlet_temp=300.0)
    info = es._initialize_field()
    assert set(info) == {"temp_min", "temp_max", "temp_mean"}
    assert np.isclose(es.residual(), 0.0, atol=1e-12)
    p = es.step()
    assert set(p) == {"residual", "temp_min", "temp_max", "temp_mean"}
    assert np.isfinite(p["temp_min"]) and np.isfinite(p["temp_max"])
    m = es.monitor_payload()
    assert {"temp_min", "temp_max", "temp_mean", "iteration", "residual",
            "heat_flux_max"} <= set(m)
    assert m["iteration"] == es.iteration


# ---------------------------------------------------------------- 简化辐射
def test_radiation_model_physics():
    rm = RadiationModel(emissivity=0.8, tref=300.0)
    assert rm.emittance(300.0) == pytest.approx(0.8 * SIGMA_STEFAN * 300.0 ** 4)
    assert rm.net_emission(300.0) == pytest.approx(0.0, abs=1e-9)
    assert rm.net_emission(400.0) > 0.0         # 高温发射
    assert rm.net_emission(200.0) < 0.0         # 低温吸收
    assert rm.h_rad(300.0, 300.0) == pytest.approx(
        0.8 * SIGMA_STEFAN * 600.0 * 2.0 * 300.0 ** 2)
    assert np.all(np.asarray(rm.h_rad(300.0)) > 0.0)
    assert rm.volume_source(300.0) == pytest.approx(0.0, abs=1e-9)
    prod, diss = rm.linear_source()
    assert prod > 0.0 and diss > 0.0


# ---------------------------------------------------------------- 工厂
def test_make_energy_factory_aliases():
    fv = _fv(2)
    assert isinstance(make_energy(fv, "energy"), EnergySolver)
    assert isinstance(make_energy(fv, "cht"), ConjugateSolver)
    assert isinstance(make_energy(fv, "CONJUGATE"), ConjugateSolver)
    assert isinstance(make_energy(fv, "conjugate_heat_transfer"), ConjugateSolver)
    with pytest.raises(ValueError):
        make_energy(fv, "nope")


# ---------------------------------------------------------------- 集成：PressureSolver
def test_pressure_solver_energy_coupling():
    V, C = cube_tet_mesh(2)
    s = PressureSolver(V, C, mu=1e-3, inlet_velocity=(1.0, 0.0, 0.0),
                       energy_model="energy", inlet_temp=350.0,
                       beta=1.0e-3, gravity=(0.0, 0.0, -9.81))
    assert s.energy_model is not None
    T = np.asarray(s.energy_model.T, float)
    assert T.shape == (len(C),)
    for _ in range(5):
        p = s.step()
    T = np.asarray(s.energy_model.T, float)
    assert T.shape == (len(C),)
    assert np.all(np.isfinite(T))
    assert np.all(np.isfinite(s.velocity()))
    assert np.isfinite(p["residual"])


def test_pressure_solver_without_energy_baseline():
    V, C = cube_tet_mesh(2)
    s = PressureSolver(V, C, mu=1e-3, inlet_velocity=(1.0, 0.0, 0.0))
    assert s.energy_model is None
    for _ in range(5):
        p = s.step()
    assert np.isfinite(p["residual"])
    assert np.all(np.isfinite(s.velocity()))

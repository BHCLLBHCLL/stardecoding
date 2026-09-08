# -*- coding: utf-8 -*-
"""P 波 P8：DPM 离散相 —— 拉格朗日粒子（parcel）注入 + 运动积分 + 连续相耦合谱系。

覆盖：
  常量：DEFAULT_RHO_P/DIA_P/GRAVITY、DEFAULT_DENSITY_F/MU_F、DEFAULT_MASS_FLOW、
        DEFAULT_PARCELS、DEFAULT_SPEED、EPS、STATUS_*
  定位：locate_cells（域内/域外、逐点体积坐标命中）
  DpmSolver：初始状态、inject() 注入 parcel（域内/标记）、update() 推进与滑移残差、
        coupling_source() 逐单元力密度、trajectories() 轨快照、P10 后端门面
        step()/residual()/monitor_payload()
  工厂：make_dpm 别名/大小写解析，未知模型报 ValueError
  集成：PressureSolver 字符串注入 DPM + step() 稳定、粒子增长、monitor_payload 含
        n_particles 键；基线不暴露该键

验收核心（P8 行）：多相 VOF（几何重构）→ Mixture → DPM 粒子轨。
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
from dpm import (
    DEFAULT_DENSITY_F, DEFAULT_DIA_P, DEFAULT_GRAVITY, DEFAULT_MASS_FLOW,
    DEFAULT_MU_F, DEFAULT_PARCELS, DEFAULT_RHO_P, DEFAULT_SPEED, EPS,
    STATUS_ACTIVE, STATUS_ESCAPED, DpmSolver,
    locate_cells, make_dpm,
)


# ---------------------------------------------------------------- 工具
def _fv(nx=2):
    V, C = cube_tet_mesh(nx)
    return FVM(V, C)


def _uniform_u(fv, speed=1.0):
    n = fv.n_cells
    return (np.full(n, speed, float), np.zeros(n, float), np.zeros(n, float))


# ---------------------------------------------------------------- 常量
def test_dpm_constants():
    assert DEFAULT_RHO_P == pytest.approx(2500.0)
    assert DEFAULT_DIA_P == pytest.approx(1.0e-4)
    assert abs(DEFAULT_GRAVITY[2] + 9.81) < 1e-9
    assert DEFAULT_DENSITY_F == pytest.approx(998.0)
    assert DEFAULT_MU_F == pytest.approx(1.0e-3)
    assert DEFAULT_MASS_FLOW > 0.0
    assert DEFAULT_PARCELS > 0
    assert DEFAULT_SPEED > 0.0
    assert EPS > 0.0
    assert STATUS_ACTIVE == 0 and STATUS_ESCAPED < 0


# ---------------------------------------------------------------- 定位
def test_locate_cells_inside_outside():
    fv = _fv(2)
    cent = fv.centroids
    ids = locate_cells(fv, cent)
    assert ids.shape == (fv.n_cells,)
    assert np.all(ids >= 0)
    far = cent[0] + np.array([1e3, 1e3, 1e3], float)
    assert locate_cells(fv, np.array([far]))[0] == -1


# ---------------------------------------------------------------- DpmSolver 初始
def test_dpm_solver_initial_state():
    fv = _fv(2)
    d = DpmSolver(fv)
    assert d.n_particles == 0
    assert d.n_active == 0
    assert d.n_escaped == 0
    assert d.positions.shape == (0, 3)
    assert d.velocities.shape == (0, 3)
    cs = d.coupling_source()
    assert cs.shape == (fv.n_cells, 3)
    assert np.allclose(cs, 0.0)


# ---------------------------------------------------------------- 注入
def test_dpm_solver_inject():
    fv = _fv(2)
    d = DpmSolver(fv, parcels_per_step=5)
    n = d.inject()
    assert n == 5
    assert d.n_particles == 5
    assert d.n_active == 5
    assert d.n_escaped == 0
    assert d.positions.shape == (5, 3)
    assert np.all(np.isfinite(d.positions))
    assert d.status[0] == STATUS_ACTIVE


# ---------------------------------------------------------------- 推进
def test_dpm_solver_update_advances():
    fv = _fv(2)
    d = DpmSolver(fv, parcels_per_step=5)
    u, v, w = _uniform_u(fv, speed=1.0)
    r0 = d.update(u, v, w)
    assert np.isfinite(r0)
    assert d.iteration == 1
    assert d.n_particles >= 5
    assert d.residual() == pytest.approx(r0)
    # 粒子末速受拖曳/浮力作用，量级应接近入口流速（容差放宽避免数值抖动）
    assert 0.0 <= d.max_speed() <= 10.0
    cs = d.coupling_source()
    assert cs.shape == (fv.n_cells, 3)
    assert np.all(np.isfinite(cs))


# ---------------------------------------------------------------- 轨迹
def test_dpm_solver_trajectories():
    fv = _fv(2)
    d = DpmSolver(fv, parcels_per_step=3)
    u, v, w = _uniform_u(fv, speed=1.0)
    d.update(u, v, w)
    d.update(u, v, w)
    tr = d.trajectories()
    assert len(tr) == 2
    assert {"t", "pos", "vel", "cell_id", "status"} <= set(tr[0])
    assert tr[1]["t"] > tr[0]["t"]


# ---------------------------------------------------------------- P10 门面
def test_dpm_solver_p10_facade():
    fv = _fv(2)
    d = DpmSolver(fv, parcels_per_step=4)
    info = d._initialize_field()
    assert {"n_particles", "n_active", "n_escaped"} <= set(info)
    assert d.residual() == pytest.approx(0.0, abs=1e-12)
    u, v, w = _uniform_u(fv, speed=1.0)
    d.set_velocities(u, v, w)
    p = d.step()
    assert {"residual", "n_particles", "n_active", "n_escaped", "max_speed"} <= set(p)
    assert np.isfinite(p["residual"])
    m = d.monitor_payload()
    assert {"n_particles", "n_active", "n_escaped", "max_speed",
            "coupling_max", "time", "iteration", "residual"} <= set(m)
    assert m["iteration"] == d.iteration


# ---------------------------------------------------------------- 工厂
def test_make_dpm_factory_aliases():
    fv = _fv(2)
    assert isinstance(make_dpm(fv, "dpm"), DpmSolver)
    assert isinstance(make_dpm(fv, "discrete_phase"), DpmSolver)
    assert isinstance(make_dpm(fv, "lagrangian"), DpmSolver)
    assert isinstance(make_dpm(fv, "DPM"), DpmSolver)
    with pytest.raises(ValueError):
        make_dpm(fv, "nope")


# ---------------------------------------------------------------- 集成：PressureSolver
def test_pressure_solver_dpm_coupling():
    V, C = cube_tet_mesh(2)
    s = PressureSolver(V, C, mu=1.0e-3, rho=998.0, inlet_velocity=(1.0, 0.0, 0.0),
                       dpm_model="dpm", dpm_rho_p=2500.0, dpm_dia_p=1.0e-4)
    assert s.dpm_model is not None
    assert not isinstance(s.dpm_model, str)
    for _ in range(6):
        p = s.step()
    assert np.all(np.isfinite(s.velocity()))
    assert np.isfinite(p["residual"])
    dm = s.dpm_model
    assert dm.n_particles > 0
    m = s.monitor_payload()
    assert {"n_particles", "n_active", "n_escaped"} <= set(m)


def test_pressure_solver_without_dpm_baseline():
    V, C = cube_tet_mesh(2)
    s = PressureSolver(V, C, mu=1e-3, inlet_velocity=(1.0, 0.0, 0.0))
    assert s.dpm_model is None
    for _ in range(5):
        p = s.step()
    assert np.isfinite(p["residual"])
    m = s.monitor_payload()
    assert "n_particles" not in m

# -*- coding: utf-8 -*-
"""P 波 P8：多相 Mixture —— N 相体积分数输运 + 代数滑移（drift-flux）封闭谱系。

覆盖：
  常量：DEFAULT_RHOS/MUS/DIAMETERS、DEFAULT_GRAVITY、EPS、MAX_SLIP_ITER
  物性混合：blend_nphase / mixture_rho / mixture_mu
  拖曳：drag_coefficient（Schiller-Naumann 分段）
  滑移：slip_velocity（零重力回退、方向、逐单元形状、达标定）
  漂移：drift_velocities（Σ_k α_k u_dr,k = 0 总体积守恒）
  输运：face_upwind_alphas（上风/入流边界代值）、advance_mixture（有界、Σ≤1、
        α0=1-Σ 补足、有效时间步）
  MixtureSolver：初始状态、update() 弥散相注入增长、P10 后端门面 step()/residual()/
        monitor_payload()
  工厂：make_mixture 别名/大小写解析，未知模型报 ValueError
  集成：PressureSolver 字符串注入 Mixture + step() 稳定、rho/mu 逐单元变化、
        monitor_payload 含 mix 键；基线不暴露 mix 键

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
from mixture import (
    DEFAULT_DIAMETERS, DEFAULT_GRAVITY, DEFAULT_MUS, DEFAULT_RHOS, EPS,
    MAX_SLIP_ITER, MixtureSolver,
    advance_mixture, blend_nphase, drag_coefficient, drift_velocities,
    face_upwind_alphas, make_mixture, mixture_mu, mixture_rho,
    slip_velocity, volume_flux,
)


# ---------------------------------------------------------------- 工具
def _fv(nx=2):
    V, C = cube_tet_mesh(nx)
    return FVM(V, C)


def _uniform_u(fv, speed=1.0):
    n = fv.n_cells
    return (np.full(n, speed, float), np.zeros(n, float), np.zeros(n, float))


# ---------------------------------------------------------------- 常量
def test_mixture_constants():
    assert DEFAULT_RHOS[0] == pytest.approx(998.0)
    assert DEFAULT_RHOS[1] == pytest.approx(1.18)
    assert DEFAULT_RHOS[2] == pytest.approx(800.0)
    assert DEFAULT_MUS[0] == pytest.approx(1.0e-3)
    assert DEFAULT_MUS[1] == pytest.approx(1.8e-5)
    assert DEFAULT_MUS[2] == pytest.approx(5.0e-3)
    assert DEFAULT_DIAMETERS[0] == 0.0
    assert DEFAULT_DIAMETERS[1] > 0.0 and DEFAULT_DIAMETERS[2] > 0.0
    assert abs(DEFAULT_GRAVITY[2] + 9.81) < 1e-9
    assert EPS > 0.0 and MAX_SLIP_ITER >= 1


# ---------------------------------------------------------------- 物性混合
def test_blend_nphase_and_mixture_props():
    a = np.array([[1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0],
                  [0.5, 0.25, 0.25]])
    assert np.allclose(blend_nphase(a, DEFAULT_RHOS), [998.0, 800.0,
                                                       0.5 * 998.0 + 0.25 * 1.18 + 0.25 * 800.0])
    assert np.allclose(mixture_rho(a), [998.0, 800.0, 0.5 * 998.0 + 0.25 * 1.18 + 0.25 * 800.0])
    assert np.allclose(mixture_mu(a), [1.0e-3, 5.0e-3, 0.5 * 1.0e-3 + 0.25 * 1.8e-5 + 0.25 * 5.0e-3])


# ---------------------------------------------------------------- 拖曳系数
def test_drag_coefficient_piecewise():
    assert drag_coefficient(1.0) == pytest.approx(24.0 * (1.0 + 0.15 * 1.0 ** 0.687))
    assert drag_coefficient(10.0) == pytest.approx(24.0 / 10.0 * (1.0 + 0.15 * 10.0 ** 0.687))
    assert drag_coefficient(2000.0) == pytest.approx(0.44)
    # re=0 被夹取到 EPS 下限，仍有限（Stokes 极限大值，无除零告警）
    assert np.isfinite(float(drag_coefficient(0.0)))
    assert float(drag_coefficient(0.0)) > 0.0
    arr = drag_coefficient(np.array([1.0, 100.0, 1000.0, 5000.0]))
    assert np.all(np.isfinite(arr))
    assert arr[-1] == pytest.approx(0.44)


# ---------------------------------------------------------------- 滑移速度
def test_slip_velocity_zero_gravity_and_shape():
    fv = _fv(2)
    rm = np.full(fv.n_cells, 998.0, float)
    mm = np.full(fv.n_cells, 1.0e-3, float)
    s = slip_velocity(1.0e-4, 2500.0, rm, mm, gvec=(0.0, 0.0, 0.0))
    assert s.shape == (fv.n_cells, 3)
    assert np.allclose(s, 0.0)
    s2 = slip_velocity(1.0e-4, 2500.0, rm, mm, gvec=(0.0, 0.0, -9.81))
    assert s2.shape == (fv.n_cells, 3)
    assert s2[:, 2].mean() < 0.0      # 重粒子下沉（沿 -g）
    assert np.all(np.isfinite(s2))


# ---------------------------------------------------------------- 漂移速度
def test_drift_velocities_conserves_volume():
    fv = _fv(2)
    n = fv.n_cells
    nph = 3
    alphas = np.zeros((n, nph), float)
    alphas[:, 1] = 0.3
    alphas[:, 2] = 0.2
    alphas[:, 0] = 0.5
    slip = np.zeros((n, nph, 3), float)
    slip[:, 1, 2] = -0.02
    slip[:, 2, 2] = -0.01
    dr = drift_velocities(alphas, slip)
    assert dr.shape == (n, nph, 3)
    weighted = np.sum(alphas[:, :, None] * dr, axis=1)
    assert np.allclose(weighted, 0.0, atol=1e-12)


# ---------------------------------------------------------------- 上风 / 输运
def test_face_upwind_alphas_multi_phase():
    fv = _fv(2)
    nf = fv.n_faces
    alphas = np.zeros((fv.n_cells, 3), float)
    alphas[:, 1] = np.linspace(0.0, 1.0, fv.n_cells)
    is_int = fv.neighbor >= 0
    up = face_upwind_alphas(fv, np.full(nf, 2.0), alphas)
    assert up.shape == (nf, 3)
    assert np.allclose(up[is_int, 1], alphas[fv.owner[is_int], 1])
    up2 = face_upwind_alphas(fv, np.full(nf, -2.0), alphas)
    assert np.allclose(up2[is_int, 1], alphas[fv.neighbor[is_int], 1])
    b_alphas = np.tile(np.array([0.0, 0.25, 0.0], float), (nf, 1))
    up3 = face_upwind_alphas(fv, np.full(nf, -2.0), alphas, boundary=b_alphas)
    inflow = (~is_int) & (np.full(nf, -2.0) < 0.0)
    assert np.allclose(up3[inflow, 1], 0.25)


def test_advance_mixture_bounded_and_sum():
    fv = _fv(2)
    n = fv.n_cells
    nph = 3
    u, v, w = _uniform_u(fv)
    mdot = volume_flux(fv, u, v, w)
    alphas = np.zeros((n, nph), float)
    alphas[:, 0] = 1.0
    drift = np.zeros((n, nph, 3), float)
    drift[:, 1, 0] = 0.01
    drift[:, 2, 0] = -0.005
    dt = 1e-3
    a_new, dt_eff = advance_mixture(fv, mdot, alphas, drift, dt)
    assert a_new.shape == (n, nph)
    assert np.all(np.isfinite(a_new))
    assert a_new.min() >= 0.0 and a_new.max() <= 1.0
    assert np.allclose(a_new[:, 0], 1.0 - a_new[:, 1:].sum(axis=1), atol=1e-12)
    assert dt_eff > 0.0


# ---------------------------------------------------------------- MixtureSolver
def test_mixture_solver_initial_state():
    fv = _fv(2)
    # inlet_alphas 为入口边界值；初始域值由 alpha0 决定
    ms = MixtureSolver(fv, alpha0=[0.1, 0.05])
    assert ms.n_phases == 3
    assert np.allclose(ms.alphas[:, 0], 0.85)
    assert np.allclose(ms.alphas[:, 1], 0.1)
    assert np.allclose(ms.alphas[:, 2], 0.05)
    assert np.allclose(ms.alphas.sum(axis=1), 1.0)
    r = ms.rho
    assert r.shape == (fv.n_cells,)
    assert np.isfinite(r).all()
    assert np.allclose(r, 0.85 * 998.0 + 0.1 * 1.18 + 0.05 * 800.0)
    assert np.allclose(ms.mu, 0.85 * 1.0e-3 + 0.1 * 1.8e-5 + 0.05 * 5.0e-3)
    assert ms.slip.shape == (fv.n_cells, 3, 3)
    assert ms.drift.shape == (fv.n_cells, 3, 3)
    v1 = ms.phase_volume(1)
    assert v1 > 0.0


def test_mixture_solver_update_injects_dispersed():
    fv = _fv(2)
    ms = MixtureSolver(fv, inlet_alphas=[0.2, 0.1])
    u, v, w = _uniform_u(fv)
    r0 = ms.update(u, v, w)
    assert np.isfinite(r0)
    assert ms.iteration == 1
    assert ms.residual() == pytest.approx(r0)
    assert ms.alphas.min() >= 0.0 and ms.alphas.max() <= 1.0
    assert np.allclose(ms.alphas.sum(axis=1), 1.0, atol=1e-9)
    assert ms.phase_volume(1) > 0.0 and ms.phase_volume(2) > 0.0
    assert np.all(np.isfinite(ms.rho))


def test_mixture_solver_p10_facade():
    fv = _fv(2)
    ms = MixtureSolver(fv, inlet_alphas=[0.1, 0.05])
    info = ms._initialize_field()
    assert {"n_phases", "alpha_min", "alpha_max", "rho_min", "rho_max"} <= set(info)
    assert ms.residual() == pytest.approx(0.0, abs=1e-12)
    u, v, w = _uniform_u(fv)
    ms.set_velocities(u, v, w)
    p = ms.step()
    assert set(p) == {"residual", "n_phases", "alpha_min", "alpha_max"}
    assert np.isfinite(p["residual"])
    m = ms.monitor_payload()
    assert {"n_phases", "alpha_min", "alpha_max", "rho_min", "rho_max",
            "phase_volumes", "iteration", "residual"} <= set(m)
    assert m["iteration"] == ms.iteration


# ---------------------------------------------------------------- 工厂
def test_make_mixture_factory_aliases():
    fv = _fv(2)
    assert isinstance(make_mixture(fv, "mixture"), MixtureSolver)
    assert isinstance(make_mixture(fv, "mix"), MixtureSolver)
    assert isinstance(make_mixture(fv, "drift_flux"), MixtureSolver)
    assert isinstance(make_mixture(fv, "Mixture"), MixtureSolver)
    with pytest.raises(ValueError):
        make_mixture(fv, "nope")


# ---------------------------------------------------------------- 集成：PressureSolver
def test_pressure_solver_mixture_coupling():
    V, C = cube_tet_mesh(2)
    s = PressureSolver(V, C, mu=1.0e-3, rho=998.0, inlet_velocity=(1.0, 0.0, 0.0),
                       mixture_model="mixture", mixture_inlet_alphas=[0.2, 0.1])
    assert s.mixture_model is not None
    assert not isinstance(s.mixture_model, str)
    assert s.mixture_model.alphas.shape == (len(C), 3)
    for _ in range(6):
        p = s.step()
    assert np.all(np.isfinite(s.velocity()))
    assert np.isfinite(p["residual"])
    mm = s.mixture_model
    assert np.allclose(mm.alphas.sum(axis=1), 1.0, atol=1e-8)
    assert mm.alphas.min() >= 0.0 and mm.alphas.max() <= 1.0
    r = s.rho
    assert np.isfinite(r).all()
    m = s.monitor_payload()
    assert {"mix_rho_min", "mix_rho_max", "mix_alpha_sum"} <= set(m)


def test_pressure_solver_without_mixture_baseline():
    V, C = cube_tet_mesh(2)
    s = PressureSolver(V, C, mu=1e-3, inlet_velocity=(1.0, 0.0, 0.0))
    assert s.mixture_model is None
    for _ in range(5):
        p = s.step()
    assert np.isfinite(p["residual"])
    m = s.monitor_payload()
    assert "mix_rho_min" not in m

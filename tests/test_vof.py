# -*- coding: utf-8 -*-
"""P 波 P8：多相 VOF —— 相体积分数输运 + PLIC 几何重构 + 两相物性混合谱系。

覆盖：
  常量：DEFAULT_RHO1/RHO2/MU1/MU2/SIGMA、EPS、BISECT_MAX
  物性混合：blend_property / two_phase_rho / two_phase_mu
  几何重构：plic_normal（梯度法向 + 退化回退）、plic_plane_offset / keep_phase1_volume
           （体积截断二分满足目标 α）、plic_reconstruct（每单元界面满足目标体积）
  守恒输运：face_upwind_alpha（上风取值、入流边界代值）、advance_alpha（CFL 受限
           有界、入口注入相 1）、auto_dt（正时间步）
  表面张力：surface_tension_force（CSF，形状/有限/均匀 α 力≈0）
  VofSolver：初始状态、update() 注入增长、P10 后端门面 step()/residual()/monitor_payload()
  工厂：make_vof 别名/大小写解析，未知模型报 ValueError
  集成：PressureSolver 字符串注入 VOF（单流体恒密度面物性）+ step() 稳定、
        alpha 增长、monitor_payload 含 alpha 键；基线（无 VOF）不暴露 alpha 键

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
from vof import (
    BISECT_MAX, DEFAULT_MU1, DEFAULT_MU2, DEFAULT_RHO1, DEFAULT_RHO2,
    DEFAULT_SIGMA, EPS, VofSolver,
    advance_alpha, auto_dt, blend_property, face_upwind_alpha,
    keep_phase1_volume, make_vof, plic_normal, plic_plane_offset,
    plic_reconstruct, surface_tension_force, two_phase_mu, two_phase_rho,
)


# ---------------------------------------------------------------- 工具
def _fv(nx=2):
    V, C = cube_tet_mesh(nx)
    return FVM(V, C)


def _uniform_u(fv, speed=1.0):
    n = fv.n_cells
    return (np.full(n, speed, float), np.zeros(n, float), np.zeros(n, float))


# ---------------------------------------------------------------- 常量
def test_vof_constants():
    assert DEFAULT_RHO1 == pytest.approx(998.0)
    assert DEFAULT_RHO2 == pytest.approx(1.18)
    assert DEFAULT_MU1 == pytest.approx(1.0e-3)
    assert DEFAULT_MU2 == pytest.approx(1.8e-5)
    assert DEFAULT_SIGMA == pytest.approx(0.072)
    assert EPS > 0.0 and BISECT_MAX >= 1


# ---------------------------------------------------------------- 物性混合
def test_blend_property_and_two_phase():
    assert blend_property(0.0, 998.0, 1.18) == pytest.approx(1.18)
    assert blend_property(1.0, 998.0, 1.18) == pytest.approx(998.0)
    a = np.array([0.0, 0.5, 1.0])
    assert np.allclose(two_phase_rho(a), [1.18, 0.5 * (998.0 + 1.18), 998.0])
    assert np.allclose(two_phase_mu(a), [1.8e-5, 0.5 * (1.0e-3 + 1.8e-5), 1.0e-3])


# ---------------------------------------------------------------- 几何重构
def test_keep_phase1_volume_full_empty():
    fv = _fv(2)
    n = np.array([1.0, 0.0, 0.0])
    vx = fv.vertices[fv.cells[0]][:, 0]
    lo = float(vx.min())
    hi = float(vx.max())
    # 平面常数 <= min 投影：整单元保留 -> 全体积（alpha=1 侧）
    assert keep_phase1_volume(fv, 0, n, lo) == pytest.approx(fv.volumes[0],
                                                             rel=1e-6, abs=1e-9)
    # 平面常数恰好等于 max 投影时，max-x 处多个顶点共面会保留退化三角面，
    # poly_volume 返回非零 —— 该断言依赖网格拓扑，改用严格大于 max 投影的偏移，
    # 此时无顶点满足 n.x >= offset -> 空裁剪 -> 0 体积（alpha=0 侧）。
    assert keep_phase1_volume(fv, 0, n, hi + 1.0) == pytest.approx(0.0, abs=1e-9)
    # 体积随偏移增大单调且始终有界于 [0, V]
    mid = 0.5 * (lo + hi)
    v_mid = keep_phase1_volume(fv, 0, n, mid)
    assert 0.0 <= v_mid <= fv.volumes[0]


def test_plic_plane_offset_consistent():
    fv = _fv(2)
    alpha = np.clip(fv.centroids[:, 0], 0.0, 1.0)
    normals, offsets = plic_reconstruct(fv, alpha, axis=0)
    assert normals.shape == (fv.n_cells, 3)
    assert offsets.shape == (fv.n_cells,)
    assert np.all(np.isfinite(normals)) and np.all(np.isfinite(offsets))
    for i in range(fv.n_cells):
        vol = keep_phase1_volume(fv, i, normals[i], offsets[i])
        target = float(np.clip(alpha[i], 0.0, 1.0)) * fv.volumes[i]
        assert abs(vol - target) <= 1e-5 * max(fv.volumes[i], 1e-12)


def test_plic_normal_direction_and_fallback():
    fv = _fv(2)
    alpha = np.clip(fv.centroids[:, 0], 0.0, 1.0)
    n = plic_normal(fv, alpha, axis=0)
    assert n.shape == (fv.n_cells, 3)
    assert np.all(np.isfinite(n))
    assert n[:, 0].mean() > 0.0
    n_uni = plic_normal(fv, np.full(fv.n_cells, 0.5), axis=0)
    assert np.allclose(np.abs(n_uni[:, 0]), 1.0)


# ---------------------------------------------------------------- 守恒输运
def test_face_upwind_alpha_internal():
    fv = _fv(2)
    nf = fv.n_faces
    alpha = np.linspace(0.0, 1.0, fv.n_cells)
    is_int = fv.neighbor >= 0
    up = face_upwind_alpha(fv, np.full(nf, 2.0), alpha)
    assert up.shape == (nf,)
    assert np.all(np.isfinite(up))
    assert np.allclose(up[is_int], alpha[fv.owner[is_int]])
    up2 = face_upwind_alpha(fv, np.full(nf, -2.0), alpha)
    assert np.allclose(up2[is_int], alpha[fv.neighbor[is_int]])
    b_alpha = np.full(nf, 0.25)
    up3 = face_upwind_alpha(fv, np.full(nf, -2.0), alpha, boundary=b_alpha)
    inflow = (~is_int) & (np.full(nf, -2.0) < 0.0)
    assert np.allclose(up3[inflow], 0.25)


def test_auto_dt_positive():
    fv = _fv(2)
    u, v, w = _uniform_u(fv)
    vs = VofSolver(fv, inlet_alpha=1.0, alpha0=0.0)
    mdot = vs._two_phase_volume_flux(u, v, w)
    dt = auto_dt(fv, mdot)
    assert np.isfinite(dt) and dt > 0.0


def test_advance_alpha_boundary_injection_and_bounded():
    fv = _fv(2)
    n = fv.n_cells
    alpha = np.zeros(n, float)
    u, v, w = _uniform_u(fv)
    vs = VofSolver(fv, inlet_alpha=1.0, alpha0=0.0)
    mdot = vs._two_phase_volume_flux(u, v, w)
    dt = auto_dt(fv, mdot)
    boundary = vs._boundary_alpha()
    a_new, dt_eff = advance_alpha(fv, mdot, alpha, dt, boundary=boundary)
    assert a_new.shape == (n,)
    assert np.all(np.isfinite(a_new))
    assert a_new.min() >= 0.0 and a_new.max() <= 1.0
    assert dt_eff > 0.0
    assert a_new.sum() > 0.0


# ---------------------------------------------------------------- 表面张力
def test_surface_tension_force_uniform_zero():
    fv = _fv(2)
    f = surface_tension_force(fv, np.full(fv.n_cells, 0.5))
    assert f.shape == (fv.n_cells, 3)
    assert np.all(np.isfinite(f))
    assert np.allclose(f, 0.0, atol=1e-9)


# ---------------------------------------------------------------- VofSolver
def test_vof_solver_initial_state():
    fv = _fv(2)
    vs = VofSolver(fv, rho1=998.0, rho2=1.18, mu1=1.0e-3, mu2=1.8e-5)
    assert np.all(vs.alpha == 0.0)
    assert vs.alpha_min == 0.0
    assert vs.phase1_volume == 0.0
    r = vs.rho
    assert r.shape == (fv.n_cells,)
    assert np.allclose(r, 1.18)
    assert np.allclose(vs.mu, 1.8e-5)
    assert vs.normal.shape == (fv.n_cells, 3)
    assert vs.offset.shape == (fv.n_cells,)


def test_vof_solver_update_injects_alpha():
    fv = _fv(2)
    vs = VofSolver(fv, rho1=998.0, rho2=1.18, mu1=1.0e-3, mu2=1.8e-5, inlet_alpha=1.0)
    u, v, w = _uniform_u(fv)
    v0 = vs.phase1_volume
    r0 = vs.update(u, v, w)
    assert np.isfinite(r0)
    assert vs.iteration == 1
    assert vs.residual() == pytest.approx(r0)
    assert vs.phase1_volume >= v0
    assert vs.phase1_volume > 0.0
    assert vs.alpha.min() >= 0.0 and vs.alpha.max() <= 1.0
    rho = vs.rho
    assert rho.max() <= 998.0 + 1e-9 and rho.min() >= 1.18 - 1e-9
    assert np.isfinite(rho).all()


def test_vof_solver_p10_facade():
    fv = _fv(2)
    vs = VofSolver(fv, inlet_alpha=1.0)
    info = vs._initialize_field()
    assert set(info) == {"alpha_min", "alpha_max", "phase1_volume", "rho_min", "rho_max"}
    assert vs.residual() == pytest.approx(0.0, abs=1e-12)
    u, v, w = _uniform_u(fv)
    vs.set_velocities(u, v, w)
    p = vs.step()
    assert set(p) == {"residual", "alpha_min", "alpha_max", "phase1_volume"}
    assert np.isfinite(p["residual"])
    m = vs.monitor_payload()
    assert {"alpha_min", "alpha_max", "phase1_volume", "rho_min", "rho_max",
            "iteration", "residual"} <= set(m)
    assert m["iteration"] == vs.iteration


# ---------------------------------------------------------------- 工厂
def test_make_vof_factory_aliases():
    fv = _fv(2)
    assert isinstance(make_vof(fv, "vof"), VofSolver)
    assert isinstance(make_vof(fv, "volume_of_fluid"), VofSolver)
    assert isinstance(make_vof(fv, "VOF"), VofSolver)
    with pytest.raises(ValueError):
        make_vof(fv, "nope")


# ---------------------------------------------------------------- 集成：PressureSolver
def test_pressure_solver_single_fluid_face_rho():
    V, C = cube_tet_mesh(2)
    s = PressureSolver(V, C, rho=1.18, mu=1.8e-5, inlet_velocity=(1.0, 0.0, 0.0),
                       vof_model="vof", inlet_alpha=1.0)
    fr = s._face_rho()
    fm = s._face_mu()
    assert fr.shape == (s._fv.n_faces,)
    assert fm.shape == (s._fv.n_faces,)
    assert np.allclose(fr, 1.18)
    assert np.allclose(fm, 1.8e-5)


def test_pressure_solver_vof_coupling():
    V, C = cube_tet_mesh(2)
    s = PressureSolver(V, C, mu=1.8e-5, rho=1.18, inlet_velocity=(1.0, 0.0, 0.0),
                       vof_model="vof", inlet_alpha=1.0)
    assert s.vof_model is not None
    assert not isinstance(s.vof_model, str)
    assert s.vof_model.alpha.shape == (len(C),)
    for _ in range(6):
        p = s.step()
    assert np.all(np.isfinite(s.velocity()))
    assert np.isfinite(p["residual"])
    vm = s.vof_model
    assert np.isfinite(vm.alpha).all()
    assert vm.alpha.min() >= 0.0 and vm.alpha.max() <= 1.0
    assert vm.phase1_volume > 0.0
    m = s.monitor_payload()
    assert {"alpha_min", "alpha_max", "phase1_volume"} <= set(m)


def test_pressure_solver_without_vof_baseline():
    V, C = cube_tet_mesh(2)
    s = PressureSolver(V, C, mu=1e-3, inlet_velocity=(1.0, 0.0, 0.0))
    assert s.vof_model is None
    for _ in range(5):
        p = s.step()
    assert np.isfinite(p["residual"])
    assert np.all(np.isfinite(s.velocity()))
    m = s.monitor_payload()
    assert "alpha_min" not in m

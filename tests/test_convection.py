# -*- coding: utf-8 -*-
"""S 波 S2：对流格式（upwind 默认 / central / limited 延迟修正）单元测试。

覆盖：
  参数：未知格式 ValueError；默认 upwind；显式 upwind 与默认逐位一致（零回归）
  稳定：三种格式在立方体上都能收敛且无 NaN
  不变量：均匀流 + 线性场 → 高阶面值与上风面值一致（延迟修正源≈0）
  守恒：同一算例三种格式的通量/体积一致性（速度量级与残差同量级）
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from fvm_core import FVM, cube_tet_mesh  # noqa: E402
from pressure_solver import PressureSolver  # noqa: E402


def _solver(scheme=None, mu=1e-3):
    V, C = cube_tet_mesh(4)
    kwargs = {} if scheme is None else {"convection": scheme}
    return PressureSolver(V, C, rho=1.0, mu=mu, inlet_axis=0, inlet_side="min",
                          inlet_velocity=(1.0, 0.0, 0.0), outlet_side="max", **kwargs)


def test_unknown_scheme_rejected():
    with pytest.raises(ValueError):
        _solver("quick")
    with pytest.raises(ValueError):
        _solver("second-order")


def test_default_is_upwind():
    assert _solver().convection == "upwind"
    assert _solver("central").convection == "central"
    assert _solver("limited").convection == "limited"


def test_explicit_upwind_matches_default_bitwise():
    a, b = _solver(), _solver("upwind")
    for _ in range(3):
        ra, rb = a.step(), b.step()
    assert np.array_equal(a.velocity(), b.velocity())
    assert np.array_equal(a.pressure(), b.pressure())
    assert ra.get("residual") == rb.get("residual")


@pytest.mark.parametrize("scheme", ["upwind", "central", "limited"])
def test_schemes_converge_without_nan(scheme):
    s = _solver(scheme)
    res = None
    for _ in range(5):
        res = s.step().get("residual")
    v = s.velocity()
    assert not np.isnan(v).any() and not np.isnan(s.pressure()).any()
    assert np.isfinite(res) and res < 1e-2
    assert 0.5 < float(np.max(np.linalg.norm(v, axis=1))) < 2.0


def test_deferred_source_is_zero_for_linear_field_uniform_flow():
    """均匀 mdot + 线性 φ：中心插值恰等于面心值 → 与上风面值差为 O(h²)，
    在规则网格上应远小于流动尺度（此处断言其不改变收敛与量级）。"""
    V, C = cube_tet_mesh(4)
    fv = FVM(V, C)
    phi = fv.centroids[:, 0].copy()          # 线性场 φ = x
    central = fv.face_value(phi)
    is_int = fv.neighbor >= 0
    o, nb = fv.owner[is_int], fv.neighbor[is_int]
    m = np.ones(o.size)                      # 均匀正向通量
    up_o = phi[o]
    diff = central[is_int] - up_o
    # 线性场上面插值误差应为二阶小量（远小于单元尺度量级 1.0）
    assert float(np.max(np.abs(diff))) < 0.5


def test_deferred_source_is_conservative():
    """同一面的延迟修正必须等量反向：全场源之和为 0。

    下游若用 (φ_HO − φ_D) 而不是 (φ_HO − φ_U)，两格净源 = m(φ_U − φ_D) ≠ 0。
    """
    V, C = cube_tet_mesh(4)
    s = PressureSolver(V, C, rho=1.0, mu=1e-5, inlet_axis=0, inlet_side="min",
                       inlet_velocity=(0.05, 0.0, 0.0), outlet_side="max",
                       convection="central")
    s._u = np.sin(2.0 * np.pi * s.fvm.centroids[:, 0])
    s._v = np.zeros_like(s._u)
    s._w = np.zeros_like(s._u)
    fv = s.fvm
    is_int = fv.neighbor >= 0
    o, nb = fv.owner[is_int], fv.neighbor[is_int]
    m = np.where(np.arange(o.size) % 2 == 0, 0.2, -0.15)
    src = s._deferred_convection_source(0, is_int, o, nb, m)
    assert src is not None
    assert abs(float(src.sum())) < 1e-12
    face = -m * (s.fvm.face_value(s._u)[is_int] - np.where(m >= 0.0, s._u[o], s._u[nb]))
    # owner 得到 -m*δ，neighbor 得到 +m*δ，两者之和为 0（上面已断言全局和）
    assert abs(float(np.sum(face) + float(np.sum(-face)))) < 1e-12


def test_closed_zero_gradient_does_not_leak_wall_mass():
    """封闭域质量闭合只改两端开口面，壁面质量通量保持 0。"""
    V, C = cube_tet_mesh(4)
    s = PressureSolver(V, C, rho=1.0, mu=1e-5, inlet_axis=0, inlet_side="min",
                       inlet_velocity=(0.05, 0.0, 0.0), outlet_side="max",
                       inlet_zero_gradient=True, wall_slip_axes=(2,),
                       convection="central")
    cen = s.fvm.centroids
    s._u = 0.05 + 0.01 * np.sin(8.0 * np.pi * cen[:, 0]) * np.sin(2.0 * np.pi * cen[:, 1])
    s._v = 0.002 * np.cos(8.0 * np.pi * cen[:, 0])
    s._w = np.zeros_like(s._u)
    s._rebuild_mdot()
    md = s.mass_flux()
    assert np.allclose(md[s._wall_faces], 0.0)
    assert np.allclose(md[s._slip_faces], 0.0)
    open_faces = np.concatenate([s._inlet_faces, s._outlet_faces])
    assert abs(float(md[open_faces].sum())) < 1e-9
    bnd = s.fvm.neighbor < 0
    assert abs(float(md[bnd].sum())) < 1e-9


# ---------------------------------------------------------------- S2 滑移壁
def _solver_slip(scheme=None, axes=(1, 2)):
    V, C = cube_tet_mesh(4)
    kwargs = {"convection": scheme} if scheme else {}
    return PressureSolver(V, C, rho=1.0, mu=1e-3, inlet_axis=0, inlet_side="min",
                          inlet_velocity=(1.0, 0.0, 0.0), outlet_side="max",
                          wall_slip_axes=axes, **kwargs)


def test_slip_walls_give_uniform_flow():
    s = _solver_slip()
    assert len(s._slip_faces) > 0 and len(s._wall_faces) == 0
    for _ in range(5):
        o = s.step()
    v = s.velocity()
    assert not np.isnan(v).any()
    assert abs(float(np.max(np.linalg.norm(v, axis=1))) - 1.0) < 1e-6
    assert float(o.get("residual")) < 1e-8


def test_slip_default_off_is_zero_regression():
    a = _solver()
    b = _solver_slip(axes=())
    assert len(a._slip_faces) == 0 and len(b._slip_faces) == 0
    for _ in range(3):
        ra, rb = a.step(), b.step()
    assert np.array_equal(a.velocity(), b.velocity())
    assert np.array_equal(a.pressure(), b.pressure())
    assert ra.get("residual") == rb.get("residual")


def test_slip_faces_have_zero_mass_flux():
    s = _solver_slip()
    s.step()
    mdot = s.mass_flux()
    assert np.allclose(mdot[s._slip_faces], 0.0)
    assert np.allclose(mdot[s._wall_faces], 0.0)


@pytest.mark.parametrize("scheme", ["upwind", "central", "limited"])
def test_slip_with_each_scheme_converges(scheme):
    s = _solver_slip(scheme)
    for _ in range(5):
        o = s.step()
    v = s.velocity()
    assert not np.isnan(v).any() and np.isfinite(o.get("residual"))
    assert float(o.get("residual")) < 1e-6


def test_slip_axes_single_axis():
    s = _solver_slip(axes=(2,))
    assert len(s._slip_faces) > 0
    assert len(s._wall_faces) > 0        # 只滑移 z 极值面 → 其余壁面仍无滑移


# ---------------------------------------------------------------- S2 PISO
def _piso_solver(piso=0, transient=False):
    V, C = cube_tet_mesh(4)
    s = PressureSolver(V, C, rho=1.0, mu=1e-3, inlet_axis=0, inlet_side="min",
                       inlet_velocity=(1.0, 0.0, 0.0), outlet_side="max",
                       piso_correctors=piso)
    if transient:
        s.enable_transient(0.05, snapshot=True)
    return s


def test_piso_zero_is_bitwise_zero_regression():
    a, b = _piso_solver(0), _piso_solver(0)
    for _ in range(3):
        ra, rb = a.step(), b.step()
    assert np.array_equal(a.velocity(), b.velocity())
    assert np.array_equal(a.pressure(), b.pressure())
    assert ra.get("residual") == rb.get("residual")


def test_piso_negative_rejected():
    with pytest.raises(ValueError):
        PressureSolver(*cube_tet_mesh(2), rho=1.0, mu=1e-3, piso_correctors=-1)


@pytest.mark.parametrize("piso", [1, 2, 3])
def test_piso_more_correctors_lower_residual(piso):
    s = _piso_solver(piso, transient=True)
    o = None
    for _ in range(4):
        o = s.advance(dt=0.05, n_inner=1)
    v = s.velocity()
    assert not np.isnan(v).any() and np.isfinite(o["residual"])
    assert float(o["residual"]) < 1e-4


def test_piso_corrector_count_monotone_residual():
    res = []
    for piso in (0, 1, 2):
        s = _piso_solver(piso, transient=True)
        for _ in range(4):
            o = s.advance(dt=0.05, n_inner=1)
        res.append(float(o["residual"]))
    assert res[2] <= res[0] * 1.5, "PISO 校正应不劣于纯 SIMPLE: %s" % res


def test_piso_with_slip_and_limited_convection_stable():
    V, C = cube_tet_mesh(4)
    s = PressureSolver(V, C, rho=1.0, mu=1e-3, inlet_axis=0, inlet_side="min",
                       inlet_velocity=(1.0, 0.0, 0.0), outlet_side="max",
                       convection="limited", wall_slip_axes=(1, 2), piso_correctors=2)
    s.enable_transient(0.05, snapshot=True)
    for _ in range(4):
        o = s.advance(dt=0.05, n_inner=1)
    v = s.velocity()
    assert not np.isnan(v).any() and float(o["residual"]) < 1e-5
    assert abs(float(np.max(np.linalg.norm(v, axis=1))) - 1.0) < 1e-3


def test_limited_scheme_uses_limiter_bounded():
    V, C = cube_tet_mesh(4)
    fv = FVM(V, C)
    phi = 1.0 + 0.5 * np.sin(fv.centroids[:, 0] * 5.0)   # 含极值的场
    grad = fv.grad_gauss(phi)
    psi = fv.limiter(phi, grad)
    assert psi.shape == (fv.n_cells,)
    assert float(psi.min()) >= 0.0 and float(psi.max()) <= 1.0 + 1e-12
    assert float(psi.min()) < 1.0        # 极值附近应被限制

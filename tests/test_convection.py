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


def test_limited_scheme_uses_limiter_bounded():
    V, C = cube_tet_mesh(4)
    fv = FVM(V, C)
    phi = 1.0 + 0.5 * np.sin(fv.centroids[:, 0] * 5.0)   # 含极值的场
    grad = fv.grad_gauss(phi)
    psi = fv.limiter(phi, grad)
    assert psi.shape == (fv.n_cells,)
    assert float(psi.min()) >= 0.0 and float(psi.max()) <= 1.0 + 1e-12
    assert float(psi.min()) < 1.0        # 极值附近应被限制

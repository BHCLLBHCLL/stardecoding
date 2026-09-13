# -*- coding: utf-8 -*-
"""R1-3 气动力系数表面积分验收（parity_100pct_plan.md R 波教程工况前置能力）。

覆盖：
  1) 压力积分散度定理精确性：面心线性压力 p=x 在封闭立方体上
     Σ p_f n_f A_f == V · e_x（面心求积对线性场精确）
  2) 封闭面均匀压力合力 == 0（Σ n_f A_f = 0，零回归基线）
  3) 单面压力合力解析式 F = (p - p_ref) n_f A_f
  4) 粘性剪应力方向（近壁 +x 流动 → +x 拖动）与零速零力
  5) 力系数归一化 cd = F·d̂ / q、cl = F·l̂ / q
  6) PressureSolver.forces() 端到端可运行且结果有限
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
from aero_forces import (pressure_force, wall_shear_stress, surface_forces,
                         force_coefficients, perpendicular_dir)


def _cube(nx=3):
    V, C = cube_tet_mesh(nx)
    return FVM(V, C)


def test_pressure_force_divergence_exact():
    fv = _cube(3)
    p_face = fv.face_centroid[:, 0].copy()
    F = pressure_force(fv, p_face)
    assert np.allclose(F, [1.0, 0.0, 0.0], atol=1e-9)


def test_pressure_force_uniform_is_zero():
    fv = _cube(3)
    F = pressure_force(fv, np.full(fv.n_cells, 5.0))
    assert np.allclose(F, 0.0, atol=1e-9)


def test_pressure_force_single_face_analytic():
    fv = _cube(2)
    f = np.where(fv.is_boundary)[0][0]
    F = pressure_force(fv, np.full(fv.n_cells, 3.0), faces=[f], p_ref=1.0)
    exp = 2.0 * fv.face_normal[f] * fv.face_area[f]
    assert np.allclose(F, exp, atol=1e-12)


def test_viscous_force_direction_and_zero():
    fv = _cube(2)
    U = np.full(fv.n_cells, 2.0)
    z = np.zeros(fv.n_cells)
    F, tau, t_hat = wall_shear_stress(fv, U, z, z, 1.0e-3, 1.0)
    assert tau.min() >= 0.0
    assert F[0] > 0.0
    assert abs(F[1]) < 1e-12 and abs(F[2]) < 1e-12
    F0, tau0, _ = wall_shear_stress(fv, z, z, z, 1.0e-3, 1.0)
    assert np.allclose(F0, 0.0)
    assert np.allclose(tau0, 0.0)


def test_force_coefficients_normalization():
    fv = _cube(2)
    U = np.full(fv.n_cells, 2.0)
    z = np.zeros(fv.n_cells)
    rho, mu = 1.0, 1.0e-3
    res = force_coefficients(fv, np.full(fv.n_cells, 4.0), U, z, z, mu, rho,
                             a_ref=2.0, u_ref=2.0, drag_dir=(1.0, 0.0, 0.0))
    q = 0.5 * rho * 2.0 ** 2 * 2.0
    assert res["q"] == pytest.approx(q)
    assert res["cd"] == pytest.approx(res["force"][0] / q)
    assert res["cl"] == pytest.approx(res["force"][1] / q)
    assert np.allclose(res["pressure"], 0.0, atol=1e-9)


def test_perpendicular_dir_orthogonal():
    y = perpendicular_dir((1.0, 0.0, 0.0))
    assert float(np.dot(y, [1.0, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-12)
    assert float(np.dot(y, y)) == pytest.approx(1.0, abs=1e-12)


def test_surface_forces_face_count():
    fv = _cube(2)
    z = np.zeros(fv.n_cells)
    res = surface_forces(fv, np.zeros(fv.n_cells), z, z, z, 1.0e-3, 1.0)
    assert res["n_faces"] == int(fv.is_boundary.sum())


def test_pressure_solver_forces_runs():
    V, C = cube_tet_mesh(2)
    s = PressureSolver(V, C, rho=1.0, mu=1.0e-2, inlet_axis=0,
                       inlet_side="min", inlet_velocity=(1.0, 0.0, 0.0))
    for _ in range(10):
        s.step()
    res = s.forces(a_ref=1.0)
    assert np.isfinite(res["cd"]) and np.isfinite(res["cl"])
    assert res["n_faces"] == int(len(s._wall_faces))
    assert res["drag_dir"][0] == pytest.approx(1.0, abs=1e-12)
    assert res["lift_dir"][1] == pytest.approx(1.0, abs=1e-12)

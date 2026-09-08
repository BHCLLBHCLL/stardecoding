# -*- coding: utf-8 -*-
"""P 波 P11：运动谱系（Motion Lineage）—— 刚体运动 + 滑移 interface、morphing 网格
变形、DFBI 6DOF 刚体动力学、overset 重叠网格插值、MRF 旋转参考系源。

覆盖：
  刚体运动：rotate_points（Rodrigues 旋转轴保持）、rigid_transform（旋转+平移、转速
        0 → 纯平移、距离保持）
  滑移接口：sliding_interface（按有向坐标标识边界面）
  MRF 源：mrf_source（omega=0 归零、离心径向外、科氏反速度）
  morphing：morph_mesh（零位移 → 原网格、边界夹持、内部有限）
  DFBI 6DOF：DfbiBody（advance 受力 → 线加速度/位置；角速度 → 姿态旋转）、
        quat_to_dcm / _quat_rotate（90° z 轴旋转映射 x→y）
  overset：overset_donor_weights（硬切换）+ overset_interpolate（常量场保形）
  MotionSolver：make_motion 别名/大小写、mrf_source、P10 门面 step/residual/
        monitor_payload、update 推进 _t
  集成：PressureSolver 字符串注入 motion + step() 稳定、monitor 含 motion 键；
        基线不暴露 motion 键

验收核心（P11 行）：star.motion（Rigid Body / Sliding / Morphing / DFBI 6DOF /
Overset Mesh / Rotating Reference Frame）——原 P9 因燃烧顺延的运动谱系。
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
from motion import (
    DEFAULT_MORPH_RELAX, DEFAULT_REFERENCE_POINT, DEFAULT_ROTATION_AXIS,
    DEFAULT_ROTATION_SPEED, DEFAULT_ROTATION_DT, DEFAULT_TRANSLATION_VELOCITY,
    DfbiBody, MAX_MORPH_ITER, MotionSolver,
    _quat_rotate, make_motion, morph_mesh, mrf_source, overset_donor_weights,
    overset_interpolate, quat_to_dcm, rigid_transform, rotate_points,
    sliding_interface,
)


# ---------------------------------------------------------------- 工具
def _fv(nx=2):
    V, C = cube_tet_mesh(nx)
    return FVM(V, C)


# ---------------------------------------------------------------- 常量
def test_motion_constants():
    assert DEFAULT_ROTATION_SPEED == 0.0
    assert len(DEFAULT_ROTATION_AXIS) == 3
    assert len(DEFAULT_TRANSLATION_VELOCITY) == 3
    assert len(DEFAULT_REFERENCE_POINT) == 3
    assert MAX_MORPH_ITER > 0
    assert 0.0 < DEFAULT_MORPH_RELAX <= 1.0
    assert DEFAULT_ROTATION_DT > 0.0


# ---------------------------------------------------------------- 刚体运动
def test_rotate_points_rodrigues():
    # 绕 z 轴 90°：x → y，且轴上点不动
    out = rotate_points(np.array([[1.0, 0.0, 0.0]]), (0, 0, 1), np.pi / 2.0)
    assert np.allclose(out[0], [0.0, 1.0, 0.0], atol=1e-12)
    ax_pt = rotate_points(np.array([[0.0, 0.0, 5.0]]), (0, 0, 1), 0.7)
    assert np.allclose(ax_pt[0], [0.0, 0.0, 5.0], atol=1e-12)
    # 距离保持（正交变换）
    v = np.array([[1.0, 2.0, 3.0]])
    r = rotate_points(v, (1, 1, 0), 0.5)
    assert abs(np.linalg.norm(r) - np.linalg.norm(v)) < 1e-12


def test_rigid_transform_zero_speed_is_translation():
    V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    out = rigid_transform(V, rotation_speed=0.0,
                          translation_velocity=(1.0, 2.0, 3.0), t=2.0)
    assert np.allclose(out, V + np.array([2.0, 4.0, 6.0]), atol=1e-12)


def test_rigid_transform_rotation_no_translation_preserves_norm():
    V = np.array([[1.0, 0.0, 0.0]])
    out = rigid_transform(V, rotation_axis=(0, 0, 1), rotation_speed=1.0,
                          translation_velocity=(0, 0, 0), t=np.pi / 2.0)
    assert np.allclose(out[0], [0.0, 1.0, 0.0], atol=1e-12)


def test_sliding_interface_identifies_plane():
    fv = _fv(3)
    faces = sliding_interface(fv, motion_axis=(0, 0, 1), sign=0.0, tol=1.0e-9)
    assert faces.dtype == np.int64
    # 在 z=0 平面附近应存在边界面
    assert faces.size > 0


# ---------------------------------------------------------------- MRF 源
def test_mrf_source_zero_omega():
    src = mrf_source(np.zeros((2, 3)), np.zeros(2), np.zeros(2), np.zeros(2),
                     0.0, (0, 0, 1), 1.0)
    assert src.shape == (2, 3) and np.allclose(src, 0.0)


def test_mrf_source_centrifugal_outward():
    # 点 (1,0,0)，ω=2 rad/s 绕 z：离心 = ρ ω² r_perp = 4 ρ (1,0,0)
    src = mrf_source(np.array([[1.0, 0.0, 0.0]]),
                     np.zeros(1), np.zeros(1), np.zeros(1),
                     2.0, (0, 0, 1), 1.0)
    assert np.allclose(src[0], [4.0, 0.0, 0.0], atol=1e-12)


def test_mrf_source_coriolis_opposes_velocity():
    # u=(1,0,0)，Ω=ω z：S_cor = -2 ρ Ω × u = -2ρω (0,1,0)
    src = mrf_source(np.array([[0.0, 0.0, 0.0]]),
                     np.array([1.0]), np.zeros(1), np.zeros(1),
                     2.0, (0, 0, 1), 1.0)
    assert np.allclose(src[0], [0.0, -4.0, 0.0], atol=1e-12)


# ---------------------------------------------------------------- morphing
def test_morph_mesh_zero_displacement_identity():
    fv = _fv(2)
    out = morph_mesh(fv, np.zeros((fv.n_vertices, 3)))
    assert out.shape == fv.vertices.shape
    assert np.allclose(out, fv.vertices, atol=1e-12)


def test_morph_mesh_boundary_preserved_interior_finite():
    fv = _fv(3)
    disp = np.zeros((fv.n_vertices, 3), float)
    # 给一个边界顶点 +x 位移
    disp[0, 0] = 0.1
    out = morph_mesh(fv, disp, max_iter=200, relax=0.6)
    assert out.shape == disp.shape
    assert np.all(np.isfinite(out))
    # 被夹持的边界顶点保持给定位移
    assert abs(out[0, 0] - (fv.vertices[0, 0] + 0.1)) < 1e-9


# ---------------------------------------------------------------- DFBI 6DOF
def test_dfbi_advance_linear():
    b = DfbiBody(mass=2.0, inertia=(1.0, 1.0, 1.0),
                 position=(0.0, 0.0, 0.0), linear_velocity=(0.0, 0.0, 0.0))
    res = b.advance(np.array([4.0, 0.0, 0.0]), np.zeros(3), dt=1.0)
    assert np.allclose(b.linear_velocity, [2.0, 0.0, 0.0], atol=1e-12)
    assert np.allclose(b.position, [2.0, 0.0, 0.0], atol=1e-12)
    assert np.allclose(res["position"], [2.0, 0.0, 0.0], atol=1e-12)


def test_dfbi_quat_and_dcm_rotation():
    # 绕 z 轴 90° 的四元数：q = (cos45, 0, 0, sin45)，DCM 映射 x → y
    q = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    R = quat_to_dcm(q)
    assert np.allclose(R @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)
    assert np.allclose(_quat_rotate(q, np.array([1.0, 0.0, 0.0])),
                       [0.0, 1.0, 0.0], atol=1e-12)


def test_dfbi_angular_updates_orientation():
    b = DfbiBody(mass=1.0, inertia=(1.0, 1.0, 1.0),
                 angular_velocity=(0.0, 0.0, np.pi / 2.0))
    b.advance(np.zeros(3), np.zeros(3), dt=1.0)
    # 角速度使姿态离开初始四元数
    assert not np.allclose(b.quaternion, [1.0, 0.0, 0.0, 0.0], atol=1e-6)


# ---------------------------------------------------------------- overset
def test_overset_donor_weight_hard_switch():
    W = overset_donor_weights(np.array([[0.0, 0.0, 0.0]]),
                              np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), k=3)
    assert W.shape == (1, 2)
    assert np.allclose(W.sum(axis=1), 1.0)
    assert np.allclose(W[0], [1.0, 0.0], atol=1e-12)


def test_overset_interpolate_constant_field():
    rp = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    dc = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    vals = np.array([2.0, 2.0, 2.0])
    out = overset_interpolate(rp, dc, vals, k=3)
    assert np.allclose(out, [2.0, 2.0], atol=1e-12)


# ---------------------------------------------------------------- MotionSolver
def test_make_motion_aliases():
    fv = _fv(2)
    assert isinstance(make_motion(fv, "motion"), MotionSolver)
    assert isinstance(make_motion(fv, "MRF"), MotionSolver)
    assert isinstance(make_motion(fv, "6dof"), MotionSolver)
    assert isinstance(make_motion(fv, "morphing"), MotionSolver)
    assert isinstance(make_motion(fv, "overset"), MotionSolver)
    assert isinstance(make_motion(fv, "rotating-reference-frame"), MotionSolver)
    with pytest.raises(ValueError):
        make_motion(fv, "nope")


def test_motion_solver_mrf_source_nonzero():
    fv = _fv(2)
    m = make_motion(fv, "motion", rotation_speed=2.0)
    src = m.mrf_source(rho=np.full(fv.n_cells, 1.0))
    assert src.shape == (fv.n_cells, 3)
    assert float(_safe_norm_src(src)) > 0.0


def _safe_norm_src(src):
    return float(np.linalg.norm(np.asarray(src, float).ravel()))


def test_motion_solver_update_advances_t():
    fv = _fv(2)
    m = make_motion(fv, "motion", rotation_speed=1.0)
    t0 = m._t
    m.update(np.zeros(fv.n_cells), np.zeros(fv.n_cells), np.zeros(fv.n_cells),
             mdot=None)
    assert m._t > t0
    assert np.isfinite(m.residual())


def test_motion_solver_facade_payload():
    fv = _fv(2)
    m = make_motion(fv, "dfbi", rotation_speed=1.0)
    init = m._initialize_field()
    assert "mode" in init
    st = m.step()
    assert "residual" in st and np.isfinite(st["residual"])
    mon = m.monitor_payload()
    assert {"mode", "n_bodies", "rotation_speed", "t", "iteration"} <= set(mon)


# ---------------------------------------------------------------- 集成
def test_pressure_solver_motion_coupling():
    V, C = cube_tet_mesh(2)
    s = PressureSolver(V, C, mu=1.0e-3, rho=998.0, inlet_velocity=(1.0, 0.0, 0.0),
                       motion_model="motion", motion_rotation_speed=0.5,
                       motion_rotation_axis=(0.0, 0.0, 1.0))
    assert s.motion_model is not None
    assert not isinstance(s.motion_model, str)
    for _ in range(6):
        p = s.step()
    assert np.all(np.isfinite(s.velocity()))
    assert np.isfinite(p["residual"])
    m = s.monitor_payload()
    assert {"motion_mode", "motion_rotation_speed", "motion_disp_max"} <= set(m)


def test_pressure_solver_motion_zero_speed_stable():
    # 转速为 0 → mrf 源为零，行为接近基线但持有 motion 键
    V, C = cube_tet_mesh(2)
    s = PressureSolver(V, C, mu=1e-3, inlet_velocity=(1.0, 0.0, 0.0),
                       motion_model="motion", motion_rotation_speed=0.0)
    for _ in range(5):
        p = s.step()
    assert np.isfinite(p["residual"])
    m = s.monitor_payload()
    assert "motion_mode" in m


def test_pressure_solver_without_motion_baseline():
    V, C = cube_tet_mesh(2)
    s = PressureSolver(V, C, mu=1e-3, inlet_velocity=(1.0, 0.0, 0.0))
    assert s.motion_model is None
    for _ in range(5):
        p = s.step()
    assert np.isfinite(p["residual"])
    m = s.monitor_payload()
    assert "motion_mode" not in m

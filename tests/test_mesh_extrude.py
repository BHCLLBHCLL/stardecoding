# -*- coding: utf-8 -*-
"""N 波 N5（三）extruder（线性/旋转）+ directed 网格验收（parity_100pct_plan.md 第 N5 节）。

覆盖（mesh_extrude 模块 docstring 验收要点）：
  线性：平板 distances 精确回显、体积=面积×总厚 精确、几何级数便捷参数
  与显式 distances 等价；非法 distances/方向 ValueError；
  旋转：跨 z 方形绕 z 轴 2π@16 层 体积≈π（Pappus V=A·θ·d，5%）、
  π/2@8 层 ≈π/4（2%）、层间角度均分；
  directed：平行板 0.3 间距 fill 模式体积=0.3 精确且顶面恰在目标面、
  固定模式体积=几何级数总厚、目标未覆盖源足迹 → ValueError。

纯 numpy，两环境（默认 / conda occ）皆可跑。
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

from mesh_io import cube_mesh
from mesh_extrude import (extrude_mesh, extrude_rotate, directed_mesh,
                          _geometric_levels)


# ---------------------------------------------------------------- 形状构造
def flat_plate(side=1.0, h=0.0):
    """源表面：单位方形平板 z=h（面绕向朝 +z）。"""
    V = [[0, 0, h], [side, 0, h], [side, side, h], [0, side, h]]
    F = [[0, 1, 2], [0, 2, 3]]
    return V, F


def source_plate(side=1.0):
    """方形源面（供 directed 发射）。"""
    V = [[0, 0, 0], [side, 0, 0], [side, side, 0], [0, side, 0]]
    F = [[0, 1, 2], [0, 2, 3]]
    return V, F


def target_plate(z=0.3, side=1.0):
    """目标面：方形平板 z=z。"""
    V = [[0, 0, z], [side, 0, z], [side, side, z], [0, side, z]]
    F = [[0, 1, 2], [0, 2, 3]]
    return V, F


# ------------------------------------------------------- 线性 extruder
def test_linear_exact_distances_and_volume():
    """平板 distances 精确回显、体积=面积×总厚 精确。"""
    V, F = flat_plate(1.0)
    dists = [0.1, 0.25, 0.5]
    r = extrude_mesh(V, F, distances=dists, direction=(0, 0, 1))
    assert r["ok"]
    assert r["levels"] == pytest.approx(dists)
    # 单位平板面积=1，总厚=0.5 → 体积=0.5
    assert r["volume_total"] == pytest.approx(0.5, rel=1e-6)
    assert r["n_cells"] == 6                 # 2 面 × 3 层
    assert r["quality"]["n_inverted"] == 0


def test_linear_geometric_equals_explicit():
    """几何级数便捷参数与显式 distances 等价。"""
    V, F = flat_plate(1.0)
    n = 4
    t0 = 0.05
    stretch = 1.5
    r_geo = extrude_mesh(V, F, n_layers=n, first_thickness=t0, stretch=stretch,
                         direction=(0, 0, 1))
    r_exp = extrude_mesh(V, F, distances=_geometric_levels(n, t0, stretch),
                         direction=(0, 0, 1))
    assert r_geo["levels"] == pytest.approx(r_exp["levels"])
    assert r_geo["volume_total"] == pytest.approx(r_exp["volume_total"],
                                                  rel=1e-6)
    assert np.allclose(r_geo["vertices"], r_exp["vertices"])


def test_linear_invalid_inputs_raise():
    """非法 distances/方向抛 ValueError。"""
    V, F = flat_plate(1.0)
    with pytest.raises(ValueError):
        extrude_mesh(V, F, distances=[], direction=(0, 0, 1))
    with pytest.raises(ValueError):
        extrude_mesh(V, F, distances=[0.1, 0.05], direction=(0, 0, 1))
    with pytest.raises(ValueError):
        extrude_mesh(V, F, distances=[0.1, 0.1], direction=(0, 0, 1))
    with pytest.raises(ValueError):
        extrude_mesh(V, F, distances=[0.1], direction=(0, 0, 0))
    with pytest.raises(ValueError):
        extrude_mesh([], [], distances=[0.1], direction=(0, 0, 1))


# ------------------------------------------------------- 旋转 extruder
def radial_face():
    """跨 z 的方形面（x∈[0,1], y=0, z∈[0,1]），绕 z 轴旋转生成实心圆柱。

    面质心 (0.5,0,0.5)，质心到 z 轴距离 d=0.5，面积 A=1；Pappus
    V = A·θ·d：整圈 θ=2π → V=π（半径1 高1 圆柱），四分之一 → π/4。
    """
    V = [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]]
    F = [[0, 1, 2], [0, 2, 3]]
    return V, F


def test_rotate_full_turn_pappus_cylinder():
    """跨 z 方形绕 z 轴 2π@16 层：Pappus V=A·θ·d=π（圆柱，5%）。"""
    V, F = radial_face()
    r = extrude_rotate(V, F, axis_point=(0, 0, 0), axis_dir=(0, 0, 1),
                       angle=2 * np.pi, n_layers=16)
    assert r["ok"]
    assert r["n_layers"] == 16
    # A=1, θ=2π, d=质心到轴距离=√(0.5²+0²)=0.5 → V=π
    expected = 1.0 * (2 * np.pi) * 0.5
    assert r["volume_total"] == pytest.approx(expected, rel=0.05)


def test_rotate_quarter_turn_pappus():
    """跨 z 方形绕 z 轴 π/2@8 层：Pappus V=A·θ·d=π/4（2%）。"""
    V, F = radial_face()
    r = extrude_rotate(V, F, axis_point=(0, 0, 0), axis_dir=(0, 0, 1),
                       angle=np.pi / 2, n_layers=8)
    assert r["ok"]
    expected = 1.0 * (np.pi / 2) * 0.5           # = π/4 ≈ 0.785
    assert r["volume_total"] == pytest.approx(expected, rel=0.02)
    # 层间角度均分
    angles = np.asarray(r["angles"])
    assert len(angles) == 8
    np.testing.assert_allclose(np.diff(angles), (np.pi / 2) / 8, rtol=1e-9)


# ------------------------------------------------------- directed
def test_directed_fill_volume_exact():
    """平行板 0.3 间距 fill 模式体积=0.3 精确且顶面恰在目标面。"""
    srcV, srcF = source_plate(1.0)
    tgtV, tgtF = target_plate(0.3, 1.0)
    r = directed_mesh(srcV, srcF, direction=(0, 0, 1), target_vertices=tgtV,
                      target_faces=tgtF, n_layers=3, first_thickness=0.05,
                      stretch=1.5, fill_to_target=True)
    assert r["ok"]
    # 单位平板面积=1，目标间距=0.3 → 体积=0.3（fill 直达目标）
    assert r["volume_total"] == pytest.approx(0.3, rel=1e-6)
    # 顶面恰在目标面 z=0.3
    V = np.asarray(r["vertices"])
    top = V[V[:, 2] > 0.29]
    assert np.allclose(top[:, 2], 0.3, atol=1e-6)
    assert r["quality"]["n_inverted"] == 0


def test_directed_fixed_mode_geometric():
    """固定模式体积=几何级数总厚（目标足够厚时不截断）。"""
    srcV, srcF = source_plate(1.0)
    tgtV, tgtF = target_plate(5.0, 1.0)          # 目标远，固定层深不截断
    n = 3
    t0 = 0.05
    stretch = 1.5
    r = directed_mesh(srcV, srcF, direction=(0, 0, 1), target_vertices=tgtV,
                      target_faces=tgtF, n_layers=n, first_thickness=t0,
                      stretch=stretch, fill_to_target=False)
    assert r["ok"]
    total = _geometric_levels(n, t0, stretch)[-1]
    # 固定模式：层深 = levels×min(1, h_v/total)，h_v=5 ≫ total → scale=1
    assert r["volume_total"] == pytest.approx(1.0 * total, rel=1e-6)


def test_directed_miss_raises():
    """目标未覆盖源足迹 → ValueError。"""
    srcV, srcF = source_plate(1.0)
    # 目标面只覆盖 x∈[2,3]，[0,1] 方形发射无命中
    tgtV = [[2, 0, 0.3], [3, 0, 0.3], [3, 1, 0.3], [2, 1, 0.3]]
    tgtF = [[0, 1, 2], [0, 2, 3]]
    with pytest.raises(ValueError):
        directed_mesh(srcV, srcF, direction=(0, 0, 1), target_vertices=tgtV,
                      target_faces=tgtF, n_layers=3, fill_to_target=True)

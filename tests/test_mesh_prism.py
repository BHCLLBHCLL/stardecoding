# -*- coding: utf-8 -*-
"""N 波 N4 验收：prism 边界层推进 + 碰撞处理（mesh_prism.py）。

覆盖：
  vertex_normals：平板纯 +z、立方体角点法向沿对角指向体内侧；
  ray_hit_distance：立方体内部射线命中距离精确、背离方向无命中；
  平板全层（side="+n"）：层数满、几何级数累计层厚、体积=面积×总厚 精确、
    首层质心高 = t0/2（y+ 控制旋钮）；
  立方体满层（side="inward" 自动定向）：12 面全满层、推进点全部留在体内；
  窄通道碰撞：gap_fraction=0.9 超比例自动降层、=0.5 两侧提升点严格分离
  （无重叠）、=0.01 全塌缩；
  质量：prism_quality 翻转数/符号一致性/体积统计；空输入/开放表面 ValueError。

纯 numpy，两环境皆可跑。
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

from mesh_io import cube_mesh, transform_vertices
from mesh_prism import (vertex_normals, ray_hit_distance, prism_layers,
                        prism_quality)

# 平板（开放表面，绕向法向 +z）
PLATE_V = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
PLATE_F = [(0, 1, 2), (0, 2, 3)]


def narrow_box():
    """窄通道盒 [0,1]×[0,0.8]×[0,0.15]（x/y 不对称避免射线退化命中）。"""
    V, F = cube_mesh(1.0)
    return transform_vertices(V, scale=(1.0, 0.8, 0.15)), F


# ---------------------------------------------------------------- 法向与射线
def test_vertex_normals_plate_and_cube():
    n = vertex_normals(PLATE_V, PLATE_F)
    assert np.allclose(n, [0.0, 0.0, 1.0])          # 平板纯 +z
    V, F = cube_mesh(1.0)
    n = vertex_normals(V, F)
    # 角点 (0,0,0) 法向三分量同号（绕向一致指向体内对角）
    assert np.all(n[0] > 0.0) or np.all(n[0] < 0.0)
    assert np.isclose(float(np.linalg.norm(n[0])), 1.0)


def test_ray_hit_distance_cube():
    V, F = cube_mesh(1.0)
    assert ray_hit_distance((0.3, 0.6, -1.0), (0, 0, 1), V, F) == \
        pytest.approx(1.0, rel=1e-12)
    assert ray_hit_distance((2.0, 0.6, 0.3), (-1, 0, 0), V, F) == \
        pytest.approx(1.0, rel=1e-12)
    assert ray_hit_distance((0.3, 0.6, -1.0), (0, 0, -1), V, F) == \
        float("inf")


# ---------------------------------------------------------------- 平板全层
def test_prism_plate_full_layers_exact():
    out = prism_layers(PLATE_V, PLATE_F, n_layers=3, first_thickness=0.05,
                       stretch=1.5, gap_fraction=0.5, side="+n")
    assert out["ok"]
    assert out["face_layers"] == [3, 3]
    assert out["n_cells"] == 6                       # 2 面 × 3 层
    # 几何级数累计层厚 t0·(r^k−1)/(r−1)
    assert out["levels"] == pytest.approx([0.05, 0.125, 0.2375], rel=1e-12)
    assert out["total_thickness"] == pytest.approx(0.2375, rel=1e-12)
    # 体积 = 平板面积(1.0) × 总厚（平壁精确）
    assert out["quality"]["volume_total"] == pytest.approx(0.2375, rel=1e-9)
    assert out["quality"]["volume_min"] > 0.0
    # 首层厚度 = t0（平壁精确）
    assert out["first_layer_thickness"] == pytest.approx(0.05, rel=1e-9)


def test_prism_plate_first_cell_centroid_yplus():
    """y+ 控制旋钮：t0 直接决定首层单元高度，首层质心高 = t0/2。"""
    t0 = 0.05
    out = prism_layers(PLATE_V, PLATE_F, n_layers=3, first_thickness=t0,
                       stretch=1.5, side="+n")
    V, C = out["vertices"], out["cells"]
    first = C[C[:, 0] < 4]                           # 底面用原始顶点 = 首层
    assert len(first) == 2
    for cell in first:
        cen = V[cell].mean(axis=0)
        assert cen[2] == pytest.approx(t0 / 2.0, rel=1e-9)
        assert cen[0] > 0.0 and cen[1] > 0.0         # 在板上方推进


# ---------------------------------------------------------------- 立方体满层
def test_prism_cube_full_layers():
    V0, F0 = cube_mesh(1.0)
    out = prism_layers(V0, F0, n_layers=3, first_thickness=0.02,
                       stretch=1.5, gap_fraction=0.5, side="inward")
    assert out["ok"]
    assert out["face_layers"] == [3] * 12            # 全部满层
    assert out["n_full"] == 12 and out["n_reduced"] == 0
    assert out["n_cells"] == 36
    # 推进点全部留在体内
    lifted = out["vertices"][len(V0):]
    assert len(lifted) == 8 * 3
    assert lifted[:, 0].min() >= -1e-9 and lifted[:, 0].max() <= 1 + 1e-9
    assert lifted[:, 1].min() >= -1e-9 and lifted[:, 1].max() <= 1 + 1e-9
    assert lifted[:, 2].min() >= -1e-9 and lifted[:, 2].max() <= 1 + 1e-9
    # 角点法向倾斜 → 首层厚度均值 < t0 但同量级
    assert 0.5 * 0.02 < out["first_layer_thickness"] <= 0.02 * (1 + 1e-9)


# ---------------------------------------------------------------- 窄通道碰撞
def test_prism_collision_narrow_gap_reduces_layers():
    V, F = narrow_box()                              # 高 0.15
    out = prism_layers(V, F, n_layers=3, first_thickness=0.05,
                       stretch=1.5, gap_fraction=0.9, side="inward")
    # 顶/底面通道 0.15：d3=0.2375 超允许深度(0.9×0.156) → 全部降为 2 层
    assert out["face_layers"] == [2] * 12
    assert out["n_full"] == 0 and out["n_reduced"] == 12
    assert out["n_cells"] == 24
    # 降层后提升点仍在体内
    lifted = out["vertices"][len(V):]
    assert lifted[:, 2].min() >= -1e-9
    assert lifted[:, 2].max() <= 0.15 + 1e-9


def test_prism_half_gap_no_overlap():
    V, F = narrow_box()
    out = prism_layers(V, F, n_layers=3, first_thickness=0.05,
                       stretch=1.5, gap_fraction=0.5, side="inward")
    # 允许深度 = 0.5×0.156 ≈ 0.078：仅 d1=0.05 满足 → 全部 1 层
    assert out["face_layers"] == [1] * 12
    assert out["n_cells"] == 12
    # 两侧提升点严格分离（各占通道半侧，中面 z=0.075）
    lifted = out["vertices"][len(V):]
    zs = lifted[:, 2]
    assert zs.min() >= 0.04 and zs.max() <= 0.11     # 分离带 (0.048, 0.102)
    assert out["quality"]["n_inverted"] == 0


def test_prism_tiny_gap_collapses():
    V, F = narrow_box()
    out = prism_layers(V, F, n_layers=3, first_thickness=0.05,
                       stretch=1.5, gap_fraction=0.01, side="inward")
    assert out["face_layers"] == [0] * 12            # 全塌缩
    assert out["n_collapsed"] == 12
    assert out["n_cells"] == 0
    assert not out["ok"]
    assert not out["quality"]["ok"]


# ---------------------------------------------------------------- 质量与守门
def test_prism_quality_metrics_plate():
    out = prism_layers(PLATE_V, PLATE_F, n_layers=3, first_thickness=0.05,
                       stretch=1.5, side="+n")
    q = out["quality"]
    assert q["n_cells"] == 6
    assert q["n_inverted"] == 0
    assert q["sign_consistent"]
    assert q["volume_min"] == pytest.approx(0.5 * 0.05, rel=1e-9)  # 首层最小
    assert q["volume_total"] == pytest.approx(0.2375, rel=1e-9)
    # 独立复算：quality 与逐单元体积和一致
    vols = [abs(prism_quality(out["vertices"], out["cells"][i:i + 1])
                ["volume_total"]) for i in range(6)]
    assert sum(vols) == pytest.approx(q["volume_total"], rel=1e-9)


def test_prism_input_gates():
    with pytest.raises(ValueError):
        prism_layers([], [])                          # 空输入
    with pytest.raises(ValueError):
        # 开放表面无法自动定向内外
        prism_layers(PLATE_V, PLATE_F, side="inward")
    with pytest.raises(ValueError):
        prism_layers(PLATE_V, PLATE_F, side="bad_side")

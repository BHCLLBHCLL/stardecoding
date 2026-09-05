# -*- coding: utf-8 -*-
"""N 波 N3b 验收（一）：poly = tet 的 Voronoi 对偶（mesh_poly.py）。

覆盖：
  clip_convex：盒被平面裁剪（半盒体积/盖帽面）/ 完全裁空；
  poly_volume：盒体积精确 = 宽×高×深；
  voronoi_dual 均匀栅格：内部单元 ≈ h³ 盒（体积/6 面）、全体单元铺满包围盒
  （体积和 == 盒体积，clipped Voronoi 不变量）；
  poly_mesh 立方体表面工作流：体积和 ≈ 1.0、全部单元正体积；
  空输入 ValueError。

纯 numpy + scipy（Delaunay），两环境皆可跑（无 scipy 的用例已 skip）。
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

from mesh_io import cube_mesh
from mesh_poly import (box_polyhedron, clip_convex, poly_volume,
                       voronoi_dual, poly_mesh)


def _has(name):
    try:
        __import__(name)
        return True
    except Exception:
        return False


# ------------------------------------------------------------- 裁剪核心
def test_clip_half_box():
    V, F = box_polyhedron([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    V2, F2 = clip_convex(V, F, [0.5, 0.0, 0.0], [1.0, 0.0, 0.0])  # 保留 x≤0.5
    assert len(F2) >= 6                       # 6 原面 + 1 盖帽 ≥ 7（退化允许 6）
    assert poly_volume(V2, F2) == pytest.approx(0.5, rel=1e-9)


def test_clip_empty_when_fully_outside():
    V, F = box_polyhedron([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    # 法向取 -x：保留侧为 x ≥ 2 → 盒 [0,1] 整体在外
    V2, F2 = clip_convex(V, F, [2.0, 0.0, 0.0], [-1.0, 0.0, 0.0])
    assert not F2
    assert poly_volume(V2, F2) == 0.0


def test_clip_two_planes_quarter():
    V, F = box_polyhedron([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    V, F = clip_convex(V, F, [0.5, 0.0, 0.0], [1.0, 0.0, 0.0])
    V, F = clip_convex(V, F, [0.0, 0.5, 0.0], [0.0, 1.0, 0.0])
    assert poly_volume(V, F) == pytest.approx(0.25, rel=1e-9)


def test_box_volume_exact():
    V, F = box_polyhedron([0.0, 0.0, 0.0], [2.0, 3.0, 5.0])
    assert poly_volume(V, F) == pytest.approx(30.0, rel=1e-12)


# ------------------------------------------------------------- Voronoi 对偶
def _grid_tets(n=4):
    """n×n×n 均匀栅格（间距 1）的 scipy Delaunay 四面体化（QJ 防退化）。"""
    from scipy.spatial import Delaunay
    xs = np.arange(n, dtype=float)
    pts = np.array([[x, y, z] for x in xs for y in xs for z in xs])
    tri = Delaunay(pts, qhull_options="QJ")
    return pts, np.asarray(tri.simplices, np.int64)


@pytest.mark.skipif(not _has("scipy"), reason="无 scipy")
def test_voronoi_grid_interior_cells_are_boxes():
    pts, tets = _grid_tets(4)
    out = voronoi_dual(pts, tets)
    assert out["ok"]
    # 内部点（坐标 ∈ {1,2} 三维内部）体积 ≈ 1³、面数 6（盒）
    interior = [c for c in out["cells"]
                if all(0.5 < pts[c["seed"]][k] < 2.5 for k in range(3))]
    assert len(interior) >= 8
    for c in interior:
        assert c["volume"] == pytest.approx(1.0, rel=1e-3)
        assert c["n_faces"] == 6
    # 全体单元正体积
    assert out["quality"]["volume_min"] > 0.0


@pytest.mark.skipif(not _has("scipy"), reason="无 scipy")
def test_voronoi_cells_tessellate_box():
    """clipped Voronoi 不变量：全体单元无重叠铺满包围盒。"""
    pts, tets = _grid_tets(4)
    out = voronoi_dual(pts, tets)
    box_vol = 3.0 ** 3                       # 栅格 [0,3]³
    assert out["volume_total"] == pytest.approx(box_vol, rel=1e-6)
    assert out["n_cells"] == len(pts)        # 每个种子一个单元


@pytest.mark.skipif(not _has("scipy"), reason="无 scipy")
def test_voronoi_dual_rejects_empty():
    with pytest.raises(ValueError):
        voronoi_dual(np.zeros((0, 3)), np.zeros((0, 4), np.int64))


# ------------------------------------------------------------- 完整工作流
@pytest.mark.skipif(not _has("scipy"), reason="无 scipy")
def test_poly_mesh_cube_volume():
    V, F = cube_mesh(1.0)
    out = poly_mesh(V, F, spacing=0.5)
    assert out["ok"]
    assert out["method"].startswith("voronoi_dual")
    assert out["n_cells"] > 0
    assert out["volume_total"] == pytest.approx(1.0, rel=2e-2)
    assert out["quality"]["volume_min"] > 0.0
    assert out["quality"]["faces_mean"] >= 4.0


@pytest.mark.skipif(not _has("scipy"), reason="无 scipy")
def test_poly_mesh_rejects_open_surface():
    V, F = cube_mesh(1.0)
    with pytest.raises(ValueError):
        poly_mesh(V, F[:-1], spacing=0.5)

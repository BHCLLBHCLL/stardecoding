# -*- coding: utf-8 -*-
"""N 波 N3b 验收（二）：trimmer 八叉树切割网格器（mesh_trimmer.py）。

覆盖：
  立方体体积守恒 ≈ 1.0（轴对齐面精确切割、六面体+切割单元并存）；
  球体积 ≈ 4π/3（凸贴体裁剪，分辨率容差内）；
  八叉树分级：近表面叶 = 目标尺寸、内部存在更粗叶；
  全部单元正体积；开表面 ValueError。

纯 numpy，两环境皆可跑。
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

from mesh_io import cube_mesh, _weld
from mesh_trimmer import (trimmer_mesh, trimmer_cell_volume,
                          trimmer_quality, _leaf_target_depth)


def unit_sphere_mesh(lat=12, lon=16, r=1.0):
    """UV 球（半径 r），顶点焊接后水密（极点重合顶点合并）。"""
    def sph(t, p):
        return [r * np.sin(t) * np.cos(p), r * np.sin(t) * np.sin(p),
                r * np.cos(t)]
    soup = []
    V = []
    for i in range(lat + 1):
        t = np.pi * i / lat
        for j in range(lon):
            p = 2.0 * np.pi * j / lon
            V.append(sph(t, p))
    F = []
    for i in range(lat):
        for j in range(lon):
            a = i * lon + j
            b = i * lon + (j + 1) % lon
            c = (i + 1) * lon + j
            d = (i + 1) * lon + (j + 1) % lon
            F.append((a, b, c))
            F.append((b, d, c))
    for tri in F:
        for k in tri:
            soup.append(V[k])
    return _weld(soup)


def _leaf_size(cell):
    if cell["kind"] == "hex":
        return float(np.max(np.asarray(cell["hi"], float)
                            - np.asarray(cell["lo"], float)))
    vv = np.asarray(cell["vertices"], float)
    return float(np.max(vv.max(axis=0) - vv.min(axis=0)))


# ---------------------------------------------------------------- 立方体
def test_trimmer_cube_volume_conservation():
    V, F = cube_mesh(1.0)
    out = trimmer_mesh(V, F, cell_size=0.25)
    assert out["ok"]
    assert out["n_hex"] > 0
    assert out["n_cut"] > 0
    # 轴对齐面：切割平面与叶面重合 → 体积精确守恒
    assert out["volume_total"] == pytest.approx(1.0, rel=1e-9)
    assert out["quality"]["volume_min"] > 0.0


def test_trimmer_cube_refine_level():
    V, F = cube_mesh(1.0)
    out = trimmer_mesh(V, F, cell_size=0.25)
    sizes = [_leaf_size(c) for c in out["cells"]]
    # 表面 = 包围盒 → 所有叶都触面、全部细分到目标尺寸 0.25
    assert all(abs(s - 0.25) < 1e-9 for s in sizes)


def test_trimmer_sphere_octree_hierarchy():
    V, F = unit_sphere_mesh(lat=12, lon=16, r=1.0)
    out = trimmer_mesh(V, F, cell_size=0.3)     # root=2 → depth3 → 叶 0.25
    sizes = [_leaf_size(c) for c in out["cells"]]
    # 近表面最深叶 = 0.25；球心区域存在 0.5 粗叶（八叉树分级加密）
    assert any(abs(s - 0.25) < 1e-9 for s in sizes)
    assert any(abs(s - 0.5) < 1e-9 for s in sizes)


def test_trimmer_cube_cut_cell_volume():
    V, F = cube_mesh(1.0)
    out = trimmer_mesh(V, F, cell_size=0.25)
    # 每个切割单元可独立复算体积（多面体散度定理）
    for c in out["cells"]:
        if c["kind"] == "cut":
            v = trimmer_cell_volume(c)
            assert v > 0.0
            assert v <= (0.25 ** 3) * (1.0 + 1e-9)   # 不超叶盒体积


# ---------------------------------------------------------------- 球
def test_trimmer_sphere_volume():
    V, F = unit_sphere_mesh(lat=12, lon=16, r=1.0)
    out = trimmer_mesh(V, F, cell_size=0.3)
    assert out["ok"]
    target = 4.0 / 3.0 * np.pi
    # 凸贴体裁剪：体积在八叉树/表面离散分辨率容差内逼近 4π/3
    assert out["volume_total"] == pytest.approx(target, rel=0.05)


def test_trimmer_sphere_all_positive_and_kinds():
    V, F = unit_sphere_mesh(lat=12, lon=16, r=1.0)
    out = trimmer_mesh(V, F, cell_size=0.3)
    vols = [trimmer_cell_volume(c) for c in out["cells"]]
    assert all(v > 0.0 for v in vols)
    assert out["n_hex"] > 0
    assert out["n_cut"] > 0
    q = trimmer_quality(out["cells"])
    assert q["n_cells"] == len(out["cells"])
    assert q["ok"]


# ---------------------------------------------------------------- 深度与守门
def test_leaf_target_depth():
    d = _leaf_target_depth([0, 0, 0], [1, 1, 1], 0.25)
    assert d == 2                       # 1 → 0.5 → 0.25
    d = _leaf_target_depth([0, 0, 0], [1, 1, 1], 4.0)
    assert d >= 1                       # 粗目标至少细分一级
    d = _leaf_target_depth([0, 0, 0], [1, 1, 1], 1e-9)
    assert d <= 10                      # 深度封顶


def test_trimmer_rejects_open_surface():
    V, F = cube_mesh(1.0)
    with pytest.raises(ValueError):
        trimmer_mesh(V, F[:-1], cell_size=0.25)

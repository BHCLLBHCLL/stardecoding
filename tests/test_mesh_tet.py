# -*- coding: utf-8 -*-
"""R1-2 非凸域四面体网格验收（parity_100pct_plan.md R 波教程工况前置能力）。

覆盖：
  1) scipy Delaunay 单元定向统一：体网格无负体积单元（历史缺陷：静默丢半）
  2) 凸域体积守恒：单位立方体 sum_vol == 1.0（零回归）
  3) 非凸域（L 形棱柱）凹腔裁剪：体积守恒 ≈ 3.0，质心全在域内
  4) 批量 even-odd `_points_in_domain` 与标量 `point_in_mesh` 逐点一致
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

from mesh_io import cube_mesh
from mesh_tet import (tet_mesh, point_in_mesh, _points_in_domain,
                      _tet_volumes)


def _has(name):
    try:
        __import__(name)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has("scipy"), reason="无 scipy")


def _l_prism():
    """L 形棱柱：2x2 方柱挖去右上 1x1 → 截面积 3、凸包 4，高 1、体积 3。"""
    poly = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0),
            (1.0, 1.0), (1.0, 2.0), (0.0, 2.0)]
    n = len(poly)
    V = [(x, y, z) for z in (0.0, 1.0) for x, y in poly]
    F = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5)]
    F += [(n + 2, n + 1, n + 0), (n + 3, n + 2, n + 0),
          (n + 4, n + 3, n + 0), (n + 5, n + 4, n + 0)]
    for i in range(n):
        j = (i + 1) % n
        F += [(i, j, n + j), (i, n + j, n + i)]
    return np.array(V, float), np.array(F, np.int64)


def _sum_vol(out):
    return float(_tet_volumes(out["vertices"], out["cells"]).sum())


def test_points_in_domain_matches_scalar_convex():
    V, F = cube_mesh(1.0)
    rng = np.random.default_rng(0)
    P = rng.uniform(-0.3, 1.3, size=(300, 3))
    ref = np.array([point_in_mesh(p, V, F) for p in P])
    assert np.array_equal(ref, _points_in_domain(P, V, F))


def test_points_in_domain_matches_scalar_nonconvex():
    V, F = _l_prism()
    rng = np.random.default_rng(1)
    P = rng.uniform(-0.3, 2.3, size=(300, 3))
    ref = np.array([point_in_mesh(p, V, F) for p in P])
    assert np.array_equal(ref, _points_in_domain(P, V, F))
    # 凹腔中心（右上 1x1 柱内）在域外；L 形实体内的点判为域内
    assert not _points_in_domain([[1.5, 1.5, 0.5]], V, F)[0]
    assert _points_in_domain([[0.5, 0.5, 0.5],
                              [1.5, 0.5, 0.5],
                              [0.5, 1.5, 0.5]], V, F).all()


def test_tet_convex_cube_volume_conserved():
    V, F = cube_mesh(1.0)
    out = tet_mesh(V, F, spacing=0.3, method="scipy")
    assert out["ok"]
    assert out["quality"]["n_negative"] == 0
    assert _sum_vol(out) == pytest.approx(1.0, abs=1e-9)


def test_tet_nonconvex_l_prism_volume_and_inside():
    V, F = _l_prism()
    out = tet_mesh(V, F, spacing=0.25, method="scipy")
    assert out["ok"]
    assert out["quality"]["n_negative"] == 0
    tot = _sum_vol(out)
    assert 0.95 * 3.0 <= tot <= 1.001 * 3.0
    cent = out["vertices"][out["cells"]].mean(axis=1)
    assert _points_in_domain(cent, V, F).all()


def test_tet_orientation_normalized_all_positive():
    V, F = _l_prism()
    out = tet_mesh(V, F, spacing=0.3, method="scipy")
    vol = _tet_volumes(out["vertices"], out["cells"])
    assert len(vol) > 0 and (vol > 0).all()

# -*- coding: utf-8 -*-
"""N 波 N2c Delaunay/前沿推进验收（parity_100pct_plan.md 第 6 节 N2 B 路线）。

覆盖：
  bowyer_watson：2D 点云增量 Delaunay —— 空外接圆（is_delaunay2d 违规=0）且覆盖凸包；
  delaunay_flip：内蕴角准则 α+β≤π —— 非 Delaunay 凸四环翻转后违规归零；
  advancing_front：2D 前沿推进 —— 域覆盖(面积||target)、全正面积、边界保持、
    带内孔(环域)覆盖；
  remesh_wavy_advancing：图曲面 z=f(x,y) 前沿推进后按 z 提升。

Delaunay 一致表面重网格化 `delaunay_refine_surface` 的水密/边界属性由
tests/test_mesh_remesh.py（N2b remesh_surface）与 tetra/3D 场景覆盖。

纯 numpy，两环境（默认 / conda occ）皆可跑。
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

from mesh_front import (bowyer_watson, is_delaunay2d, delaunay_metric,
                        delaunay_flip, advancing_front,
                        remesh_wavy_advancing)


def tri_edges(F):
    e = set()
    for (a, b, c) in F:
        e.add((min(a, b), max(a, b)))
        e.add((min(b, c), max(b, c)))
        e.add((min(c, a), max(c, a)))
    return e


def edge_len(u, v):
    return float(np.linalg.norm(np.asarray(u, float) - np.asarray(v, float)))


def poly_area(poly):
    s = 0.0
    for i in range(len(poly)):
        p0, p1 = np.asarray(poly[i], float), np.asarray(poly[(i + 1) % len(poly)], float)
        s += p0[0] * p1[1] - p0[1] * p1[0]
    return 0.5 * s


def mesh_area(V, F):
    return sum(poly_area([V[i], V[j], V[k]]) for i, j, k in F)


# ---------------------------------------------------------------- 形状构造
SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


# ================================================================ Bowyer-Watson
def test_bowyer_watson_empty_circumcircle():
    rng = np.random.default_rng(7)
    pts = rng.uniform(0.0, 1.0, (120, 2))
    tris = bowyer_watson(pts)
    assert len(tris) >= len(pts)  # 充分覆盖凸包
    assert is_delaunay2d(pts, tris) == 0


def test_bowyer_watson_covers_hull():
    pts = SQUARE
    tris = bowyer_watson(pts)
    area = mesh_area(pts, tris)
    assert area == pytest.approx(1.0, abs=1e-6)


# ================================================================ Delaunay flip
def test_delaunay_flip_repairs_non_delaunay_quad():
    # 凸四环：A(0,0) B(3,0) C(1,1) D(0,2)，取对角 AC 的三角剖分为非 Delaunay。
    quad = np.array([[0.0, 0.0], [3.0, 0.0], [1.0, 1.0], [0.0, 2.0]])
    bad = [(0, 1, 2), (0, 2, 3)]
    vio_init, n_int = delaunay_metric(quad, bad)
    assert n_int == 1
    assert vio_init == 1
    fixed = delaunay_flip(quad, bad)
    vio_final, _ = delaunay_metric(quad, fixed)
    assert vio_final == 0


# ================================================================ 前沿推进
def _boundary_missing(sq, target, V, F, R=11):
    """返回缺失的边界子边数。"""
    VS = [tuple(vP) for vP in V]
    idx = {v: i for i, v in enumerate(VS)}
    E = tri_edges(F)
    missing = 0
    for k in range(len(sq)):
        p0, p1 = np.asarray(sq[k], float), np.asarray(sq[(k + 1) % len(sq)], float)
        seg = p1 - p0
        n = max(1, int(np.ceil(edge_len(sq[k], sq[(k + 1) % len(sq)]) / target)))
        sub = [(round(float((p0 + (m / n) * seg)[0]), R),
                round(float((p0 + (m / n) * seg)[1]), R)) for m in range(n)]
        sub.append((round(float(p1[0]), R), round(float(p1[1]), R)))
        for m in range(len(sub) - 1):
            a, b = idx[sub[m]], idx[sub[m + 1]]
            if (min(a, b), max(a, b)) not in E:
                missing += 1
    return missing


@pytest.mark.parametrize("target", [0.5, 0.3, 0.2, 0.1])
def test_advancing_front_coverage_and_positive(target):
    V, F = advancing_front(SQUARE, (), target, 50000)
    area = mesh_area(V, F)
    assert area == pytest.approx(1.0, rel=0.05)      # 覆盖 ≈ 域面积
    for (i, j, k) in F:
        assert poly_area([V[i], V[j], V[k]]) > 0.0    # 全正面积


@pytest.mark.parametrize("target", [0.5, 0.2, 0.1])
def test_advancing_front_boundary_preserved(target):
    V, F = advancing_front(SQUARE, (), target, 50000)
    assert _boundary_missing(SQUARE, target, V, F) == 0


def test_advancing_front_annulus():
    outer = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    hole = [(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)]
    expected = abs(poly_area(outer)) - abs(poly_area(hole))  # 1 - 0.04 = 0.96
    V, F = advancing_front(outer, (hole,), 0.1, 50000)
    area = mesh_area(V, F)
    assert area == pytest.approx(expected, rel=0.05)
    for (i, j, k) in F:
        assert poly_area([V[i], V[j], V[k]]) > 0.0


def test_remesh_wavy_advancing_lifts_z():
    def z_fn(x, y):
        return 0.2 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
    V3, F3 = remesh_wavy_advancing(SQUARE, z_fn, 0.2, 50000)
    assert len(V3) > 4 and len(F3) > 0
    for i, (x, y, z) in enumerate(V3):
        assert z == pytest.approx(float(z_fn(x, y)), abs=1e-9)
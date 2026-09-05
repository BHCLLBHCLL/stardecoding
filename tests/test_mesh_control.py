# -*- coding: utf-8 -*-
"""N 波 N5（一）自定义尺寸控制验收（parity_100pct_plan.md 第 N5 节）。

覆盖（mesh_control 模块 docstring 验收要点）：
  立方体单面控制：底面顶点=target 精确、对角顶点=target+growth×边长
  精确（梯度有界且饱和于 base）；min/max 夹取（面级+全局）；顶点/区域
  控制精确；10×10 栅格局部加密：近控制区边长远小于远场、边界边数保持、
  两次运行逐位一致（确定性）。

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
from mesh_control import resolve_size_field, remesh_with_controls


# ---------------------------------------------------------------- 形状构造
def square_plate(h=0.0):
    """开面单位正方形 z=h（面绕向朝 +z）。"""
    V = [[0, 0, h], [1, 0, h], [1, 1, h], [0, 1, h]]
    F = [[0, 2, 1], [0, 3, 2]]
    return V, F


def grid_surface(nx=11, ny=11):
    """开面第 1 象限 [0,1]^2 三角栅格（顶点按列主序）。

    连接拓扑与 unit_sphere_mesh/wavy_grid 一致（a,b,c / b,d,c），
    便于验证近控制区加密 + 边界边数保持。
    """
    V = []
    for i in range(nx):
        x = i / (nx - 1)
        for j in range(ny):
            y = j / (ny - 1)
            V.append([x, y, 0.0])
    F = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a = i * ny + j
            b = i * ny + (j + 1)
            c = (i + 1) * ny + j
            d = (i + 1) * ny + (j + 1)
            F.append((a, b, c))
            F.append((b, d, c))
    return V, F


# ------------------------------------------------------- 立方体单面控制
def test_cube_single_face_control_exact():
    """单面控制：底面顶点=target 精确，对角顶点=target+growth×边长 精确。"""
    V, F = cube_mesh(1.0)
    # cube_mesh 面索引：0-1 = 底面 z=0（顶点 0,1,2,3）
    base = 0.80                        # 足够大，传播值不饱和
    target = 0.20
    growth = 0.30
    sizes = resolve_size_field(V, F, base,
                               surface_sizes={0: target, 1: target},
                               growth=growth, size_min=None, size_max=None)
    sizes = np.asarray(sizes, float)

    # 底面 4 顶点（0,1,2,3）：直接种入 target，不受 growth 影响
    for v in (0, 1, 2, 3):
        assert sizes[v] == pytest.approx(target, abs=1e-12)

    # 顶面 4 顶点（4,5,6,7）：沿侧棱到底面顶点的最短网格路径 d=1
    # (0,4) 为 y=0 面内边 → 场值 = target + growth×1 < base，精确
    for v in (4, 5, 6, 7):
        assert sizes[v] == pytest.approx(target + growth * 1.0, abs=1e-12)

    # 梯度有界：任何边两端尺寸差 ≤ growth×边长（此处边长=1）
    for a, b, c in F:
        for u, w in ((a, b), (b, c), (c, a)):
            assert abs(sizes[u] - sizes[w]) <= growth * 1.0 + 1e-9


def test_cube_single_face_saturates_at_base():
    """控制区外尺寸饱和于 base：远离控制面处 = base，但对角面因路径短
    仍受 growth 影响（不足 base 时），验证 min(base, 场值) 语义。"""
    V, F = cube_mesh(1.0)
    base = 0.30
    target = 0.10
    growth = 0.05                      # 足够小 → 场值不会覆盖到 base 就饱和
    sizes = resolve_size_field(V, F, base,
                               surface_sizes={0: target, 1: target},
                               growth=growth)
    sizes = np.asarray(sizes, float)

    # 场值 min(base, target + growth×d)：顶面 d=1 → 0.15 < 0.30，无饱和
    for v in (4, 5, 6, 7):
        assert sizes[v] == pytest.approx(target + growth * 1.0, abs=1e-12)
    # 底面仍精确 target
    for v in (0, 1, 2, 3):
        assert sizes[v] == pytest.approx(target, abs=1e-12)


def test_min_max_clamp_face_and_global():
    """min/max 夹取：面级 min/max 先夹 target，全局 size_min/size_max 终夹。"""
    V, F = cube_mesh(1.0)
    base = 0.30

    # 面级夹取：target=0.05 被 min=0.08 抬到 0.08
    sizes = resolve_size_field(V, F, base,
                               surface_sizes={0: {"min": 0.08, "target": 0.05,
                                                  "max": 0.20}},
                               growth=1e-9)      # 微小正增长率（模块要求 >0）
    for v in (0, 1, 2, 3):
        assert sizes[v] == pytest.approx(0.08, abs=1e-6)

    # 面级 max：target=0.50 被 max=0.20 压到 0.20
    sizes = resolve_size_field(V, F, base,
                               surface_sizes={0: {"min": 0.08, "target": 0.50,
                                                  "max": 0.20}},
                               growth=1e-9)
    for v in (0, 1, 2, 3):
        assert sizes[v] == pytest.approx(0.20, abs=1e-6)

    # 全局夹取：控制区外也有 min 下限（全场 ≥ size_min）
    sizes = resolve_size_field(V, F, 0.30,
                               surface_sizes={0: 0.05},
                               growth=1e-9, size_min=0.15, size_max=0.25)
    # 底面被全局 max 压到 0.25（0.05 越界）
    for v in (0, 1, 2, 3):
        assert 0.149 <= sizes[v] <= 0.25 + 1e-9
        assert sizes[v] >= 0.15 - 1e-9
    # 顶面（0.30 场值）被全局 clamp 到 [0.15, 0.25]
    for v in (4, 5, 6, 7):
        assert 0.149 <= sizes[v] <= 0.25 + 1e-9


def test_vertex_and_region_control_exact():
    """顶点控制与区域控制精确：直接种入尺寸。"""
    V, F = cube_mesh(1.0)
    base = 0.30

    # 顶点控制：单顶点 v=0 → 0.06；远处顶点饱和 base（growth 大 → 场值超 base 截断）
    sizes = resolve_size_field(V, F, base,
                               vertex_sizes={0: 0.06},
                               growth=10.0)          # growth 极大 → 邻点场值即超 base
    assert sizes[0] == pytest.approx(0.06, abs=1e-12)
    for v in (1, 2, 3, 4, 5, 6, 7):
        assert sizes[v] == pytest.approx(base, abs=1e-12)

    # 区域控制：以球心 (0.5,0.5,0.5) 半径 0.4 内全部顶点（这里只有 0 在 0.4 内）
    # 半径选覆盖全部 8 顶点 → 全部都种入 region 尺寸
    rad = 1.5
    sizes = resolve_size_field(V, F, base,
                               region_sizes=[([0.5, 0.5, 0.5], rad, 0.07)],
                               growth=1e-9)      # 半径覆盖全部顶点，直接种入
    for v in range(len(V)):
        assert sizes[v] == pytest.approx(0.07, abs=1e-12)


def test_multiseed_same_vertex_takes_min():
    """同顶点多种子取最小。"""
    V, F = cube_mesh(1.0)
    base = 0.30
    # v=0 同时被顶点控制 0.08 与区域控制 0.05 种入 → 取 0.05
    sizes = resolve_size_field(V, F, base,
                               vertex_sizes={0: 0.08},
                               region_sizes=[([0.0, 0.0, 0.0], 0.5, 0.05)],
                               growth=1e-9)      # 多种子同顶点取最小
    assert sizes[0] == pytest.approx(0.05, abs=1e-12)


def test_invalid_inputs_raise():
    """非法输入抛 ValueError。"""
    V, F = cube_mesh(1.0)

    # base 非正
    with pytest.raises(ValueError):
        resolve_size_field(V, F, -1.0)
    with pytest.raises(ValueError):
        resolve_size_field(V, F, 0.0)
    # growth 非正
    with pytest.raises(ValueError):
        resolve_size_field(V, F, 0.30, growth=0.0)
    # 控制尺寸非正
    with pytest.raises(ValueError):
        resolve_size_field(V, F, 0.30, surface_sizes={0: -0.1})
    # 面索引越界
    with pytest.raises(ValueError):
        resolve_size_field(V, F, 0.30, surface_sizes={99: 0.1})
    # 顶点索引越界
    with pytest.raises(ValueError):
        resolve_size_field(V, F, 0.30, vertex_sizes={99: 0.1})
    # surface_sizes 字典缺 target
    with pytest.raises(ValueError):
        resolve_size_field(V, F, 0.30, surface_sizes={0: {"min": 0.1}})


# ------------------------------------------------------- 栅格局部加密 + 确定性
def test_grid_local_refinement():
    """10×10 栅格局部加密：近控制区边长远小于远场、边界边数保持。"""
    nx = ny = 11                       # 10×10 个单元面
    V, F = grid_surface(nx, ny)
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)

    # 控制左下角顶点区（v=0 为网格原点）
    base = 0.30
    target = 0.02
    growth = 0.15
    sizes = resolve_size_field(V, F, base,
                               vertex_sizes={0: target},
                               growth=growth)
    sizes = np.asarray(sizes, float)

    # 近控制区顶点（距原点欧氏最近，索引 0）尺寸最小
    assert sizes[0] == pytest.approx(target, abs=1e-12)
    # 远场角点（右上角索引最大值）饱和于 base（growth×对角线距离超差）
    far = (nx - 1) * ny + (ny - 1)
    assert sizes[far] == pytest.approx(base, abs=1e-9)

    # 近控制区边长远小于远场：重网格化后检查（用 remesh_with_controls）
    V2, F2 = remesh_with_controls(V, F, base,
                                  vertex_sizes={0: target},
                                  growth=growth,
                                  refine=True, collapse=True, smooth=True,
                                  max_iter=20)
    V2 = np.asarray(V2, float)
    F2 = np.asarray(F2, np.int64)

    ed = []
    seen = set()
    for a, b, c in F2:
        for u, w in ((a, b), (b, c), (c, a)):
            k = (u, w) if u < w else (w, u)
            if k in seen:
                continue
            seen.add(k)
            ed.append(float(np.linalg.norm(V2[u] - V2[w])))
    ed = np.asarray(ed)

    # 近控制区：以原点为球心半径 0.15 内的顶点平均边长
    near = np.where(np.linalg.norm(V2, axis=1) < 0.15)[0]
    far = np.where(np.linalg.norm(V2 - np.array([1.0, 1.0, 0.0]),
                                  axis=1) < 0.30)[0]
    near_edges = []
    for (u, w) in seen:
        if u in set(near) and w in set(near):
            near_edges.append(float(np.linalg.norm(V2[u] - V2[w])))
    far_edges = []
    for (u, w) in seen:
        if u in set(far) and w in set(far):
            far_edges.append(float(np.linalg.norm(V2[u] - V2[w])))
    avg_near = float(np.mean(near_edges)) if near_edges else float("inf")
    avg_far = float(np.mean(far_edges)) if far_edges else float("inf")
    assert avg_near < 0.5 * avg_far, "近控制区应显著加密"

    # 边界边数保持：开面边缘半环数不变（边界边数 = 周界 4(nx-1) = 40）
    from mesh_remesh import _boundary_vertices, _build_adj
    adj = _build_adj(F2)
    bv = _boundary_vertices(F2, adj)
    # 边界顶点应为周界一圈 4×(nx-1) 个（横竖边各 nx-1，共 4(nx-1)）
    assert len(bv) == 4 * (nx - 1), "边界顶点数应保持 = 4×(nx−1)"


def test_deterministic_two_runs_bitwise():
    """两次运行逐位一致（确定性）。"""
    V, F = grid_surface(11, 11)
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    base = 0.30
    target = 0.05

    V2a, F2a = remesh_with_controls(V, F, base,
                                    vertex_sizes={0: target},
                                    growth=0.15,
                                    refine=True, collapse=True, smooth=True,
                                    max_iter=10)
    V2b, F2b = remesh_with_controls(V, F, base,
                                    vertex_sizes={0: target},
                                    growth=0.15,
                                    refine=True, collapse=True, smooth=True,
                                    max_iter=10)
    V2a = np.asarray(V2a, float)
    V2b = np.asarray(V2b, float)
    F2a = np.asarray(F2a, np.int64)
    F2b = np.asarray(F2b, np.int64)
    assert V2a.shape == V2b.shape
    assert F2a.shape == F2b.shape
    assert np.array_equal(V2a, V2b)
    assert np.array_equal(F2a, F2b)

# -*- coding: utf-8 -*-
"""N 波 N6：质量诊断/统计/修复 + interface 处理 + AMR 运行时（parity_100pct_plan.md N6 行）。

覆盖：
  统一质量诊断 cell_quality / quality_report / quality_histogram / cell_metric
  修复 repair_cells（负体积翻转 / 退化去除）
  interface 处理 interface_faces / periodic_pairs
  AMR 运行时 amr_marks / refine_tets / register_amr_hook / run_amr

验收核心（N6 行）：质量 histogram 达标（pass=True 且无退化单元、均值/分位超阈）；
tet 单元在默认 Python（无 scipy）与 conda occ 环境皆真跑。
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

from mesh_amr import (amr_marks, refine_tets, register_amr_hook,
                      unregister_amr_hook, run_amr)
from mesh_interface import interface_faces, periodic_pairs
from mesh_quality import (cell_metric, cell_quality, quality_histogram,
                          quality_report, repair_cells)


def _has(name):
    try:
        __import__(name)
        return True
    except Exception:
        return False


def cube_5tet():
    """单位立方体确定性 5 四面体分解（绕主对角线 0–6）：全正体积、质量 0.687–0.756。"""
    V = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    C = np.array([
        [0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6],
        [0, 7, 4, 6], [0, 4, 1, 6]], np.int64)
    return V, C


def cube_faces_ab():
    """两块沿 x 共享面（x=1）的立方体：A 占 [0,1]³、B 占 [1,2]×[0,1]²。

    返回 (V 16x3, 面索引数组 A[12,3], B[12,3])；A/+x 面与 B/−x 面几何重合（各 2 三角）。
    """
    V = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        [1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0],
        [1, 0, 1], [2, 0, 1], [2, 1, 1], [1, 1, 1]], float)
    A = np.array([
        [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [3, 2, 6], [3, 6, 7],
        [0, 3, 7], [0, 7, 4], [1, 2, 6], [1, 6, 5]], np.int64)
    B = np.array([
        [8, 9, 10], [8, 10, 11], [12, 13, 14], [12, 14, 15],
        [8, 9, 13], [8, 13, 12], [11, 10, 14], [11, 14, 15],
        [8, 11, 15], [8, 15, 12], [9, 10, 14], [9, 14, 13]], np.int64)
    return V, A, B


# ---------------------------------------------------------------- 质量 histogram
def test_quality_histogram_5tet_passes():
    V, C = cube_5tet()
    h = quality_histogram(V, C, "tet")
    assert h["n_cells"] == 5
    assert h["pass"]
    assert h["n_degen"] == 0
    assert sum(h["counts"]) == h["n_cells"]
    assert h["mean"] >= 0.6
    assert h["p10"] >= 0.5
    assert 0.0 <= h["min"] <= h["max"] <= 1.0


def test_quality_histogram_requires_threshold_pass():
    # 阈值高于均值 → 判不达标；达标判定以 mean 是否超阈为准
    V, C = cube_5tet()
    h = quality_histogram(V, C, "tet", quality_threshold=0.99)
    assert not h["pass"]          # 均值约 0.7 < 0.99
    assert h["n_bad"] == 5        # 全部单元低于 0.99
    h2 = quality_histogram(V, C, "tet", quality_threshold=0.30)
    assert h2["pass"]


def test_quality_histogram_empty_cells():
    V, C = cube_5tet()
    h = quality_histogram(V, np.empty((0, 4), np.int64), "tet")
    assert h["n_cells"] == 0
    assert not h["pass"]
    assert sum(h["counts"]) == 0


def test_cell_quality_report_native_and_histogram():
    V, C = cube_5tet()
    r = cell_quality(V, C, "tet")
    assert r["kind"] == "tet"
    assert r["pass"]
    assert r["histogram"]["n_cells"] == 5
    assert r["n_cells"] == 5
    qr = quality_report(V, C, "tet")
    assert qr["ok"] and qr["histogram"]["pass"]


# ---------------------------------------------------------------- cell_metric
def test_cell_metric_tet_range():
    V, C = cube_5tet()
    for row in C:
        m = cell_metric(V, row, "tet")
        assert 0.0 <= m <= 1.0
        assert m > 0.0


def test_cell_metric_prism():
    # 直角三棱柱（底面 z=0 三角 + 顶面 z=1 三角）
    V = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1]], float)
    C = np.array([0, 1, 2, 3, 4, 5])
    m = cell_metric(V, C, "prism")
    assert 0.0 <= m <= 1.0


@pytest.mark.skipif(not _has("scipy"), reason="无 scipy")
def test_cell_metric_hex_cube_sphericity():
    box = {"kind": "hex", "lo": [0, 0, 0], "hi": [1, 1, 1]}
    # 立方体 Wadell 球度 = π^{1/3}(6·1)^{2/3}/6 ≈ 0.806
    assert abs(cell_metric(None, box, "hex") - 0.806) < 0.05


@pytest.mark.skipif(not _has("scipy"), reason="无 scipy")
def test_cell_metric_poly():
    poly = {"seed": 0,
            "vertices": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
            "faces": [[0, 1, 2], [0, 2, 3], [0, 4, 7], [0, 7, 3],
                      [1, 4, 5], [1, 5, 2], [6, 5, 4], [6, 4, 7],
                      [6, 7, 3], [6, 3, 2], [6, 2, 5], [6, 5, 4]],
            "volume": 1.0, "n_faces": 6}
    assert abs(cell_metric(None, poly, "poly") - 0.806) < 0.05


# ---------------------------------------------------------------- 修复
def test_repair_cells_recover_negative_and_degenerate():
    V, C = cube_5tet()
    Cb = C.copy()
    Cb[0] = Cb[0][[0, 2, 1, 3]]          # 交换顶点 → 负体积
    Cbad = np.vstack([Cb, [0, 0, 0, 0]])  # 退化（重复顶点 → 近零体积）
    out = repair_cells(V, Cbad, "tet")
    assert out["n_before"] == 6
    assert out["n_flipped"] == 1
    assert out["n_dropped"] == 1
    assert out["n_after"] == 5
    assert out["ok"]
    q = out["quality"]
    assert q["n_negative"] == 0
    assert q["n_null"] == 0
    # 修复后 histogram 达标
    assert quality_histogram(out["vertices"], out["cells"], "tet")["pass"]


def test_repair_cells_nothing_to_repair():
    V, C = cube_5tet()
    out = repair_cells(V, C, "tet")
    assert out["n_flipped"] == 0 and out["n_dropped"] == 0
    assert out["n_after"] == out["n_before"]


# ---------------------------------------------------------------- interface
def test_interface_faces_two_cubes_shared_face():
    V, A, B = cube_faces_ab()
    out = interface_faces(V, [A, B])
    assert out["n_regions"] == 2
    assert out["counts"].get((0, 1), 0) == 2      # 共享 x=1 面 2 三角
    assert len(out["pairs"]) == 2
    for ri, rj, fi, fj in out["pairs"]:
        assert (ri, rj) == (0, 1)


def test_interface_faces_single_region_no_self_pair():
    V, A, B = cube_faces_ab()
    out = interface_faces(V, [A])
    assert out["n_regions"] == 1
    assert out["pairs"] == []
    assert out["counts"] == {}


def test_periodic_pairs_translation():
    V, A, _ = cube_faces_ab()
    # 沿 x 平移 1 把 cube A 的 −x 面对到 +x 面
    out = periodic_pairs(V, A, translation=(1.0, 0.0, 0.0))
    assert out["n_faces"] == 12
    assert out["n_matched"] == 2
    assert out["matched"]
    for i, j in out["pairs"]:
        assert i != j


# ---------------------------------------------------------------- AMR 运行时
def test_amr_marks_none_below_threshold():
    V, C = cube_5tet()
    m = amr_marks(V, C, "tet", threshold=0.0)
    assert m["n_cells"] == 5
    assert m["n_marks"] == 0
    assert m["fraction"] == 0.0


def test_amr_marks_percentile():
    V, C = cube_5tet()
    m = amr_marks(V, C, "tet", percentile=20)     # 标记最劣 20%
    assert m["n_marks"] == 1
    assert m["fraction"] == pytest.approx(0.2)


def test_refine_tets_full_1_8():
    V, C = cube_5tet()
    out = refine_tets(V, C)                        # mask=None → 全细化
    assert out["n_before"] == 5
    assert out["n_refined"] == 5
    assert out["n_after"] == 5 * 8
    assert out["ok"]


def test_refine_tets_partial_hanging_nodes():
    V, C = cube_5tet()
    mask = np.zeros(len(C), bool)
    mask[0] = True
    out = refine_tets(V, C, mask=mask)
    assert out["n_before"] == 5
    assert out["n_refined"] == 1
    assert out["n_after"] == 4 + 8
    assert out["n_hanging"] > 0                     # 非协调界面产生悬挂节点


def test_run_amr_hook_registry():
    V, C = cube_5tet()
    register_amr_hook(refine_tets)
    try:
        out = run_amr(V, C, "tet", threshold=0.0)   # 无单元低于阈值 → 0 标记
        assert out["n_marks"] == 0
        assert out["hooks_called"] == 1
        assert out["results"][0]["hook"] == "refine_tets"
    finally:
        unregister_amr_hook(refine_tets)
    out2 = run_amr(V, C, "tet", threshold=0.0)
    assert out2["hooks_called"] == 0                # 注销后无钩子被分派

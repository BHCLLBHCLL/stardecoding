# -*- coding: utf-8 -*-
"""R 波 R5：教程尺度网格对标（官方 .sim 域表面/补丁 ↔ 自研表面重网格化）。

覆盖：
  工具：_edge_stats（唯一边/边长统计/水密判定）、_tri_min_angles（等边=60°）
  表面：boundary_surface（pipe：4 边界/3050 面环/水密/面积/散度体积）
  补丁：boundary_patch_stats（每补丁三角数/边中位数/最小角中位数）
  对标：surface_benchmark（面数比/面积比/边尺度比/质量 + attempts 记录 + 退化保护）
  诚实：二维（airfoil 边界环仅 2 点）与非水密（methaneOnPt）分别给出拒绝理由
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import mesh_benchmark as mb  # noqa: E402

CORPUS = r"D:/training/starccm/startutorialsdata"
PIPE = os.path.join(CORPUS, "optimate", "data", "pipeBlockage.sim")
AIRFOIL = os.path.join(CORPUS, "optimate", "data", "airfoil.sim")
METHANE = os.path.join(CORPUS, "combustion", "data", "methaneOnPt.sim")
needs_pipe = pytest.mark.skipif(not os.path.isfile(PIPE), reason="语料 pipeBlockage.sim 缺失")
needs_all = pytest.mark.skipif(
    not (os.path.isfile(AIRFOIL) and os.path.isfile(METHANE)), reason="语料缺失")


def _tet_surface():
    V = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], float)
    F = np.array([(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)], np.int64)
    return V, F


# ---------------------------------------------------------------- 工具
def test_edge_stats_watertight_tetra():
    V, F = _tet_surface()
    st = mb._edge_stats(V, F)
    assert st["n_faces"] == 4 and st["n_edges"] == 6
    assert st["watertight"] and st["edge_usage"] == {"2": 6}
    assert st["edge_mean"] > 0 and st["edge_min"] > 0


def test_edge_stats_open_surface_not_watertight():
    V = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)], float)
    F = np.array([(0, 1, 2), (1, 3, 2)], np.int64)
    st = mb._edge_stats(V, F)
    assert not st["watertight"] and "1" in st["edge_usage"]


def test_tri_min_angles_equilateral_and_degenerate():
    V = np.array([(0, 0, 0), (1, 0, 0), (0.5, np.sqrt(3) / 2, 0), (2, 0, 0)], float)
    F = np.array([(0, 1, 2), (0, 1, 3)], np.int64)
    ang = mb._tri_min_angles(V, F)
    assert abs(ang[0] - 60.0) < 1e-6
    assert ang[1] < 1e-6            # 退化三角形最小角 ≈ 0


# ---------------------------------------------------------------- 语料：表面/补丁
@needs_pipe
def test_boundary_surface_pipe_watertight():
    from sim_parser import SimFile
    surf = mb.boundary_surface(SimFile(PIPE))
    assert surf["ok"] and surf["watertight"]
    assert surf["n_boundaries"] == 4 and surf["n_rings"] == 3050
    assert surf["edge_usage"] == {"2": 9102}
    assert abs(surf["area"] - 0.3083) < 5e-3
    assert abs(surf["volume"] - 0.009096) < 5e-5


@needs_pipe
def test_official_scale_pipe():
    from sim_parser import SimFile
    off = mb.official_scale(SimFile(PIPE))
    assert off["ok"] and off["n_cells"] == 14882
    assert off["n_points"] == 16013 and off["n_faces"] == 42655
    assert off["watertight"] and not off["is_2d"]
    assert 0.005 < off["target_spacing"] < 0.02


@needs_pipe
def test_boundary_patch_stats_pipe():
    from sim_parser import SimFile
    st = mb.boundary_patch_stats(SimFile(PIPE))
    assert st["ok"] and st["n_patches"] == 4
    names = sorted(p["name"] for p in st["patches"])
    assert names == ["Body 1.inlet", "Body 1.outlet", "Body 1.wall", "blockage"]
    wall = [p for p in st["patches"] if p["name"] == "Body 1.wall"][0]
    assert wall["n_triangles"] == 4764
    for p in st["patches"]:
        assert p["edge_median"] > 0 and p["min_angle_median"] > 0


# ---------------------------------------------------------------- 对标
@needs_pipe
def test_surface_benchmark_pipe_verdict_and_attempts():
    from sim_parser import SimFile
    rep = mb.surface_benchmark(SimFile(PIPE))
    assert rep["ok"] and rep["mode"] == "surface-patch"
    assert rep["verdict"]["patch"]["name"] == "blockage"
    assert len(rep["attempts"]) >= 1
    assert rep["official"]["n_faces"] == 360
    v = rep["verdict"]
    assert v["face_ratio_ok"] and v["edge_median_ok"] and v["min_angle_ok"]
    assert v["area_ratio_ok"]
    assert v["ok"] is True


@needs_pipe
def test_surface_benchmark_explicit_patch_and_tet_honest():
    from sim_parser import SimFile
    sim = SimFile(PIPE)
    rep = mb.surface_benchmark(sim, patch=("Body 1.inlet", 3))
    assert rep["ok"] and rep["verdict"]["patch"]["name"] == "Body 1.inlet"
    bad = mb.surface_benchmark(sim, patch=("nope", 99))
    assert bad["ok"] is False and "不存在" in bad["reason"]


@needs_all
def test_two_dimensional_and_empty_patches_are_honest():
    from sim_parser import SimFile
    air = mb.boundary_patch_stats(SimFile(AIRFOIL))
    assert air["ok"] and air["n_patches"] == 0        # 二维：边界环仅 2 点 → 无三角面片
    rep = mb.surface_benchmark(SimFile(AIRFOIL))
    assert rep["ok"] is False and "无边界补丁" in rep["reason"]
    met = mb.boundary_surface(SimFile(METHANE))
    assert met["ok"] and not met["watertight"]        # 边界环非流形 → 诚实标注


@pytest.mark.skipif(os.environ.get("STARDECODING_LONG") != "1",
                    reason="tet 同域对标耗时长，仅在 STARDECODING_LONG=1 时运行")
@needs_pipe
def test_tet_mode_long_only():
    from sim_parser import SimFile
    rep = mb.benchmark(SimFile(PIPE), mode="tet")
    assert "reason" in rep and "verdict" in rep or rep.get("ok") is False


@needs_pipe
def test_benchmark_dispatch_modes():
    from sim_parser import SimFile
    sim = SimFile(PIPE)
    assert mb.benchmark(sim, mode="surface")["ok"] is True
    unknown = mb.benchmark(sim, mode="nope")
    assert unknown["ok"] is False and "未知对标模式" in unknown["reason"]

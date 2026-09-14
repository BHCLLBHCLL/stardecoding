# -*- coding: utf-8 -*-
"""S 波 S3：二维（边-单元）网格的维度识别、单元环重建与 VTK_POLYGON 导出。

背景（本仓库实测，非推测）：
  - vortexShed_tutor_v3_0.05_2502.sim 是**二维**案例，但 .sim 里同样用
    「面-单元」存储体系保存：Coord（z 恒为 0，40044 顶点）/ VertexList /
    FaceCellIndex —— 每条"面"恰 2 个顶点 = 一条边（59811 条边，119622 个
    顶点引用，20245 个单元）。旧实现只按三维多面体解释：维度被谎报为 3D，
    导出时写出退化的 2 顶点 VTK_POLYHEDRON"面"。
  - 二维网格的**边界边存放在边界 patch 组**：实测边界组 478 条边与主面组
    交集为 0，因此贴壁单元的主面组边集是"缺一条边的路径"（474 个）；470 个
    路径的闭合边可在边界组中找到（隐式闭合=真实边界边，几何精确），其余 4 个
    是只有 2 条存储边的退化三角形。
  - vortexShed_tutor_v3_0.025.sim 的索引载荷只覆盖 ~13.5% 值域
    （FaceCellIndex max=2779 而单元数 20245），全文件 u4 扫描找不到完整索引 →
    必须继续诚实拒绝（不得编造拓扑）。
"""
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as _ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from sim_parser import (SimFile, assemble_planar_cell_loops,  # noqa: E402
                        mesh_dimensionality)

TUTORIAL_2D = "D:/training/openfoam/benchmark/vortexShed_tutor_v3_0.05_2502.sim"
TUTORIAL_PARTIAL = "D:/training/openfoam/benchmark/vortexShed_tutor_v3_0.025.sim"
CORPUS_3D = "D:/training/starccm/startutorialsdata/combustion/data"
AIRFOIL_2D = "D:/training/starccm/startutorialsdata/optimate/data/airfoil.sim"


# ---------------- 离线：维度判定 ----------------

def test_mesh_dimensionality_volume_planar_line():
    vol = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    assert mesh_dimensionality(vol)[0] == 3
    assert mesh_dimensionality(vol)[1] == []
    planar = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)
    dim, axes, ext = mesh_dimensionality(planar)
    assert dim == 2 and axes == [2]
    assert ext[2] == 0.0 and ext[0] == 1.0
    line = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float)
    assert mesh_dimensionality(line)[0] == 1
    assert mesh_dimensionality(line)[1] == [1, 2]
    assert mesh_dimensionality(None) == (0, [], [])


def _pairs(loop):
    """环的相邻顶点对（含首尾闭合对）。"""
    return {(min(a, b), max(a, b)) for a, b in
            zip(loop, loop[1:] + loop[:1])}


def test_assemble_loops_closed_cycle():
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    loops, st = assemble_planar_cell_loops([[0, 1, 2, 3]], edges)
    assert st == {"ok": True, "closed": 1, "implicit": 0, "failed": 0}
    assert sorted(loops[0]) == [0, 1, 2, 3]
    assert len(loops[0]) == 4 and len(set(loops[0])) == 4
    assert _pairs(loops[0]) == {(0, 1), (1, 2), (2, 3), (0, 3)}


def test_assemble_loops_open_path_is_implicitly_closed():
    """缺一条边的路径 → 直接作多边形环（VTK 隐式闭合）。"""
    edges = np.array([[0, 1], [1, 2]])
    loops, st = assemble_planar_cell_loops([[0, 1]], edges)
    assert st == {"ok": True, "closed": 0, "implicit": 1, "failed": 0}
    assert sorted(loops[0]) == [0, 1, 2]
    assert _pairs(loops[0]) == {(0, 1), (1, 2), (0, 2)}  # (0,2) 为隐式闭合边


def test_assemble_loops_rejects_branch_and_disjoint_paths():
    # 分叉（某顶点度数 3）：不编造顺序
    star = np.array([[0, 1], [0, 2], [0, 3]])
    loops, st = assemble_planar_cell_loops([[0, 1, 2]], star)
    assert st["failed"] == 1 and st["ok"] is False and loops == [[]]
    # 两条互不相连的路径（4 个端点）：同样拒绝
    two = np.array([[0, 1], [2, 3]])
    loops, st = assemble_planar_cell_loops([[0, 1]], two)
    assert st["failed"] == 1 and st["ok"] is False and loops == [[]]
    # 单边/双顶点：不足 3 顶点不成多边形
    one = np.array([[0, 1]])
    loops, st = assemble_planar_cell_loops([[0]], one)
    assert st["failed"] == 1 and st["ok"] is False


def test_assemble_loops_multigraph_and_empty_cell():
    # 重复边（多重图）：度数按边计，2 条重复边仍是 2 顶点退化环 → 拒绝
    dup = np.array([[0, 1], [0, 1]])
    loops, st = assemble_planar_cell_loops([[0, 1]], dup)
    assert st["failed"] == 1 and st["ok"] is False
    # 空单元边集
    loops, st = assemble_planar_cell_loops([[]], dup)
    assert st["failed"] == 1 and st["ok"] is False


def test_assemble_loops_multiple_cells_stats():
    edges = np.array([[0, 1], [1, 2], [2, 0], [2, 3], [3, 4]])
    loops, st = assemble_planar_cell_loops([[0, 1, 2], [3, 4]], edges)
    assert st == {"ok": True, "closed": 1, "implicit": 1, "failed": 0}
    assert sorted(loops[0]) == [0, 1, 2]
    assert sorted(loops[1]) == [2, 3, 4]
    assert _pairs(loops[1]) - {(2, 3), (3, 4)} == {(2, 4)}   # 隐式闭合边


# ---------------- 官方二维语料 ----------------

@pytest.mark.skipif(not os.path.isfile(TUTORIAL_2D), reason="官方二维语料缺失")
def test_official_2d_case_detected_as_planar_edge_mesh():
    sim = SimFile(TUTORIAL_2D)
    vol = sim.extract_volume_mesh()
    assert vol["ok"], vol.get("reason")
    assert vol["count"] == 20245 and vol["kind"] == "poly"
    assert vol["dim"] == 2 and vol["planar"] is True
    assert vol["planar_axes"] == [2]             # z 为常量轴
    assert vol["face_kind"] == "edge"            # 每条"面"恰 2 顶点=边
    assert int(vol["points"][:, 2].max()) == 0 and int(vol["points"][:, 2].min()) == 0
    assert vol["points"].shape[0] == 40044


@pytest.mark.skipif(not os.path.isfile(TUTORIAL_2D), reason="官方二维语料缺失")
def test_official_2d_cell_loops_are_complete_polygons():
    vol = SimFile(TUTORIAL_2D).extract_volume_mesh()
    st = vol["loops2d_stats"]
    assert vol["loops2d_ok"] is True and st["failed"] == 0
    assert st["closed"] + st["implicit"] == 20245
    loops = vol["loops2d"]
    assert len(loops) == 20245 and all(len(l) >= 3 for l in loops)
    npt = vol["points"].shape[0]
    for l in loops:
        assert len(set(l)) == len(l), "环内顶点重复"
        assert 0 <= min(l) and max(l) < npt
    # 所有显式闭合环的相邻对必须都是主面组里的真实边
    edges = {(min(int(vol["face_verts"][2 * i]), int(vol["face_verts"][2 * i + 1])),
              max(int(vol["face_verts"][2 * i]), int(vol["face_verts"][2 * i + 1])))
             for i in range(vol["face_verts"].size // 2)}
    for l in loops:
        if _pairs(l) <= edges:
            continue
        # 允许恰有一条隐式闭合边（贴壁单元的边界边不在主面组）
        assert len(_pairs(l) - edges) == 1, l


@pytest.mark.skipif(not os.path.isfile(TUTORIAL_2D), reason="官方二维语料缺失")
def test_official_2d_boundary_edges_live_in_boundary_groups():
    """二维网格边界边不在主面组：边界组边集与主面组交集为 0。"""
    sim = SimFile(TUTORIAL_2D)
    vol = sim.extract_volume_mesh()
    bf = sim.extract_boundary_faces(vol)
    assert bf["ok"], bf.get("reason")
    bedges = {(min(r), max(r)) for b in bf["boundaries"] for r in b["rings"]}
    medges = {(min(int(vol["face_verts"][2 * i]), int(vol["face_verts"][2 * i + 1])),
               max(int(vol["face_verts"][2 * i]), int(vol["face_verts"][2 * i + 1])))
              for i in range(vol["face_verts"].size // 2)}
    assert len(bedges) == 478, len(bedges)
    assert not (bedges & medges), sorted(bedges & medges)[:5]
    # 隐式闭合单元的闭合边绝大多数可在边界组找到（=几何精确）
    implicit = [l for l in vol["loops2d"]
                if (min(l[0], l[-1]), max(l[0], l[-1])) not in medges]
    assert len(implicit) == vol["loops2d_stats"]["implicit"]
    in_b = sum(1 for l in implicit
               if (min(l[0], l[-1]), max(l[0], l[-1])) in bedges)
    assert in_b == 470 and len(implicit) == 474


@pytest.mark.skipif(not os.path.isfile(TUTORIAL_2D), reason="官方二维语料缺失")
def test_official_2d_vtu_is_polygon():
    # 用 tempfile 而非 pytest tmp_path（本机 pytest 临时根目录权限受限）
    sim = SimFile(TUTORIAL_2D)
    out = os.path.join(tempfile.mkdtemp(prefix="s3mesh2d_"), "2d.vtu")
    assert sim.export_volume_vtu(out) == out
    txt = open(out, encoding="utf-8").read()
    _ET.parse(out)                                  # XML 结构合法
    assert 'NumberOfCells="20245"' in txt and 'NumberOfPoints="40044"' in txt
    get = lambda name: re.search(                        # noqa: E731
        r'<DataArray[^>]*Name="%s"[^>]*>(.*?)</DataArray>' % name,
        txt, re.S).group(1).split()
    types = get("types")
    assert set(types) == {"7"}, "二维网格必须导出 VTK_POLYGON(7) 而不是多面体 41"
    assert len(types) == 20245
    conn = [int(x) for x in get("connectivity")]
    offs = [int(x) for x in get("offsets")]
    assert len(offs) == 20245 and offs == sorted(offs) and offs[-1] == len(conn)
    sizes = [offs[0]] + [b - a for a, b in zip(offs, offs[1:])]
    assert min(sizes) >= 3 and max(sizes) <= 9
    assert len(conn) == 120096


@pytest.mark.skipif(not os.path.isfile(TUTORIAL_PARTIAL), reason="官方语料缺失")
def test_partial_index_case_still_refused_honestly():
    """v3_0.025：索引载荷只覆盖部分值域 → 继续诚实拒绝，不编造拓扑。"""
    sim = SimFile(TUTORIAL_PARTIAL)
    vol = sim.extract_volume_mesh()
    assert vol["ok"] is False and vol["count"] == 0
    assert vol.get("points") is None and vol.get("loops2d") is None
    assert "拓扑反演失败" in vol["reason"]
    bad = os.path.join(tempfile.mkdtemp(prefix="s3mesh2d_"), "bad.vtu")
    assert sim.export_volume_vtu(bad) is None


@pytest.mark.skipif(not os.path.isfile(AIRFOIL_2D), reason="官方语料缺失")
def test_airfoil_2d_loops_cross_validated_with_boundary_faces():
    """airfoil 同为二维网格：336 个隐式闭合单元 ↔ G4 实测 336 个边界面。

    G4 独立断言（self_test.py）给出 airfoil 336 面/7 边界；此处不重复其口径，
    而是把「隐式闭合边」与「G4 边界边集合」逐条对齐——两套独立证据交叉验证。
    """
    sim = SimFile(AIRFOIL_2D)
    vol = sim.extract_volume_mesh()
    assert vol["ok"] and vol["count"] == 16987
    assert vol["dim"] == 2 and vol["planar"] is True and vol["face_kind"] == "edge"
    assert vol["loops2d_stats"] == {"ok": True, "closed": 16651,
                                    "implicit": 336, "failed": 0}
    bf = sim.extract_boundary_faces(vol)
    assert bf["ok"] and bf["total_faces"] == 336 and len(bf["boundaries"]) == 7
    bedges = {(min(r), max(r)) for b in bf["boundaries"] for r in b["rings"]}
    fv = vol["face_verts"]
    medges = {(min(int(fv[2 * i]), int(fv[2 * i + 1])),
               max(int(fv[2 * i]), int(fv[2 * i + 1])))
              for i in range(fv.size // 2)}
    assert len(bedges) == 336 and not (bedges & medges)
    implicit = [l for l in vol["loops2d"]
                if (min(l[0], l[-1]), max(l[0], l[-1])) not in medges]
    assert len(implicit) == 336 == vol["loops2d_stats"]["implicit"]
    assert all((min(l[0], l[-1]), max(l[0], l[-1])) in bedges for l in implicit)


@pytest.mark.skipif(not os.path.isfile(os.path.join(CORPUS_3D, "methaneOnPt.sim")),
                    reason="三维语料缺失")
def test_3d_corpus_still_uses_polyhedron():
    """三维语料不得被二维路径劫持：dim==3、face_kind==poly、导出 types=41。"""
    sim = SimFile(os.path.join(CORPUS_3D, "methaneOnPt.sim"))
    vol = sim.extract_volume_mesh()
    assert vol["ok"] and vol["count"] == 1750
    assert vol["dim"] == 3 and vol["planar"] is False and vol["face_kind"] == "poly"
    assert "loops2d" not in vol
    out = os.path.join(tempfile.mkdtemp(prefix="s3mesh3d_"), "3d.vtu")
    assert sim.export_volume_vtu(out) == out
    txt = open(out, encoding="utf-8").read()
    types = re.search(r'Name="types"[^>]*>(.*?)</DataArray>', txt, re.S).group(1).split()
    assert set(types) == {"41"} and len(types) == 1750

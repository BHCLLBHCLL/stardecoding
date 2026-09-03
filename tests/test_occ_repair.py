# -*- coding: utf-8 -*-
"""C 波 C4：表面修复工具集（hole fill / coarse / fine / quality metrics）。

hole fill 与 quality 为纯 numpy，任意 Python 环境可用；coarse/fine 走 OCC
（B-Rep 细分密度控制），随 occ_bridge.has_occ() 门控。真跑 OCC 用例：
    <conda>/envs/occ/python.exe -m pytest tests/test_occ_repair.py -q
"""
import pytest

from occ_repair import (boundary_edges, boundary_loops, fill_holes,
                        quality_metrics, has_occ, surface_mesh)
from mesh_io import cube_mesh

HAS_OCC = has_occ()


# --- 纯 numpy：孔洞检测 / 补洞 / 质量（两环境皆跑） -----------------------
def _open_box():
    """立方体网格，挖掉底面 2 三角 → 4 边方孔的开盒。"""
    V, F = cube_mesh(2.0)
    return V, F[2:]


def test_c4_hole_detect_cube_closed():
    """无孔网格：闭合（0 边界边）、无边界环。"""
    V, F = cube_mesh(2.0)
    assert len(boundary_edges(V, F)[0]) == 0
    assert boundary_loops(V, F) == []


def test_c4_fill_holes_open_box():
    """补洞：开底方孔(4 边) 补成 2 三角 → 闭合、恢复 12 三角、无退化。"""
    V, F = _open_box()
    loops = boundary_loops(V, F)
    assert len(loops) == 1 and len(loops[0]) == 4
    V2, F2 = fill_holes(V, F)
    assert len(boundary_edges(V2, F2)[0]) == 0
    assert len(F2) == 12
    assert quality_metrics(V2, F2)["n_degenerate"] == 0


def test_c4_quality_metrics_cube():
    """立方体质量：12 三角、闭合、无退化/极瘦三角、角度在 (0,180)。"""
    V, F = cube_mesh(2.0)
    m = quality_metrics(V, F)
    assert m["n_triangles"] == 12 and m["closed"]
    assert m["n_degenerate"] == 0 and m["n_skinny"] == 0
    assert m["area"]["min"] > 0
    assert 0 < m["min_angle_deg"] < m["max_angle_deg"] < 180


# --- OCC：coarse/fine 与 B-Rep 一条龙（occ 环境真跑） ---------------------
@pytest.mark.skipif(not HAS_OCC, reason="需 conda 环境 occ")
def test_c4_surface_mesh_fine_coarse():
    """粗细网格：球面 fine 三角数远大于 coarse（细分密度可控）。"""
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
    s = BRepPrimAPI_MakeSphere(1.0).Shape()
    f = surface_mesh(s, mode="fine")
    c = surface_mesh(s, mode="coarse")
    assert f["n_triangles"] > c["n_triangles"]
    assert f["n_triangles"] > 500 and c["n_triangles"] < 1500


@pytest.mark.skipif(not HAS_OCC, reason="需 conda 环境 occ")
def test_c4_surface_mesh_box_planar_invariant():
    """平面面：box 无论粗细恒每面 2 三角（12 三角，deflection 不影响平面）。"""
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    b = BRepPrimAPI_MakeBox(2, 3, 4).Shape()
    assert surface_mesh(b, mode="fine")["n_triangles"] == 12
    assert surface_mesh(b, mode="coarse")["n_triangles"] == 12


@pytest.mark.skipif(not HAS_OCC, reason="需 conda 环境 occ")
def test_c4_repair_surface_brep_roundtrip():
    """B-Rep 一条龙：box → STEP → repair_surface，补洞后闭合、无退化。"""
    import os
    import tempfile
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from occ_bridge import export_shape
    from occ_repair import repair_surface
    b = BRepPrimAPI_MakeBox(4, 4, 4).Shape()
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "c4.step")
        export_shape(b, p)
        out = repair_surface(p, mode="fine")
        assert out["ok"]
        assert out["metrics"]["n_degenerate"] == 0
        assert out["counts"]["n_triangles"] > 0
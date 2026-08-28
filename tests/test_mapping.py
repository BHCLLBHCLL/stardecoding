# -*- coding: utf-8 -*-
"""解析侧新增：边界→面片映射（boundary_faces / part_surface_patches / 着色 polydata）。"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest

SIM = os.path.join(ROOT, "adjointWing_start.sim")


def test_part_surface_patches():
    from sim_parser import SimFile
    sim = SimFile(SIM)
    psp = sim.part_surface_patches()
    assert psp["parts"], "应有 part 网格分组"
    main = max(psp["parts"], key=lambda p: p["triangles"])
    assert main["name"] == "Fluid Domain" and main["triangles"] == 2824
    assert main["patch"], "主 part 应有每面 patch 表"
    assert main["patch"].get(10) == 460  # Wing End Plate patch


def test_boundary_faces():
    from sim_parser import SimFile
    sim = SimFile(SIM)
    bf = sim.boundary_faces()
    assert bf is not None
    assert bf["total"] >= 2824
    by = bf["by_boundary"]
    assert "Wing End Plate" in by
    faces = by["Wing End Plate"]["Fluid Domain"]
    assert len(faces) == 460
    # 面索引为 1 基有序去重
    assert sorted(faces) == faces and len(set(faces)) == len(faces)


def test_boundary_colored_polydata():
    from sim_parser import SimFile
    from star_gui_vtk import boundary_colored_polydata
    sim = SimFile(SIM)
    res = boundary_colored_polydata(sim)
    assert res is not None
    pd = res["polydata"]
    assert pd.GetNumberOfCells() == 2824
    assert pd.GetCellData().GetScalars().GetNumberOfTuples() == 2824
    assert 5 in res["label_names"] and res["label_names"][5] == "Wing End Plate"

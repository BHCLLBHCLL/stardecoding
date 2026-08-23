# -*- coding: utf-8 -*-
"""面索引压缩：1 基 / 分块偏移，避免 clip 到末顶点把外框拧成星形。"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

from sim_parser import SimFile, compact_face_indices

HELI = r"D:\training\starccm\startutorialsdata\motion\data\genericHelicopter_start.sim"


def test_compact_offset_block():
    faces = np.array([[195, 196, 197], [122580, 122581, 122582]], dtype=np.int64)
    n = 122582 - 195 + 1
    out, ok = compact_face_indices(faces, n)
    assert ok
    assert int(out.min()) == 0
    assert int(out.max()) == n - 1


def test_compact_one_based():
    faces = np.array([[1, 2, 3], [6, 7, 8]], dtype=np.int64)
    out, ok = compact_face_indices(faces, 8)
    assert ok
    assert list(out[0]) == [0, 1, 2]
    assert list(out[1]) == [5, 6, 7]


def test_compact_already_zero_based():
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    out, ok = compact_face_indices(faces, 8)
    assert ok
    assert list(out[0]) == [0, 1, 2]


@pytest.mark.skipif(not os.path.exists(HELI), reason="tutorial helicopter .sim not present")
def test_helicopter_extract_mesh_consistent():
    sim = SimFile(HELI)
    m = sim.extract_mesh()
    assert m["faces"] is not None and m["vertices"] is not None
    assert m["faces"].shape[0] == 244768
    assert m["vertices"].shape[0] == 122388
    assert m["consistent"] is True
    assert int(m["faces"].min()) >= 0
    assert int(m["faces"].max()) < m["vertices"].shape[0]


@pytest.mark.skipif(not os.path.exists(HELI), reason="tutorial helicopter .sim not present")
def test_helicopter_box_parts_own_vertices():
    from star_gui_vtk import part_meshes
    sim = SimFile(HELI)
    parts = part_meshes(sim)
    by = {p["name"]: p for p in parts}
    assert "Wind Tunnel" in by
    wt = by["Wind Tunnel"]
    assert wt["triangles"] == 244768
    assert wt["vertices"].shape[0] == 122388
    assert int(wt["faces"].max()) < wt["vertices"].shape[0]
    used = np.unique(wt["faces"])
    assert used.size > 10000  # 不是钉在少数顶点上的星形
    for name in ("Heli Field", "Far Field"):
        p = by[name]
        assert p["vertices"].shape[0] == 8
        assert p["triangles"] == 12
        assert int(p["faces"].max()) < 8
        ext = p["vertices"].max(0) - p["vertices"].min(0)
        assert float(ext.min()) > 0.5


@pytest.mark.skipif(not os.path.exists(HELI), reason="tutorial helicopter .sim not present")
def test_helicopter_mesh_scene_robin_and_plane():
    from star_gui_vtk import build_scene_actors
    sim = SimFile(HELI)
    scene = [o for o in sim.objects
             if o.class_name == "star.vis.Scene" and o.name == "Mesh Scene 1"][0]
    actors, _cam = build_scene_actors(sim, scene)
    names = [n for k, n, _i, _a in actors]
    keys = [k for k, _n, _i, _a in actors]
    assert "ROBIN" in names
    assert any(k.startswith("plane:") or n == "Plane Section" for k, n in zip(keys, names))
    assert "Outlet" not in names
    assert "Ceiling" not in names
    assert not any("Heli Field" in k or "Far Field" in k for k in keys)
    robin = [a for k, n, _i, a in actors if n == "ROBIN" and k.startswith("part:")][0]
    ncells = robin.GetMapper().GetInput().GetNumberOfCells()
    # Automated Mesh.Remesh 面网格（约 2.3 万），不是 CAD 三角化（24 万）
    assert 15000 < ncells < 40000


@pytest.mark.skipif(not os.path.exists(HELI), reason="tutorial helicopter .sim not present")
def test_helicopter_mesh_scene_uses_remesh_not_cad():
    from star_gui_vtk import (
        build_scene_actors, representation_source_id, mesh_bundle_for_part,
        part_meshes, scene_displayers,
    )
    sim = SimFile(HELI)
    scene = [o for o in sim.objects
             if o.class_name == "star.vis.Scene" and o.name == "Mesh Scene 1"][0]
    mesh_disp = [d for d in scene_displayers(sim, scene) if d.name == "Mesh 1"][0]
    src_id = representation_source_id(sim, mesh_disp)
    assert src_id > 0
    src = sim.objmap[src_id]
    assert "DescriptionSource" in (src.class_name or "")
    wt = [o for o in sim.objects
          if o.class_name == "star.meshing.MeshPart" and o.name == "Wind Tunnel"][0]
    tess = {p["id"]: p for p in part_meshes(sim)}
    remesh = mesh_bundle_for_part(sim, wt, src_id, tess_by_id=tess)
    cad = tess[wt.id]
    assert remesh["triangles"] == 38258
    assert cad["triangles"] == 244768
    assert remesh["triangles"] < cad["triangles"] / 4


@pytest.mark.skipif(not os.path.exists(HELI), reason="tutorial helicopter .sim not present")
def test_helicopter_geometry_scene_filters_boxes():
    from star_gui_model import StarSceneModel
    from star_gui_vtk import build_scene_actors
    sim = SimFile(HELI)
    m = StarSceneModel(sim)
    root = m.sim_tree()[0]
    scenes = [c for c in root.children if c.label == "Scenes"][0]
    geo = [c for c in scenes.children if c.label == "Geometry Scene 1"][0]
    geom1 = [c for c in geo.children if c.label == "Geometry 1"][0]
    parts = [c for c in geom1.children if c.label == "Parts"][0]
    names = [c.label for c in parts.children]
    assert "Wind Tunnel" in names
    assert "Heli Field" not in names
    assert "Far Field" not in names
    wt = [c for c in parts.children if c.label == "Wind Tunnel"][0]
    surfs = [c.label for c in wt.children]
    assert "ROBIN" in surfs
    assert "Outlet" in surfs
    scene = [o for o in sim.objects
             if o.class_name == "star.vis.Scene" and o.name == "Geometry Scene 1"][0]
    actors, _cam = build_scene_actors(sim, scene)
    names = [n for k, n, _i, _a in actors if k.startswith("part:")]
    assert "ROBIN" in names
    assert "Outlet" in names
    assert "Heli Field" not in names
    assert "Far Field" not in names


@pytest.mark.skipif(not os.path.exists(HELI), reason="tutorial helicopter .sim not present")
def test_helicopter_geometry_scene_is_cad_not_mesh():
    """Geometry Scene：CAD 曲面 + 域边框；不是 Remesh 蛛网。"""
    from star_gui_vtk import build_scene_actors, representation_source_id, scene_displayers
    sim = SimFile(HELI)
    scene = [o for o in sim.objects
             if o.class_name == "star.vis.Scene" and o.name == "Geometry Scene 1"][0]
    geo = [d for d in scene_displayers(sim, scene) if d.name == "Geometry 1"][0]
    outline = [d for d in scene_displayers(sim, scene) if d.name == "Outline 1"][0]
    assert representation_source_id(sim, geo) == 0
    assert representation_source_id(sim, outline) == 0
    actors, _cam = build_scene_actors(sim, scene)
    robin = [a for k, n, _i, a in actors if n == "ROBIN" and k.startswith("part:")][0]
    ncells = robin.GetMapper().GetInput().GetNumberOfCells()
    assert ncells > 100000
    assert robin.GetProperty().GetEdgeVisibility() == 0
    assert robin.GetProperty().GetOpacity() > 0.9
    outlet_s = [a for k, n, _i, a in actors if n == "Outlet" and k.startswith("part:")][0]
    assert outlet_s.GetMapper().GetInput().GetNumberOfCells() <= 16
    assert outlet_s.GetProperty().GetOpacity() < 0.5
    for k, n, _i, a in actors:
        if k.startswith("edges:") and n != "ROBIN":
            nc = a.GetMapper().GetInput().GetNumberOfCells()
            assert nc < 40, (k, nc)

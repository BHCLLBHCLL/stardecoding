# -*- coding: utf-8 -*-
"""M2 冒烟：3D 网格 actors + 离屏渲染 PNG + GUI 视口（minimal 平台）。"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest

SIM = os.path.join(ROOT, "adjointWing_start.sim")


@pytest.fixture(scope="module")
def app():
    from PyQt5.QtWidgets import QApplication
    qapp = QApplication.instance() or QApplication([])
    return qapp


@pytest.fixture
def work_dir():
    d = os.path.join(ROOT, "_tmp_tests")
    os.makedirs(d, exist_ok=True)
    return d


def test_part_meshes():
    from sim_parser import SimFile
    from star_gui_vtk import part_meshes
    sim = SimFile(SIM)
    parts = part_meshes(sim)
    assert len(parts) >= 3  # Fluid Domain + 2 blocks
    total = sum(p["triangles"] for p in parts)
    assert total == 2848  # 2824 + 12 + 12
    by = {p["name"]: p for p in parts}
    assert by["Fluid Domain"]["vertices"].shape[0] == 1412
    assert int(by["Fluid Domain"]["faces"].max()) < 1412
    for name in ("Small Block", "Large Block"):
        assert by[name]["vertices"].shape[0] == 8
        assert by[name]["triangles"] == 12
        assert int(by[name]["faces"].max()) < 8
        ext = by[name]["vertices"].max(0) - by[name]["vertices"].min(0)
        assert float(ext.min()) > 0.05


def test_build_actors_and_offscreen_render(work_dir):
    from sim_parser import SimFile
    from star_gui_vtk import build_mesh_actors, render_offscreen_png
    sim = SimFile(SIM)
    actors = build_mesh_actors(sim)
    assert len(actors) >= 3
    cells = sum(a.GetMapper().GetInput().GetNumberOfCells() for _k, _n, _i, a in actors)
    assert cells == 2848
    out = os.path.join(work_dir, "mesh.png")
    render_offscreen_png(actors, out)
    assert os.path.exists(out) and os.path.getsize(out) > 5000


def test_gui_mesh_tab(app):
    """无头模式：viewport 为占位 QLabel，场景标签页逻辑仍被验证。"""
    from star_gui import StarMainWindow
    import time
    win = StarMainWindow()
    win.show()
    win.load_file(SIM)
    t0 = time.time()
    while win.sim is None and time.time() - t0 < 30:
        app.processEvents()
        time.sleep(0.02)
    assert win.sim is not None
    t0 = time.time()
    while win.graphics_tabs.count() < 2 and time.time() - t0 < 30:
        app.processEvents()
        time.sleep(0.02)
    assert win.graphics_tabs.count() >= 2
    win.close()

# -*- coding: utf-8 -*-
"""M2 冒烟：3D 网格 actors + 离屏渲染 PNG + GUI 视口（offscreen）。"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest

SIM = os.path.join(ROOT, "adjointWing_start.sim")


@pytest.fixture(scope="module")
def app():
    from PyQt5.QtWidgets import QApplication
    qapp = QApplication.instance() or QApplication([])
    return qapp



def test_gui_3d_viewport(app):
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
    while not win.viewport.actors and time.time() - t0 < 30:
        app.processEvents()
        time.sleep(0.02)
    assert len(win.viewport.actors) >= 3
    win.close()

def test_part_meshes():
    from sim_parser import SimFile
    from star_gui_vtk import part_meshes
    sim = SimFile(SIM)
    parts = part_meshes(sim)
    assert len(parts) >= 3  # Fluid Domain + 2 blocks
    total = sum(p["triangles"] for p in parts)
    assert total == 2848  # 2824 + 12 + 12
    assert any(p["name"] == "Fluid Domain" for p in parts)


@pytest.fixture
def work_dir():
    """工作区本地临时目录（pytest 系统 tmp_path 在该会话被沙箱限制）。"""
    d = os.path.join(ROOT, "_tmp_tests")
    os.makedirs(d, exist_ok=True)
    return d


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


def test_gui_3d_viewport(app):
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
    while not win.viewport.actors and time.time() - t0 < 30:
        app.processEvents()
        time.sleep(0.02)
    assert len(win.viewport.actors) >= 3
    win.close()
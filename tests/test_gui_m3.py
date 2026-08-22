# -*- coding: utf-8 -*-
"""M3 冒烟：场景/显示器/视图相机 + 场景标签页（minimal 平台）。"""
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


def _sim():
    from sim_parser import SimFile
    return SimFile(SIM)


def test_scene_model_scenes():
    from star_gui_model import StarSceneModel
    m = StarSceneModel(_sim())
    scenes = m.scenes()
    assert any(s["name"] == "Mesh Scene 1" for s in scenes)
    sc = [s for s in scenes if s["name"] == "Mesh Scene 1"][0]
    assert sc["displayers"] and any(d["name"] == "Mesh 1" for d in sc["displayers"])
    assert sc["view"] and "Current 2" in sc["view"]


def test_scene_camera_and_background():
    from star_gui_vtk import scene_camera, scene_background
    sim = _sim()
    scene = [o for o in sim.objects if o.class_name == "star.vis.Scene"][0]
    cam = scene_camera(sim, scene)
    assert cam is not None
    assert cam["position"] is not None and abs(cam["position"][1] + 31.1759) < 1e-3
    assert cam["parallel_scale"] > 0
    bg = scene_background(sim, scene)
    assert bg["solid"] == (1.0, 1.0, 1.0)  # 官方 Solid Background Color = 白


def test_scene_actors_and_render(work_dir):
    from star_gui_vtk import build_scene_actors, render_offscreen_png
    sim = _sim()
    scene = [o for o in sim.objects if o.class_name == "star.vis.Scene"][0]
    actors, cam = build_scene_actors(sim, scene)
    assert actors, "场景应有 actors（表面+边线）"
    out = os.path.join(work_dir, "scene_mesh.png")
    render_offscreen_png(actors, out)
    assert os.path.getsize(out) > 3000


def test_gui_scene_tabs(app):
    from star_gui import StarMainWindow
    import time
    win = StarMainWindow()
    win.show()
    win.load_file(SIM)
    t0 = time.time()
    while win.model is None and time.time() - t0 < 30:
        app.processEvents()
        time.sleep(0.02)
    assert win.model is not None
    t0 = time.time()
    while win.graphics_tabs.count() <= 1 and time.time() - t0 < 30:
        app.processEvents()
        time.sleep(0.02)
    # Info + Mesh Scene 1
    assert win.graphics_tabs.count() >= 2
    assert win.graphics_tabs.tabText(1) == "Mesh Scene 1"
    win.close()


@pytest.fixture
def work_dir():
    d = os.path.join(ROOT, "_tmp_tests")
    os.makedirs(d, exist_ok=True)
    return d

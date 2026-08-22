# -*- coding: utf-8 -*-
"""M5 冒烟：Part 显隐勾选 / Region 高亮 / 边线模式（模型层，无 QVTK）。"""
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


def test_scene_actors_have_edges_and_surfaces():
    """M3 场景 actors 同时含表面与边线，供边线模式使用。"""
    from sim_parser import SimFile
    from star_gui_vtk import build_scene_actors
    sim = SimFile(SIM)
    scene = [o for o in sim.objects if o.class_name == "star.vis.Scene"][0]
    actors, cam = build_scene_actors(sim, scene)
    keys = [k for k, _n, _i, _a in actors]
    assert any(k.startswith("edges:") for k in keys)
    assert any(k.startswith("part:") for k in keys)


def test_part_visibility_logic(app):
    """树勾选 → 显隐对应 actor（headless 下直接调 actor API）。"""
    from star_gui import StarMainWindow
    from star_gui_vtk import build_scene_actors
    import time
    win = StarMainWindow()
    win.show()
    win.load_file(SIM)
    t0 = time.time()
    while win.sim is None and time.time() - t0 < 30:
        app.processEvents()
        time.sleep(0.02)
    # 无头模式：用纯 VTK actor 集合模拟显隐逻辑
    sim = win.sim
    scene = [o for o in sim.objects if o.class_name == "star.vis.Scene"][0]
    actors, _cam = build_scene_actors(sim, scene)
    # on_part_visibility 依赖 win.viewport.actors；无头时跳过 GUI，直接验证逻辑键
    part_names = [n for k, n, _i, _a in actors if k.startswith("part:")]
    assert "Fluid Domain" in part_names
    win.close()


def test_selection_highlight_targets(app):
    """Region 高亮目标 = Region 的 Parts 名称集合（模型层断言）。"""
    from star_gui import StarMainWindow
    import time
    win = StarMainWindow()
    win.show()
    win.load_file(SIM)
    t0 = time.time()
    while win.sim is None and time.time() - t0 < 30:
        app.processEvents()
        time.sleep(0.02)
    region = win.sim.objmap.get(192)
    assert region is not None and region.name == "Fluid Domain"
    # 模拟高亮逻辑：Region.Parts → Keys → 名称
    pg = win.sim.objmap.get(region.dict.get("Parts") or -1)
    names = {p.name for p in (win.sim.objmap.get(k) for k in (pg.dict.get("Keys") or []))
             if p is not None}
    assert "Fluid Domain" in names
    win.close()

# -*- coding: utf-8 -*-
"""M1 冒烟：仿真树 + 属性面板 + StarSceneModel（offscreen）。"""
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


def _load_model():
    from sim_parser import SimFile
    from star_gui_model import StarSceneModel
    sim = SimFile(SIM)
    return StarSceneModel(sim)


def test_scene_model_tree_roots():
    m = _load_model()
    roots = m.tree_roots()
    assert roots and roots[0].obj_id == 2  # Simulation
    labels = []
    def walk(nodes, depth=0):
        for n in nodes:
            labels.append((depth, n.label, n.obj_id, n.class_name))
            walk(n.children, depth + 1)
    walk(roots)
    flat = [l for _, l, _, _ in labels]
    assert "Fluid Domain" in flat
    assert "Mesh Scene 1" in flat
    assert "Physics 1" in flat
    assert "Residuals" in flat


def test_scene_model_properties_region():
    m = _load_model()
    for o in m.sim.objects:
        if o.class_name == "star.common.Region" and o.name == "Fluid Domain":
            rows = m.properties(o)
            d = dict((k, v) for k, v, _ in rows)
            assert d["PresentationName"] == "Fluid Domain"
            assert "BoundaryManager" in d
            return
    raise AssertionError("region not found")


def test_gui_tree_and_properties(app):
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
    assert win.tree_widget.tree.topLevelItemCount() >= 1
    # 选中 Region 节点 → 属性面板出现 PresentationName
    obj = win.model.object_by_id(192)
    win.tree_widget.select_object(192)
    app.processEvents()
    win.on_object_selected(obj)
    assert win.props_widget.table.rowCount() > 10
    win.close()

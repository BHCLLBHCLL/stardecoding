# -*- coding: utf-8 -*-
"""U1：STAR-CCM+ 分组仿真树（Geometry / Regions / Scenes / …）。"""
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
    return StarSceneModel(SimFile(SIM))


def _walk_labels(nodes, acc=None, depth=0):
    acc = acc if acc is not None else []
    for n in nodes:
        acc.append((depth, n.label, n.obj_id, n.class_name))
        _walk_labels(n.children, acc, depth + 1)
    return acc


def test_sim_tree_folders():
    m = _load_model()
    roots = m.sim_tree()
    assert len(roots) == 1
    assert roots[0].label == "adjointWing_start"
    folders = [c.label for c in roots[0].children]
    for name in ("Geometry", "Continua", "Regions", "Solvers",
                 "Plots", "Monitors", "Scenes", "Tools"):
        assert name in folders, folders


def test_sim_tree_geometry_parts():
    m = _load_model()
    labels = [l for _, l, _, _ in _walk_labels(m.sim_tree())]
    assert "Fluid Domain" in labels
    assert "Small Block" in labels
    assert "Large Block" in labels
    assert "Wing Lower Element" in labels  # PartSurface under CadPart


def test_sim_tree_region_boundaries():
    m = _load_model()
    rows = _walk_labels(m.sim_tree())
    # Regions → Fluid Domain → Boundaries → Inlet
    by_label = {}
    for depth, label, oid, cn in rows:
        by_label.setdefault(label, []).append((depth, oid, cn))
    assert "Regions" in by_label
    assert "Inlet" in by_label
    assert "Outlet" in by_label
    region = [r for r in by_label["Fluid Domain"] if r[2] == "star.common.Region"]
    assert region, by_label["Fluid Domain"]
    inlet_depth = by_label["Inlet"][0][0]
    assert inlet_depth > region[0][0]


def test_sim_tree_scene_displayer():
    m = _load_model()
    labels = [l for _, l, _, _ in _walk_labels(m.sim_tree())]
    assert "Mesh Scene 1" in labels
    assert "Mesh 1" in labels
    assert "Residuals" in labels
    assert "Physics 1" in labels


def test_sim_tree_solvers_friendly_names():
    m = _load_model()
    labels = [l for _, l, _, _ in _walk_labels(m.sim_tree())]
    assert "Coupled Implicit" in labels
    assert "Steady" in labels
    assert "Kw Turb" in labels or "KwTurb" in labels


def test_gui_tree_uses_sim_tree(app):
    from star_gui import StarMainWindow
    import time
    win = StarMainWindow()
    win.show()
    win.load_file(SIM)
    t0 = time.time()
    while win.model is None and time.time() - t0 < 30:
        app.processEvents()
        time.sleep(0.02)
    assert win.tree_widget.tree.headerItem().text(0) == "模型 / 场景/绘图"
    top = [win.tree_widget.tree.topLevelItem(i).text(0)
           for i in range(win.tree_widget.tree.topLevelItemCount())]
    assert top == ["adjointWing_start"]
    sim_item = win.tree_widget.tree.topLevelItem(0)
    folders = [sim_item.child(i).text(0) for i in range(sim_item.childCount())]
    assert "Geometry" in folders and "Scenes" in folders and "Regions" in folders
    win.close()

# -*- coding: utf-8 -*-
"""U4：两列属性检查器 + 输出标签随文件名 + 状态栏网格统计。"""
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


def test_properties_two_columns(app):
    from star_gui_panes import PropertiesPanel
    m = _load_model()
    panel = PropertiesPanel()
    panel.set_model(m)
    region = m.object_by_id(192)
    titles = []
    panel.title_changed.connect(titles.append)
    panel.show_object(region)
    assert panel.table.columnCount() == 2
    assert panel.table.rowCount() > 10
    assert titles and "Fluid Domain" in titles[-1] and "属性" in titles[-1]
    assert "192" in panel.hint.text()


def test_property_filter(app):
    from star_gui_panes import PropertiesPanel
    m = _load_model()
    panel = PropertiesPanel()
    panel.set_model(m)
    panel.show_object(m.object_by_id(192))
    n = panel.table.rowCount()
    panel.filter.setText("Presentation")
    assert 0 < panel.table.rowCount() < n
    panel.filter.setText("")
    assert panel.table.rowCount() == n


def test_gui_output_tab_named_after_file(app):
    from star_gui import StarMainWindow
    import time
    win = StarMainWindow()
    win.show()
    win.load_file(SIM)
    t0 = time.time()
    while win.model is None and time.time() - t0 < 30:
        app.processEvents()
        time.sleep(0.02)
    assert win.bottom_tabs.tabText(0) == "adjointWing_start"
    assert "顶点" in win.status_helper.mesh.text()
    win.tree_widget.select_object(192)
    app.processEvents()
    assert "Fluid Domain" in win.props_pane.title_label.text()
    win.close()

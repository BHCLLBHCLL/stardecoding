# -*- coding: utf-8 -*-
"""M0 冒烟：主窗口骨架 / 打开文件 / 摘要窗格（offscreen）。"""
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


def test_main_window_constructs(app):
    from star_gui import StarMainWindow
    win = StarMainWindow()
    win.show()
    assert win.actions.get("File>Open") is not None
    assert win.actions.get("Help>About") is not None
    win.close()
    app.processEvents()


def test_load_sim_summary(app):
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
    assert len(win.sim.objects) == 2076
    txt = win.summary_pane.view.toPlainText()
    assert "adjointWing_start" in txt
    assert "语义层" in txt
    win.close()

# -*- coding: utf-8 -*-
"""M4 冒烟：导出（STL/摘要/报告）与 --cli 无窗口模式。"""
import json
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


def test_cli_export(work_dir):
    """CLI 无窗口模式：解析 + STL + 报告导出。"""
    import subprocess
    stl = os.path.join(work_dir, "cli_mesh.stl")
    rep = os.path.join(work_dir, "cli_report.json")
    p = subprocess.run([sys.executable, os.path.join(ROOT, "star_gui.py"),
                        "--cli", SIM, "--stl", stl, "--export-dir", work_dir],
                       cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-500:]
    assert os.path.exists(stl) and os.path.getsize(stl) > 100000
    data = json.load(open(os.path.join(work_dir, "semantic_report.json"), encoding="utf-8"))
    assert data["regions"][0]["name"] == "Fluid Domain"
    assert os.path.exists(os.path.join(work_dir, "summary.txt"))
    assert os.path.exists(os.path.join(work_dir, "array_02_Float8_n4236.npy"))


def test_gui_export_stl(app, work_dir):
    from star_gui import StarMainWindow
    import time
    win = StarMainWindow()
    win.show()
    win.load_file(SIM)
    t0 = time.time()
    while win.sim is None and time.time() - t0 < 30:
        app.processEvents()
        time.sleep(0.02)
    out = os.path.join(work_dir, "gui_mesh.stl")
    win.export_stl_to(out)
    assert os.path.exists(out) and os.path.getsize(out) > 100000
    win.close()

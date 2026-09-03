# -*- coding: utf-8 -*-
"""N 波 GUI 接线：Mesh>GenerateVolume 动作→本地体网格内核（minimal 平台）。"""
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


def test_generate_volume_action_wired(app):
    from star_gui import StarMainWindow
    win = StarMainWindow()
    win.show()
    act = win.actions.get("Mesh>GenerateVolume")
    assert act is not None
    assert act.isEnabled()          # 非 _kernel_nyi 禁用态
    # 已从 _kernel_nyi stub 换成真实内核方法
    assert hasattr(StarMainWindow, "cmd_generate_volume_mesh")
    win.close()
    app.processEvents()


def test_generate_volume_no_sim_graceful(app):
    from star_gui import StarMainWindow
    win = StarMainWindow()
    win.show()
    win.sim = None
    res = win.cmd_generate_volume_mesh()   # 应安全返回，不崩溃
    assert win._volume_mesh_result is None
    win.close()
    app.processEvents()


def test_generate_volume_happy_path(app):
    from star_gui import StarMainWindow
    from sim_parser import SimFile
    win = StarMainWindow()
    win.show()
    win.load_file(SIM)
    t0 = __import__("time").time()
    while win.sim is None and __import__("time").time() - t0 < 30:
        app.processEvents()
        __import__("time").sleep(0.02)
    assert win.sim is not None
    # 用注入的水密表面驱动内核（不依赖真实抽取的水密性）
    from mesh_io import cube_mesh
    V, F = cube_mesh(1.0)
    win.sim.extract_mesh = lambda: {"vertices": V, "faces": F}
    win.sim_path = SIM
    win.cmd_generate_volume_mesh()
    res = win._volume_mesh_result
    assert res is not None
    assert res["cells"].shape[0] > 0
    assert res["vertices"].shape[1] == 3
    win.close()
    app.processEvents()
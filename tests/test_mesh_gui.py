# -*- coding: utf-8 -*-
"""N 波 GUI 接线：Mesh>GenerateVolume/Poly/Trimmer 动作→本地网格内核（minimal 平台）。"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest

SIM = os.path.join(ROOT, "adjointWing_start.sim")


def _has(name):
    try:
        __import__(name)
        return True
    except Exception:
        return False


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


@pytest.mark.skipif(not _has("scipy"), reason="无 scipy")
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


def _open_with_cube_surface(app, sim=SIM):
    """打开真实 .sim 并注入立方体水密表面（不依赖真实抽取的水密性）。"""
    from star_gui import StarMainWindow
    from mesh_io import cube_mesh
    win = StarMainWindow()
    win.show()
    win.load_file(sim)
    t0 = __import__("time").time()
    while win.sim is None and __import__("time").time() - t0 < 30:
        app.processEvents()
        __import__("time").sleep(0.02)
    assert win.sim is not None
    V, F = cube_mesh(1.0)
    win.sim.extract_mesh = lambda: {"vertices": V, "faces": F}
    win.sim_path = sim
    return win


# ---------------------------------------------------------------- N3b poly
def test_generate_poly_action_wired(app):
    from star_gui import StarMainWindow
    win = StarMainWindow()
    win.show()
    act = win.actions.get("Mesh>GeneratePoly")
    assert act is not None
    assert act.isEnabled()
    assert hasattr(StarMainWindow, "cmd_generate_poly_mesh")
    win.close()
    app.processEvents()


def test_generate_poly_no_sim_graceful(app):
    from star_gui import StarMainWindow
    win = StarMainWindow()
    win.show()
    win.sim = None
    win.cmd_generate_poly_mesh()          # 应安全返回，不崩溃
    assert win._poly_mesh_result is None
    win.close()
    app.processEvents()


@pytest.mark.skipif(not _has("scipy"), reason="无 scipy")
def test_generate_poly_happy_path(app):
    win = _open_with_cube_surface(app)
    win.cmd_generate_poly_mesh()
    res = win._poly_mesh_result
    assert res is not None
    assert res["n_cells"] > 0
    assert res["volume_total"] > 0.9        # 立方体体积 ≈ 1
    assert res["method"].startswith("voronoi_dual")
    win.close()
    app.processEvents()


# ------------------------------------------------------------- N3b trimmer
def test_generate_trimmer_action_wired(app):
    from star_gui import StarMainWindow
    win = StarMainWindow()
    win.show()
    act = win.actions.get("Mesh>GenerateTrimmer")
    assert act is not None
    assert act.isEnabled()
    assert hasattr(StarMainWindow, "cmd_generate_trimmer_mesh")
    win.close()
    app.processEvents()


def test_generate_trimmer_no_sim_graceful(app):
    from star_gui import StarMainWindow
    win = StarMainWindow()
    win.show()
    win.sim = None
    win.cmd_generate_trimmer_mesh()      # 应安全返回，不崩溃
    assert win._trimmer_mesh_result is None
    win.close()
    app.processEvents()


def test_generate_trimmer_happy_path(app):
    win = _open_with_cube_surface(app)
    win.cmd_generate_trimmer_mesh()
    res = win._trimmer_mesh_result
    assert res is not None
    assert res["n_cells"] > 0
    assert res["n_hex"] > 0
    assert res["n_cut"] > 0
    assert res["volume_total"] > 0.9      # 立方体体积 ≈ 1
    win.close()
    app.processEvents()
# -*- coding: utf-8 -*-
"""U3：3D 视口 — 法向、视图预设、方向指示器（左下）、当前视口。"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest
import numpy as np

SIM = os.path.join(ROOT, "adjointWing_start.sim")


@pytest.fixture
def work_dir():
    d = os.path.join(ROOT, "_tmp_tests")
    os.makedirs(d, exist_ok=True)
    return d


def test_mesh_polydata_has_normals():
    from star_gui_vtk import mesh_polydata
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2]], dtype=int)
    pd = mesh_polydata(verts, faces, one_based=False)
    assert pd.GetNumberOfCells() == 1
    nd = pd.GetPointData().GetNormals()
    assert nd is not None
    assert nd.GetNumberOfTuples() >= 3


def test_view_presets_move_camera():
    import vtk
    from star_gui_vtk import apply_view_preset, VIEW_PRESETS
    ren = vtk.vtkRenderer()
    cam = ren.GetActiveCamera()
    cam.SetFocalPoint(0, 0, 0)
    cam.SetPosition(0, -10, 0)
    assert apply_view_preset(ren, "+x")
    pos = cam.GetPosition()
    assert pos[0] > pos[1] and pos[0] > pos[2]
    assert apply_view_preset(ren, "iso")
    pos = cam.GetPosition()
    assert pos[0] > 0 and pos[1] > 0 and pos[2] > 0
    for name in VIEW_PRESETS:
        assert apply_view_preset(ren, name)
    assert apply_view_preset(ren, "nope") is False


def test_orientation_marker_bottom_left():
    import vtk
    from star_gui_vtk import orientation_marker_widget
    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(rw)
    marker = orientation_marker_widget(iren)
    vp = marker.GetViewport()
    assert vp[0] == 0.0 and vp[1] == 0.0
    assert vp[2] > 0.1 and vp[3] > 0.1
    marker.SetEnabled(0)


def test_offscreen_render_with_normals(work_dir):
    from sim_parser import SimFile
    from star_gui_vtk import build_mesh_actors, render_offscreen_png
    sim = SimFile(SIM)
    actors = build_mesh_actors(sim)
    out = os.path.join(work_dir, "u3_mesh.png")
    render_offscreen_png(actors, out)
    assert os.path.getsize(out) > 5000


@pytest.fixture(scope="module")
def app():
    from PyQt5.QtWidgets import QApplication
    qapp = QApplication.instance() or QApplication([])
    return qapp


def test_current_viewport_after_load(app):
    from star_gui import StarMainWindow
    import time
    win = StarMainWindow()
    win.show()
    win.load_file(SIM)
    t0 = time.time()
    while win.model is None and time.time() - t0 < 30:
        app.processEvents()
        time.sleep(0.02)
    vp = win.current_viewport()
    # 无头模式：占位 QLabel 没有 fit_view，current_viewport 应为 None
    # 有头模式：应为 Star3DViewport
    if os.environ.get("QT_QPA_PLATFORM", "").lower() in ("minimal", "offscreen"):
        assert vp is None or not hasattr(vp, "renderer")
    else:
        assert vp is not None
        assert hasattr(vp, "set_view")
    win.close()

# -*- coding: utf-8 -*-
"""CCM 导入：面流三角化（无 DLL）+ 有 ccmio.dll 时读回。"""
import os
import sys
import tempfile
import shutil

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest

from ccm_io import ccmio_available

SIM = os.path.join(ROOT, "adjointWing_start.sim")
CCMIO_OK = ccmio_available()


def test_triangulate_ccm_face_stream():
    from ccm_io import triangulate_face_stream

    assert triangulate_face_stream([3, 1, 2, 3]) == [[0, 1, 2]]
    assert triangulate_face_stream([4, 1, 2, 3, 4]) == [[0, 1, 2], [0, 2, 3]]
    assert triangulate_face_stream([3, 1, 2, 3, 3, 5, 6, 7]) == [
        [0, 1, 2], [4, 5, 6],
    ]
    assert triangulate_face_stream([]) == []


def test_ccmio_probe_is_bool():
    from ccm_io import ccmio_available, ccmio_unavailable_reason

    ok = ccmio_available()
    assert isinstance(ok, bool)
    if not ok:
        assert ccmio_unavailable_reason()


def _write_demo_ccm(path):
    from ccm_io import _gph2ccm_root
    root = _gph2ccm_root()
    if not root:
        pytest.skip("gph2ccm 不在")
    if root not in sys.path:
        sys.path.insert(0, root)
    sys.path.insert(0, os.path.join(root, "tools"))
    from make_demo_ccm import make_hex_mesh
    from gph2ccm.convert import convert_model
    mesh = make_hex_mesh(2, 2, 2)
    convert_model(mesh, path, title="stardecoding_demo", verbose=False, compress=False)


@pytest.mark.skipif(not CCMIO_OK, reason="ccmio.dll 或 gph2ccm 不可用")
def test_read_ccm_demo_hex():
    from ccm_io import read_ccm
    tmp = tempfile.mkdtemp(prefix="star_ccm_")
    try:
        path = os.path.join(tmp, "demo.ccm")
        _write_demo_ccm(path)
        mesh = read_ccm(path)
        assert mesh["n_vertices"] == 27
        assert mesh["n_cells"] == 8
        assert mesh["faces"]
        assert mesh["n_boundary_faces"] > 0
        lo = min(min(t) for t in mesh["faces"])
        hi = max(max(t) for t in mesh["faces"])
        assert lo >= 0 and hi < mesh["n_vertices"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def app():
    from PyQt5.QtWidgets import QApplication
    qapp = QApplication.instance() or QApplication([])
    return qapp


def test_import_volume_without_file(app):
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
    assert win.import_volume_from_path(os.path.join(ROOT, "no_such.ccm")) is None
    assert "CCM" in win.messages.view.toPlainText()
    win.document.mark_clean()
    win.close()


@pytest.mark.skipif(not CCMIO_OK, reason="ccmio.dll 或 gph2ccm 不可用")
def test_gui_import_volume_ccm(app):
    from star_gui import StarMainWindow
    import time
    tmp = tempfile.mkdtemp(prefix="star_ccm_gui_")
    try:
        path = os.path.join(tmp, "demo.ccm")
        _write_demo_ccm(path)
        win = StarMainWindow()
        win.show()
        win.load_file(SIM)
        t0 = time.time()
        while win.sim is None and time.time() - t0 < 30:
            app.processEvents()
            time.sleep(0.02)
        oid = win.import_volume_from_path(path)
        assert oid is not None
        part = win.document.object(oid)
        assert part is not None
        assert len(part.dict.get("ImportedFaces") or []) >= 12
        assert int(part.dict.get("CcmCellCount") or 0) == 8
        assert "已导入 CCM" in win.messages.view.toPlainText()
        win.document.mark_clean()
        win.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

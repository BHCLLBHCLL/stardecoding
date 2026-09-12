# -*- coding: utf-8 -*-
"""X2：多仿真文档窗口 + 跨仿真复制粘贴（GUI 集成）。"""
import os
import shutil
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest

SIM = os.path.join(ROOT, "adjointWing_start.sim")


@pytest.fixture(scope="module")
def app():
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _wait_loaded(win, app, timeout=30):
    t0 = time.time()
    while win.sim is None and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.02)
    assert win.sim is not None


def _open_copy(app, tmp, name="wing.sim"):
    """把基线算例拷到临时目录再载入，避免写坏仓库里的原始文件。"""
    from star_gui import StarMainWindow
    dest = os.path.join(tmp, name)
    shutil.copy2(SIM, dest)
    win = StarMainWindow()
    win.show()
    win.load_file(dest)
    _wait_loaded(win, app)
    return win, dest


def _region(win):
    return next(o for o in win.sim.objects
                if o.dict.get("PresentationName") == "Fluid Domain"
                and o.class_name == "star.common.Region")


def _close(*wins):
    for w in wins:
        try:
            w.document.mark_clean()
            w.close()
        except Exception:
            pass


def test_file_menu_has_new_window(app):
    from star_gui import StarMainWindow
    win = StarMainWindow()
    assert "File>New Window" in win.actions
    _close(win)


def test_new_window_opens_independent_document(app):
    from star_gui import StarMainWindow
    from star_gui_documents import WORKSPACE
    win = StarMainWindow()
    other = win.cmd_new_window()
    assert other is not win
    assert other.document is not win.document
    assert WORKSPACE.owner_of(other.document) is other
    assert other.document in WORKSPACE.documents
    _close(other)
    assert other.document not in WORKSPACE.documents      # 关窗即注销
    _close(win)


def test_same_document_paste_uses_command_bus(app):
    tmp = tempfile.mkdtemp(prefix="star_x2_")
    try:
        win, _dest = _open_copy(app, tmp, "a.sim")
        obj = _region(win)
        win.tree_widget.select_object(obj.id)
        assert win._selected_obj() is not None
        win.cmd_copy()
        before = len(win.document.created)
        win.cmd_paste()                                    # 同文档 → 走 CopyObjectCommand
        assert len(win.document.created) == before + 1
        assert win.document.dirty
        _close(win)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cross_document_paste_remaps_and_attaches(app):
    from star_gui_documents import CLIPBOARD
    tmp = tempfile.mkdtemp(prefix="star_x2_")
    try:
        win_a, _da = _open_copy(app, tmp, "a.sim")
        win_b, _db = _open_copy(app, tmp, "b.sim")
        obj = _region(win_a)
        win_a.tree_widget.select_object(obj.id)
        win_a.cmd_copy()
        assert CLIPBOARD.has_content()
        assert CLIPBOARD.is_cross(win_b.document)
        before = set(win_b.sim.objmap)
        new_root = win_b.cmd_paste()
        assert new_root is not None
        assert new_root not in before                       # id 不冲突
        pasted = win_b.document.object(new_root)
        assert pasted.dict["PresentationName"].endswith(" Copy")
        mgr = next(o for o in win_b.sim.objects
                   if o.class_name == "star.common.RegionManager")
        assert pasted.dict["Parent"] == mgr.id              # 挂到目标同类节点
        assert new_root in mgr.dict["Keys"]
        assert win_b.document.dirty
        assert win_a.document.created == {}                 # 源文档不受影响
        _close(win_a, win_b)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_save_all_covers_all_open_documents(app):
    from star_gui_commands import SetPropertyCommand
    from sim_parser import SimFile
    tmp = tempfile.mkdtemp(prefix="star_x2_")
    try:
        win_a, dest_a = _open_copy(app, tmp, "a.sim")
        win_b, dest_b = _open_copy(app, tmp, "b.sim")
        for win in (win_a, win_b):
            obj = next(o for o in win.sim.objects if o.dict.get("PresentationName"))
            win.document.execute(SetPropertyCommand(
                obj.id, "PresentationName", "Renamed X",
                obj.dict.get("PresentationName")))
        assert win_a.cmd_save_all() is True
        assert not win_a.document.dirty
        assert not win_b.document.dirty
        for dest in (dest_a, dest_b):
            assert any(o.dict.get("PresentationName") == "Renamed X"
                       for o in SimFile(dest).objects)
        _close(win_a, win_b)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

# -*- coding: utf-8 -*-
"""X1：会话生命周期 GUI 集成（Save All / 备份 / AutoSave / CHECKPOINT / 模板）。"""
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


def _session_items(win):
    return ("File>Save All", "File>Save As Template", "File>New from Template",
            "File>AutoSave", "File>AutoSave Now", "File>Checkpoint")


def test_file_menu_has_session_items(app):
    from star_gui import StarMainWindow
    win = StarMainWindow()
    for key in _session_items(win):
        assert key in win.actions, key
    assert win.actions["File>AutoSave"].isCheckable()
    win.close()


def test_save_all_without_session(app):
    from star_gui import StarMainWindow
    win = StarMainWindow()
    assert win.cmd_save_all() is False
    win.close()


def test_save_all_writes_backup_and_cleans(app):
    from star_gui_commands import SetPropertyCommand
    from sim_parser import SimFile
    tmp = tempfile.mkdtemp(prefix="star_x1_")
    try:
        win, dest = _open_copy(app, tmp)
        assert win.cmd_save_all() is True           # 覆盖既有会话文件 → 写前备份
        assert os.path.isfile(dest)
        assert not win.document.dirty
        assert os.path.exists(dest + "~")

        obj = next(o for o in win.sim.objects
                   if o.dict.get("PresentationName") == "Fluid Domain"
                   and o.class_name == "star.common.Region")
        old = obj.dict["PresentationName"]
        win.document.execute(
            SetPropertyCommand(obj.id, "PresentationName", "Fluid Domain X", old))
        assert win.document.dirty
        assert win.cmd_save_all() is True           # 二次落盘 → 生成 path~
        assert os.path.isfile(dest + "~")
        assert SimFile(dest).objmap[obj.id].dict["PresentationName"] == "Fluid Domain X"
        assert SimFile(dest + "~").objmap[obj.id].dict["PresentationName"] == old
        win.document.mark_clean()
        win.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_snapshot_writes_at_n_and_rotates(app):
    tmp = tempfile.mkdtemp(prefix="star_x1_")
    try:
        win, dest = _open_copy(app, tmp)
        win.autosave_policy.keep = 2
        paths = [win._do_snapshot("自动") for _ in range(3)]
        assert [os.path.basename(p) for p in paths] == [
            "wing@1.sim", "wing@2.sim", "wing@3.sim"]
        remaining = sorted(n for n in os.listdir(tmp) if "@" in n)
        assert remaining == ["wing@2.sim", "wing@3.sim"]   # 仅留最近 keep=2
        assert win.sim_path == dest                        # 快照不改会话路径
        win.autosave_policy.keep = 3
        win.document.mark_clean()
        win.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_autosave_tick_requires_dirty(app):
    from star_gui_commands import SetPropertyCommand
    tmp = tempfile.mkdtemp(prefix="star_x1_")
    try:
        win, dest = _open_copy(app, tmp)
        win.autosave_policy.enabled = True
        win.document.mark_clean()
        win._on_autosave_tick()                            # 不脏 → 不产快照
        assert not [n for n in os.listdir(tmp) if "@" in n]
        obj = next(o for o in win.sim.objects if o.dict.get("PresentationName"))
        win.document.execute(
            SetPropertyCommand(obj.id, "PresentationName", "Dirty", obj.dict.get("PresentationName")))
        win._on_autosave_tick()                            # 脏 → 产快照
        assert [n for n in os.listdir(tmp) if "@" in n]
        win.autosave_policy.enabled = False
        win.document.mark_clean()
        win.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_checkpoint_trigger_file_is_consumed(app):
    tmp = tempfile.mkdtemp(prefix="star_x1_")
    try:
        win, dest = _open_copy(app, tmp)
        trigger = os.path.join(tmp, "stop.trigger")
        win.autosave_policy.enabled = False                # 断点保存不依赖 AutoSave 开关
        win.autosave_policy.trigger = trigger
        with open(trigger, "w") as f:
            f.write("go")
        win._on_autosave_tick()
        assert not os.path.exists(trigger)                 # 触发文件被消费
        assert [n for n in os.listdir(tmp) if "@" in n]    # 已产出快照
        win.document.mark_clean()
        win.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_save_template_keeps_session_path(app):
    from sim_parser import SimFile
    from star_gui_session import template_path
    tmp = tempfile.mkdtemp(prefix="star_x1_")
    try:
        win, dest = _open_copy(app, tmp)
        tpath = template_path(dest)
        assert win._write_sim(tpath, update_state=False, backup=False)
        assert os.path.isfile(tpath)
        assert win.sim_path == dest                        # 模板写出不改会话路径
        assert SimFile(tpath).objects
        win.document.mark_clean()
        win.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_new_from_template_clears_path(app):
    from star_gui import StarMainWindow
    tmp = tempfile.mkdtemp(prefix="star_x1_")
    try:
        tpl = os.path.join(tmp, "base.simt")
        shutil.copy2(SIM, tpl)
        win = StarMainWindow()
        win.show()
        win.load_file(tpl)
        _wait_loaded(win, app)
        assert win.from_template is True
        assert win.sim_path is None                        # 模板 → 首次保存走另存为 .sim
        assert "模板" in win.windowTitle()
        win.document.mark_clean()
        win.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_toggle_autosave_action(app):
    from star_gui import StarMainWindow
    win = StarMainWindow()
    win.autosave_policy.enabled = False
    win.cmd_toggle_autosave(True)
    assert win.autosave_policy.enabled is True
    assert win.actions["File>AutoSave"].isChecked()
    win.cmd_toggle_autosave(False)
    assert win.autosave_policy.enabled is False
    assert not win.actions["File>AutoSave"].isChecked()
    win.close()

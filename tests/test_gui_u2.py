# -*- coding: utf-8 -*-
"""U2：双行工具栏 + 中文菜单（文件/网格/求解/窗口）+ 视图按钮。"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest


@pytest.fixture(scope="module")
def app():
    from PyQt5.QtWidgets import QApplication
    qapp = QApplication.instance() or QApplication([])
    return qapp


def test_chinese_menus_and_actions(app):
    from star_gui import StarMainWindow
    win = StarMainWindow()
    titles = [a.text().replace("&", "") for a in win.menuBar().actions()]
    blob = " ".join(titles)
    for name in ("文件", "编辑", "网格", "场景", "求解", "工具", "连接", "窗口", "帮助"):
        assert name in blob, titles
    assert win.actions["File>Open"].text() == "打开..."
    assert win.actions["Scene>View>+x"] is not None
    assert win.actions["Solution>Run"] is not None
    assert win.actions["Window>Tree"].isCheckable()
    win.close()


def test_dual_toolbars(app):
    from star_gui import StarMainWindow
    win = StarMainWindow()
    names = [tb.objectName() for tb in win.findChildren(win.tb_file.__class__)]
    for n in ("File", "Solve", "View", "Display"):
        assert n in names, names
    assert win.tb_view.actions()
    win.close()


def test_painted_view_icons(app):
    from star_gui_icons import AppIcons
    ic = AppIcons()
    assert not ic.get("view_+x").isNull()
    assert not ic.get("view_iso").isNull()
    assert not ic.get("solid").isNull()


def test_i18n_tr():
    from star_gui_i18n import tr, set_language
    set_language("zh")
    assert tr("File") == "文件"
    assert tr("Fit View") == "适配视图"
    set_language("en")
    assert tr("File") == "File"
    set_language("zh")

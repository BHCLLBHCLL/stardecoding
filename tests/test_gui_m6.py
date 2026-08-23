# -*- coding: utf-8 -*-
"""M6 冒烟：主题 QSS / 图标引擎 / 帮助对话框（minimal 平台）。"""
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


def test_theme_qss_exists():
    qss = os.path.join(ROOT, "star_gui_theme.qss")
    assert os.path.exists(qss)
    with open(qss, encoding="utf-8") as f:
        text = f.read()
    assert "QTreeWidget" in text
    assert "#005f78" in text
    assert "#f0f0f0" in text
    assert "#cde8f6" in text


def test_theme_applies(app):
    import star_gui
    star_gui._apply_theme(app)
    assert app.styleSheet() != ""


def test_icons():
    from star_gui_icons import AppIcons
    icons = AppIcons()
    assert not icons.get("open").isNull()
    assert not icons.get("part").isNull()
    assert not icons.get("layer_physics").isNull()
    # 不同功能应画出不同像素，避免再退化成同一套通用图标
    pm_open = icons.get("open").pixmap(24, 24)
    pm_play = icons.get("play").pixmap(24, 24)
    pm_scene = icons.get("scene").pixmap(24, 24)
    assert pm_open.toImage() != pm_play.toImage()
    assert pm_play.toImage() != pm_scene.toImage()


def test_about_message(app):
    from star_gui import StarMainWindow
    win = StarMainWindow()
    win.show()
    # About 对话框存在（不弹出，仅验证动作注册）
    assert win.actions.get("Help>About") is not None
    assert win.actions.get("Scene>Edges") is not None
    win.close()

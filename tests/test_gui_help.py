# -*- coding: utf-8 -*-
"""X3：帮助系统（doc_javadoc_catalog 联动）+ 关于/licensing UI。"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest

import star_gui_help as hs


@pytest.fixture(scope="module")
def app():
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_catalog_parses_packages():
    st = hs.stats()
    assert st["packages"] >= 50
    assert st["sections"] >= 15
    pkgs = hs.packages()
    assert "star.common" in pkgs and "star.vis" in pkgs and "star.meshing" in pkgs


def test_package_description():
    assert "Simulation" in hs.package_description("star.common")
    assert hs.package_description("") == ""


def test_semantic_layer():
    assert hs.semantic_layer("star.common.Region") == ("core", "仿真主干")
    assert hs.semantic_layer("star.vis.Scene") == ("visualization", "场景可视化")
    assert hs.semantic_layer("star.zzz.Foo") == ("unknown", "未分类")


def test_resolve_alias_and_unknown():
    assert hs.explain("star.vis.View")["resolved"] == "star.vis.VisView"
    info = hs.explain("star.common.Region")
    assert info["package"] == "star.common"
    assert info["layer_cn"] == "仿真主干"
    assert hs.package_of("star.zzz.Foo") == "star.zzz"


def test_class_help_text():
    text = hs.class_help_text("star.common.Region")
    assert "star.common.Region" in text
    assert "仿真主干" in text
    assert hs.class_help_text("") == "未选择对象。"


def test_search():
    assert hs.search("") == []
    hits = hs.search("turbulence")
    assert hits and all("turbulence" in h["name"].lower() or "turbulence" in h["text"].lower()
                        for h in hits)


def test_documentation_and_licensing_text():
    doc = hs.documentation_text()
    assert hs.CATALOG_PATH in doc
    assert "doc_javadoc_catalog.md" in doc
    lic = hs.licensing_text()
    assert hs.APP_NAME in lic and "PyQt5" in lic
    assert hs.APP_NAME in hs.about_text()


def test_help_menu_items(app):
    from star_gui import StarMainWindow
    win = StarMainWindow()
    for key in ("Help>Help Contents", "Help>Documentation", "Help>Licensing", "Help>About"):
        assert key in win.actions, key
    assert win.actions["Help>Help Contents"].shortcut().toString() == "F1"
    win.close()


def test_help_window_documentation_topic(app):
    from star_gui import StarMainWindow
    win = StarMainWindow()
    hw = win.cmd_help_documentation()
    assert "doc_javadoc_catalog.md" in hw.current_text()
    win.close()


def test_help_window_license_topic(app):
    from star_gui_helpwin import HelpWindow
    hw = HelpWindow()
    hw.show_topic("许可信息")
    assert "PyQt5" in hw.current_text()
    hw.show_topic("帮助内容")
    assert "包总览" in hw.current_text()


def test_help_window_context_class(app):
    from star_gui_helpwin import HelpWindow
    hw = HelpWindow(class_name="star.common.Region")
    assert "仿真主干" in hw.current_text()
    assert hw.topics.currentItem() is not None
    assert hw.topics.currentItem().text() == "star.common"


def test_help_search_filters_topics(app):
    from star_gui_helpwin import HelpWindow
    hw = HelpWindow()
    hw.search.setText("turbulence")
    assert hw.topics.count() >= 1
    for i in range(hw.topics.count()):
        assert "turbulence" in hw.topics.item(i).text().lower()

# -*- coding: utf-8 -*-
"""star_gui_helpwin.py — 帮助窗口（QDialog）。

左侧主题列表（包总览 + 帮助内容/文档目录/许可信息），右侧文本浏览，
顶部检索框。可由选中对象的 ClassName 打开，直接展示该类语义帮助。
"""
from PyQt5.QtWidgets import (
    QDialog, QHBoxLayout, QLineEdit, QListWidget, QListWidgetItem, QSplitter,
    QTextBrowser, QVBoxLayout, QWidget,
)

import star_gui_help as help_

SPECIAL_TOPICS = ("帮助内容", "文档目录", "许可信息")


class HelpWindow(QDialog):
    def __init__(self, parent=None, class_name=None):
        super().__init__(parent)
        from star_gui_i18n import tr
        self.setWindowTitle(tr("Help Contents"))
        self.resize(860, 620)
        self._class_name = class_name

        self.search = QLineEdit(self)
        self.search.setPlaceholderText("检索包 / 类关键词…")
        self.search.textChanged.connect(self._populate)

        self.topics = QListWidget(self)
        self.topics.currentItemChanged.connect(self._on_topic)

        self.view = QTextBrowser(self)
        self.view.setOpenExternalLinks(False)

        left = QWidget(self)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(self.search)
        lv.addWidget(self.topics)
        split = QSplitter(self)
        split.addWidget(left)
        split.addWidget(self.view)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)
        lay = QVBoxLayout(self)
        lay.addWidget(split)

        self._populate()
        if class_name:
            self.show_class(class_name)

    def _populate(self, *_):
        query = self.search.text().strip().lower()
        self.topics.blockSignals(True)
        self.topics.clear()
        for name in SPECIAL_TOPICS:
            if not query or query in name.lower():
                QListWidgetItem(name, self.topics)
        for pkg, desc in help_.packages().items():
            if not query or query in pkg.lower() or query in desc.lower():
                QListWidgetItem(pkg, self.topics)
        self.topics.blockSignals(False)
        if self.topics.count():
            self.topics.setCurrentRow(0)

    def _on_topic(self, item, _prev=None):
        if item is None:
            return
        self.show_topic(item.text())

    def show_topic(self, name):
        if name == "帮助内容":
            self.view.setPlainText(help_.help_contents_text())
        elif name == "文档目录":
            self.view.setPlainText(help_.documentation_text())
        elif name == "许可信息":
            self.view.setPlainText(help_.licensing_text())
        else:
            self.view.setPlainText(self._package_text(name))

    def _package_text(self, pkg):
        desc = help_.package_description(pkg)
        sec = help_.package_section(pkg)
        lines = ["包：%s" % pkg, "作用：%s" % (desc or "—")]
        if sec is not None:
            lines.append("章节：%s" % sec["title"])
            lines.append("")
            for b in sec["bullets"]:
                lines.append("  · %s" % b)
        return "\n".join(lines)

    def show_class(self, class_name):
        self._class_name = class_name
        self.view.setPlainText(help_.class_help_text(class_name))
        pkg = help_.package_of(class_name)
        for i in range(self.topics.count()):
            if self.topics.item(i).text() == pkg:
                self.topics.setCurrentRow(i)
                break

    def current_text(self):
        return self.view.toPlainText()

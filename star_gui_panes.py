# -*- coding: utf-8 -*-
"""star_gui_panes.py — 主窗口窗格。

对齐 cabdecoding/cstdecoding 的 PaneFrame/MessageWindow/ProgressPanel 范式：
- PaneFrame: 带标题栏的 QFrame 容器
- MessageWindow: 底部日志（时间戳 + 级别 + 文本）
- ProgressPanel: 进度条 + 文本
- StatusBar 辅助：坐标/统计文本更新
M0 先提供骨架；M1 加 SimulationTree/PropertiesPanel，M2 加 GraphicsTabs/3DViewport。
"""

import time

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QProgressBar,
    QToolButton, QVBoxLayout, QWidget,
)


class PaneFrame(QFrame):
    """带标题栏的窗格容器（对齐 cabdecoding PaneFrame）。"""

    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(2, 2, 2, 2)
        self._outer.setSpacing(2)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("paneTitle")
        self.title_label.setStyleSheet("font-weight: bold; padding: 1px;")
        self._outer.addWidget(self.title_label)
        self.body = QWidget()
        self._body_layout = QVBoxLayout(self.body)
        self._body_layout.setContentsMargins(2, 2, 2, 2)
        self._outer.addWidget(self.body, 1)

    def body_layout(self):
        return self._body_layout

    def set_body(self, widget):
        self._body_layout.addWidget(widget, 1)

    def set_title(self, title):
        self.title_label.setText(title)


class MessageWindow(QWidget):
    """底部消息/日志窗口（时间戳 + 级别）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setMaximumBlockCount(2000)
        lay.addWidget(self.view)

    def log(self, text, level="info"):
        stamp = time.strftime("%H:%M:%S")
        prefix = {"info": " * ", "warn": " ! ", "error": " X ", "nyi": " ~ "}.get(level, " * ")
        self.view.appendPlainText("%s%s%s" % (stamp, prefix, text))

    def nyi(self, name):
        """未实现功能占位（对齐 cabdecoding _nyi 文案）。"""
        self.log("[%s] not available in star_gui viewer (STAR-CCM+ only / not yet mapped)." % name,
                 "nyi")


class ProgressPanel(QWidget):
    """进度条 + 文本。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        self.label = QLabel("就绪")
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setFixedWidth(220)
        lay.addWidget(self.label, 1)
        lay.addWidget(self.bar)

    def set_progress(self, value, text=""):
        self.bar.setValue(value)
        if text:
            self.label.setText(text)

    def done(self, text=""):
        self.bar.setValue(100)
        if text:
            self.label.setText(text)


class StatusBarHelper(QWidget):
    """状态栏左侧辅助：坐标/统计文本。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 0, 2, 0)
        self.coord = QLabel("(x,y,z): -")
        self.stats = QLabel("")
        self.unit = QLabel("")
        for lbl in (self.coord, self.stats, self.unit):
            lbl.setStyleSheet("padding-left: 8px;")
            lay.addWidget(lbl)
        lay.addStretch(1)

    def set_coord(self, xyz):
        self.coord.setText("(x,y,z): (%s)" % ", ".join("%.4g" % v for v in xyz))

    def set_stats(self, text):
        self.stats.setText(text)

    def set_unit(self, text):
        self.unit.setText("单位: %s" % text)


class SummaryPane(QWidget):
    """M0 摘要窗格：打开文件后的总体信息（后续并入 GraphicsTabs 的 Info 页）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        lay.addWidget(self.view)

    def show_summary(self, text):
        self.view.setPlainText(text)

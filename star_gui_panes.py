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
    QTableWidget, QTableWidgetItem, QTreeWidget, QToolButton,
    QVBoxLayout, QWidget,
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


class SimulationTree(QWidget):
    """M1 仿真树：对象图 → QTreeWidget，节点携带 obj_id（UserRole）。"""

    object_selected = pyqtSignal(object)   # SimObject 或 None

    def __init__(self, model=None, icons=None, parent=None):
        super().__init__(parent)
        self.model = model
        self._icons = icons
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Simulation Tree"])
        self.tree.itemSelectionChanged.connect(self._on_selection)
        lay.addWidget(self.tree)

    def set_model(self, model):
        self.model = model
        self.rebuild()

    def rebuild(self):
        self.tree.clear()
        if self.model is None:
            return
        from star_gui_model import Node
        for root in self.model.tree_roots():
            item = self._make_item(root)
            self.tree.addTopLevelItem(item)
        self.tree.expandToDepth(1)

    def _make_item(self, node):
        from PyQt5.QtWidgets import QTreeWidgetItem
        item = QTreeWidgetItem([node.label])
        item.setData(0, 32, node.obj_id)   # Qt.UserRole = 32
        if node.obj_id is not None:
            item.setData(0, 33, node.class_name or "")
        if self._icons is not None:
            item.setIcon(0, self._icons.get(self._icon_key(node)))
        for child in node.children:
            item.addChild(self._make_item(child))
        return item

    def _icon_key(self, node):
        layer_map = {
            "cad-geometry": "layer_geometry",
            "meshing": "layer_meshing",
            "physics": "layer_physics",
            "visualization": "layer_visualization",
            "post-processing": "layer_post",
        }
        return layer_map.get(node.layer, node.layer)

    def _on_selection(self):
        items = self.tree.selectedItems()
        if not items:
            self.object_selected.emit(None)
            return
        item = items[0]
        oid = item.data(0, 32)
        obj = self.model.object_by_id(oid) if (self.model and oid is not None) else None
        self.object_selected.emit(obj)

    def select_object(self, oid):
        """按对象 id 递归查找并选中（供 3D 拾取联动）。"""
        def walk(item):
            if item.data(0, 32) == oid:
                return item
            for i in range(item.childCount()):
                hit = walk(item.child(i))
                if hit is not None:
                    return hit
            return None
        for i in range(self.tree.topLevelItemCount()):
            hit = walk(self.tree.topLevelItem(i))
            if hit is not None:
                self.tree.setCurrentItem(hit)
                self.tree.scrollToItem(hit)
                return


class PropertiesPanel(QWidget):
    """M1 属性面板：选中对象的属性表（名称/类型/值三列 + 引用跳转）。"""

    reference_activated = pyqtSignal(object)

    def __init__(self, model=None, parent=None):
        super().__init__(parent)
        self.model = model
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["属性", "类型", "值"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 150)
        self.table.setColumnWidth(1, 110)
        self.table.setEditTriggers(self.table.NoEditTriggers)
        self.table.cellDoubleClicked.connect(self._on_double)
        lay.addWidget(self.table)
        self._current = None

    def set_model(self, model):
        self.model = model

    def show_object(self, obj):
        self._current = obj
        self.table.setRowCount(0)
        if obj is None:
            return
        rows = self.model.properties(obj) if self.model else []
        for attr, val, raw in rows:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(attr))
            self.table.setItem(row, 1, QTableWidgetItem(type(raw).__name__))
            vitem = QTableWidgetItem(val)
            vitem.setData(34, raw)   # Qt.UserRole+2：原始值
            self.table.setItem(row, 2, vitem)

    def _on_double(self, row, col):
        if col != 2 or self.model is None:
            return
        item = self.table.item(row, 2)
        raw = item.data(34) if item else None
        if isinstance(raw, int) and raw in self.model.objmap:
            self.reference_activated.emit(self.model.objmap[raw])

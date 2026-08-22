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
    QTableWidget, QTableWidgetItem, QTabWidget, QTreeWidget, QToolButton,
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


class Star3DViewport(QWidget):
    """3D 视口：QVTK + 方向指示器 + 六向/等轴测 + 拾取（对齐 cab_vtk / STAR-CCM+）。"""

    picked = pyqtSignal(object, object)   # (info_tuple, xyz)

    def __init__(self, parent=None):
        super().__init__(parent)
        from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
        from vtkmodules.vtkRenderingCore import vtkRenderer
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        lay.addWidget(self.vtk_widget)
        self.renderer = vtkRenderer()
        self.renderer.SetBackground(0.10, 0.12, 0.16)
        self.renderer.SetBackground2(0.04, 0.05, 0.07)
        try:
            self.renderer.GradientBackgroundOn()
        except Exception:
            pass
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.actors = []          # [(key, name, part_id, vtkActor)]
        self._by_actor = {}
        self._picker = None
        self._orient = None
        self._rep_mode = "solid"   # solid | wireframe | edges
        self._opacity = 1.0
        self._init_interactor()
        self._init_picker()
        self._init_orientation_marker()
        self.setFocusPolicy(Qt.StrongFocus)

    def _init_interactor(self):
        try:
            self.vtk_widget.Initialize()
        except Exception:
            pass

    def _init_orientation_marker(self):
        from star_gui_vtk import orientation_marker_widget
        try:
            iren = self.vtk_widget.GetRenderWindow().GetInteractor()
            self._orient = orientation_marker_widget(iren)
        except Exception:
            self._orient = None

    def _init_picker(self):
        from vtkmodules.vtkRenderingCore import vtkCellPicker
        picker = vtkCellPicker()
        self._picker = picker
        self.vtk_widget.GetRenderWindow().GetInteractor().SetPicker(picker)
        picker.AddObserver("EndPickEvent", self._on_pick)

    def _on_pick(self, picker, event):
        actor = picker.GetActor()
        info = self._by_actor.get(id(actor))
        if info is None:
            return
        xyz = picker.GetPickPosition()
        self.picked.emit(info, (xyz[0], xyz[1], xyz[2]))

    def set_actors(self, actors):
        for k, n, pid, actor in self.actors:
            self.renderer.RemoveActor(actor)
        self.actors = list(actors)
        self._by_actor = {id(a): (k, n, pid) for k, n, pid, a in self.actors}
        for k, n, pid, actor in self.actors:
            self.renderer.AddActor(actor)
        self._apply_rep()
        self.fit_view()

    def fit_view(self):
        self.renderer.ResetCamera()
        self.render()

    def reset_view(self):
        from star_gui_vtk import apply_view_preset
        apply_view_preset(self.renderer, "iso")
        self.render()

    def set_view(self, name):
        from star_gui_vtk import apply_view_preset
        if apply_view_preset(self.renderer, name):
            self.render()
            return True
        return False

    def render(self):
        self.vtk_widget.GetRenderWindow().Render()

    def _is_edge_actor(self, key):
        return (key or "").startswith("edges:")

    def _apply_rep(self):
        mode = self._rep_mode
        for k, n, pid, actor in self.actors:
            is_edge = self._is_edge_actor(k)
            if mode == "edges":
                actor.SetVisibility(is_edge)
            else:
                actor.SetVisibility(True)
                if is_edge:
                    continue
                if mode == "wireframe":
                    actor.GetProperty().SetRepresentationToWireframe()
                else:
                    actor.GetProperty().SetRepresentationToSurface()
                actor.GetProperty().SetOpacity(self._opacity)

    def set_representation(self, mode):
        if mode == "edges_only":
            mode = "edges"
        self._rep_mode = mode if mode in ("solid", "wireframe", "edges") else "solid"
        self._apply_rep()
        self.render()

    def set_opacity(self, opacity):
        self._opacity = max(0.05, min(1.0, float(opacity)))
        self._apply_rep()
        self.render()

    def toggle_transparency(self):
        self.set_opacity(0.4 if self._opacity > 0.7 else 1.0)

    def set_background(self, rgb, rgb2=None):
        self.renderer.SetBackground(*rgb)
        if rgb2:
            self.renderer.SetBackground2(*rgb2)
            try:
                self.renderer.GradientBackgroundOn()
            except Exception:
                pass
        self.render()

    def keyPressEvent(self, event):
        key = event.key()
        shift = bool(event.modifiers() & Qt.ShiftModifier)
        if key == Qt.Key_F:
            self.fit_view()
            return
        mapping = {Qt.Key_X: "x", Qt.Key_Y: "y", Qt.Key_Z: "z"}
        axis = mapping.get(key)
        if axis:
            name = ("-" if shift else "+") + axis
            self.set_view(name)
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        try:
            if self._orient is not None:
                self._orient.SetEnabled(0)
        except Exception:
            pass
        try:
            self.vtk_widget.close()
        except Exception:
            pass
        super().closeEvent(event)


class GraphicsTabs(QTabWidget):
    """图形区：Info 标签 + 每 Scene 一页 3D。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDocumentMode(True)
        self.setTabsClosable(False)
        self.setMovable(True)

    def add_info_tab(self, summary_pane):
        self.addTab(summary_pane, "Info")

    def add_mesh_tab(self, title, viewport):
        idx = self.addTab(viewport, title)
        return idx

    def current_viewport(self):
        """当前标签若为 Star3DViewport 则返回，否则在所有标签中找第一个。"""
        w = self.currentWidget()
        if hasattr(w, "fit_view") and hasattr(w, "set_actors"):
            return w
        for i in range(self.count()):
            w = self.widget(i)
            if hasattr(w, "fit_view") and hasattr(w, "set_actors"):
                return w
        return None


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
    """M1 仿真树：对象图 → QTreeWidget，节点携带 obj_id（UserRole）。

    M5 增加勾选显隐：有 obj_id 的节点带复选框，变更时发 check_changed(obj_id, checked)。
    """

    object_selected = pyqtSignal(object)   # SimObject 或 None
    check_changed = pyqtSignal(object, bool)  # (obj_id, checked)

    def __init__(self, model=None, icons=None, parent=None):
        super().__init__(parent)
        self.model = model
        self._icons = icons
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["模型 / 场景/绘图"])
        self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(True)
        self.tree.itemSelectionChanged.connect(self._on_selection)
        self.tree.itemChanged.connect(self._on_item_changed)
        lay.addWidget(self.tree)

    def set_model(self, model):
        self.model = model
        self.rebuild()

    def rebuild(self):
        self.tree.clear()
        if self.model is None:
            return
        from star_gui_model import Node  # noqa: F401
        roots = self.model.sim_tree() if hasattr(self.model, "sim_tree") else self.model.tree_roots()
        for root in roots:
            item = self._make_item(root)
            self.tree.addTopLevelItem(item)
        self.tree.expandToDepth(2)

    def _make_item(self, node):
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QTreeWidgetItem
        item = QTreeWidgetItem([node.label])
        item.setData(0, 32, node.obj_id)   # Qt.UserRole = 32
        if node.obj_id is not None:
            item.setData(0, 33, node.class_name or "")
        if node.obj_id is not None:
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Checked)
        if self._icons is not None:
            item.setIcon(0, self._icons.get(self._icon_key(node)))
        for child in node.children:
            item.addChild(self._make_item(child))
        return item

    def _on_item_changed(self, item, column):
        if column != 0 or not (item.flags() & 2):   # ItemIsUserCheckable = 2
            return
        from PyQt5.QtCore import Qt
        oid = item.data(0, 32)
        if oid is not None:
            self.check_changed.emit(oid, item.checkState(0) == Qt.Checked)

    def _icon_key(self, node):
        cn = node.class_name or ""
        short = cn.split(".")[-1]
        if short in ("Region",):
            return "region"
        if short in ("Boundary",):
            return "boundary"
        if short in ("Scene",):
            return "scene"
        if "Plot" in short:
            return "plot"
        if "Monitor" in short:
            return "monitor"
        if "Part" in short or short.endswith("Part"):
            return "part"
        layer_map = {
            "cad-geometry": "layer_geometry",
            "meshing": "layer_meshing",
            "physics": "layer_physics",
            "visualization": "layer_visualization",
            "post-processing": "layer_post",
            "solver": "layer_physics",
            "query": "info",
            "core": "simulation",
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

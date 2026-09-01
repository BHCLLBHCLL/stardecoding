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

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView, QAction, QColorDialog, QDialog, QDialogButtonBox,
    QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMenu, QPlainTextEdit, QProgressBar, QPushButton, QTableWidget,
    QTableWidgetItem, QTabWidget, QTreeWidget, QToolButton, QVBoxLayout,
    QWidget, QHeaderView,
)


class PaneFrame(QFrame):
    """STAR-CCM+ 停靠块：浅灰标题条 + 白底内容（对齐截图块状窗格）。"""

    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setObjectName("PaneFrame")
        self.setFrameShape(QFrame.NoFrame)
        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(0, 0, 0, 0)
        self._outer.setSpacing(0)
        bar = QFrame()
        bar.setObjectName("PaneTitleBar")
        hb = QHBoxLayout(bar)
        hb.setContentsMargins(0, 0, 0, 0)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("paneTitle")
        hb.addWidget(self.title_label)
        hb.addStretch(1)
        self._outer.addWidget(bar)
        self.body = QWidget()
        self.body.setObjectName("PaneBody")
        self._body_layout = QVBoxLayout(self.body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._body_layout.setSpacing(0)
        self._outer.addWidget(self.body, 1)

    def body_layout(self):
        return self._body_layout

    def set_body(self, widget):
        self._body_layout.addWidget(widget, 1)

    def set_title(self, title):
        self.title_label.setText(title)


class MessageWindow(QWidget):
    """底部输出窗口（对齐 STAR-CCM+ Output：时间戳 + 级别）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setMaximumBlockCount(4000)
        self.view.setPlaceholderText("输出…")
        font = self.view.font()
        font.setFamily("Consolas")
        font.setPointSize(9)
        self.view.setFont(font)
        lay.addWidget(self.view)

    def log(self, text, level="info"):
        stamp = time.strftime("%H:%M:%S")
        prefix = {"info": "  ", "warn": "! ", "error": "X ", "nyi": "~ "}.get(level, "  ")
        self.view.appendPlainText("%s %s%s" % (stamp, prefix, text))
        self.view.verticalScrollBar().setValue(self.view.verticalScrollBar().maximum())

    def nyi(self, name):
        self.log("[%s] not available in star_gui viewer (STAR-CCM+ only / not yet mapped)." % name,
                 "nyi")

    def clear(self):
        self.view.clear()


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
    """状态栏：坐标 / 网格规模 / 单位 / 选择模式。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 0, 2, 0)
        self.coord = QLabel("(x,y,z): -")
        self.mesh = QLabel("")
        self.stats = QLabel("")
        self.unit = QLabel("")
        self.mode = QLabel("选择: 对象")
        for lbl in (self.coord, self.mesh, self.stats, self.unit, self.mode):
            lbl.setStyleSheet("padding-left: 10px;")
            lay.addWidget(lbl)
        lay.addStretch(1)

    def set_coord(self, xyz):
        self.coord.setText("(x,y,z): (%s)" % ", ".join("%.4g" % v for v in xyz))

    def set_mesh(self, nverts, nfaces):
        self.mesh.setText("顶点 %s · 面 %s" % (nverts, nfaces))

    def set_stats(self, text):
        self.stats.setText(text)

    def set_unit(self, text):
        self.unit.setText("单位: %s" % text)

    def set_mode(self, text):
        self.mode.setText("选择: %s" % text)


class Star3DViewport(QWidget):
    """3D 视口：QVTK + 方向指示器 + 六向/等轴测 + 拾取（对齐 cab_vtk / STAR-CCM+）。"""

    picked = pyqtSignal(object, object)   # (info_tuple, xyz)
    context_command = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
        from vtkmodules.vtkRenderingCore import vtkRenderer
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        lay.addWidget(self.vtk_widget)
        self.renderer = vtkRenderer()
        from star_gui_vtk import apply_starccm_background
        apply_starccm_background(self.renderer)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.actors = []          # [(key, name, part_id, vtkActor)]
        self._by_actor = {}
        self._picker = None
        self._orient = None
        self._watermark = None
        self._rep_mode = "solid"   # solid | wireframe | edges
        self._opacity = None      # None = 保留 actor 自带透明度（场景 TransparencyOverride）
        self._base_opacity = {}
        self._iren_ready = False
        self._shutting_down = False
        self._show_tries = 0
        self._style = None
        self._init_watermark()
        self.setFocusPolicy(Qt.StrongFocus)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context)
        try:
            self.vtk_widget.setContextMenuPolicy(Qt.CustomContextMenu)
            self.vtk_widget.customContextMenuRequested.connect(self._on_context)
        except Exception:
            pass

    def _vtk_window_ready(self):
        """QVTK 已映射出原生 HWND 后才可 Initialize / Render（否则 Win32 wglMakeCurrent 报 handle invalid）。"""
        if self._shutting_down or self.vtk_widget is None:
            return False
        try:
            if not self.isVisible() or not self.vtk_widget.isVisible():
                return False
            return int(self.vtk_widget.winId()) != 0
        except Exception:
            return False

    def _ensure_gl(self):
        if self._iren_ready or self._shutting_down:
            return self._iren_ready
        if not self._vtk_window_ready():
            return False
        try:
            self.vtk_widget.Initialize()
        except Exception:
            return False
        self._init_interactor_style()
        self._init_depth_peel()
        self._init_picker()
        self._init_orientation_marker()
        self._iren_ready = True
        return True

    def _init_interactor_style(self):
        """STAR-CCM+ 键位：左旋 / 中键缩放 / 右平移，降低旋转与滚轮灵敏度。"""
        from star_gui_interactor import install_starccm_interactor
        try:
            iren = self.vtk_widget.GetRenderWindow().GetInteractor()
            self._style = install_starccm_interactor(iren)
        except Exception:
            self._style = None

    def _safe_render(self):
        if self._shutting_down or not self._ensure_gl():
            return
        try:
            self.vtk_widget.GetRenderWindow().Render()
        except Exception:
            pass

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._on_shown)

    def _on_shown(self):
        if self._shutting_down:
            return
        if not self._vtk_window_ready():
            self._show_tries += 1
            if self._show_tries < 40:
                QTimer.singleShot(50, self._on_shown)
            return
        self._ensure_gl()
        self._safe_render()

    def _init_watermark(self):
        """3D 区顶部淡水印，对齐 STAR-CCM+ 图形窗口。"""
        try:
            import vtk
            t = vtk.vtkTextActor()
            t.SetInput("Simcenter STAR-CCM+")
            prop = t.GetTextProperty()
            prop.SetFontFamilyToArial()
            prop.SetFontSize(16)
            prop.SetBold(0)
            prop.SetColor(0.72, 0.72, 0.74)
            prop.SetOpacity(0.45)
            prop.SetJustificationToCentered()
            t.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            t.SetPosition(0.50, 0.93)
            self.renderer.AddActor2D(t)
            self._watermark = t
        except Exception:
            self._watermark = None

    def _init_depth_peel(self):
        """半透明外框需要 depth peeling，否则内外面排序会花。"""
        try:
            rw = self.vtk_widget.GetRenderWindow()
            rw.SetAlphaBitPlanes(1)
            rw.SetMultiSamples(0)
            self.renderer.SetUseDepthPeeling(1)
            self.renderer.SetMaximumNumberOfPeels(8)
            self.renderer.SetOcclusionRatio(0.1)
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
        self._base_opacity = {
            id(a): a.GetProperty().GetOpacity() for k, n, pid, a in self.actors
        }
        for k, n, pid, actor in self.actors:
            self.renderer.AddActor(actor)
        self._apply_rep()
        if self._vtk_window_ready():
            self.fit_view()

    def fit_view(self):
        try:
            self.renderer.ResetCamera()
        except Exception:
            pass
        self._safe_render()

    def reset_view(self):
        from star_gui_vtk import apply_view_preset
        apply_view_preset(self.renderer, "iso")
        self._safe_render()

    def set_view(self, name):
        from star_gui_vtk import apply_view_preset
        if apply_view_preset(self.renderer, name):
            self._safe_render()
            return True
        return False

    def render(self):
        self._safe_render()

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
                base = self._base_opacity.get(id(actor), 1.0)
                op = base if self._opacity is None else self._opacity
                actor.GetProperty().SetOpacity(op)

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
        if self._opacity is None or self._opacity > 0.7:
            self.set_opacity(0.4)
        else:
            self._opacity = None
            self._apply_rep()
            self.render()

    def set_background(self, rgb, rgb2=None):
        self.renderer.SetBackground(*rgb)
        if rgb2:
            self.renderer.SetBackground2(*rgb2)
            try:
                self.renderer.GradientBackgroundOn()
            except Exception:
                pass
        self.render()

    def _on_context(self, pos):
        menu = QMenu(self)
        def add(key, label):
            act = menu.addAction(label)
            act.triggered.connect(lambda checked=False, k=key: self.context_command.emit(k))
        add("fit", "适配视图")
        add("reset", "重置视图")
        menu.addSeparator()
        for key, label in (("+x", "+X"), ("-x", "-X"), ("+y", "+Y"), ("-y", "-Y"),
                           ("+z", "+Z"), ("-z", "-Z"), ("iso", "等轴测")):
            add("view:" + key, label)
        menu.addSeparator()
        add("solid", "实体")
        add("wire", "线框")
        add("edges", "仅边线")
        add("transp", "透明")
        add("mesh_on", "网格开")
        add("mesh_off", "网格关")
        menu.addSeparator()
        add("align_normal", "对齐到零件法向")
        add("derived", "创建派生零件")
        add("distance", "测距")
        add("rubber", "框选缩放")
        add("copy_image", "复制图像")
        try:
            w = self.vtk_widget if self.vtk_widget is not None else self
            menu.exec_(w.mapToGlobal(pos))
        except Exception:
            menu.exec_(self.mapToGlobal(pos))

    def enable_rubber_zoom(self, on=True):
        """临时换成 VTK 橡胶带缩放；适配视图后可回到 STAR 键位。"""
        if not on or self.vtk_widget is None:
            return False
        try:
            from vtkmodules.vtkInteractionStyle import vtkInteractorStyleRubberBandZoom
            iren = self.vtk_widget.GetRenderWindow().GetInteractor()
            style = vtkInteractorStyleRubberBandZoom()
            iren.SetInteractorStyle(style)
            self._style = style
            return True
        except Exception:
            return False

    def copy_image_to_clipboard(self):
        try:
            from PyQt5.QtWidgets import QApplication
            pix = self.grab()
            QApplication.clipboard().setPixmap(pix)
            return True
        except Exception:
            return False

    def keyPressEvent(self, event):
        key = event.key()
        shift = bool(event.modifiers() & Qt.ShiftModifier)
        if key == Qt.Key_F:
            self.fit_view()
            return
        if key == Qt.Key_R:
            self.reset_view()
            return
        mapping = {Qt.Key_X: "x", Qt.Key_Y: "y", Qt.Key_Z: "z"}
        axis = mapping.get(key)
        if axis:
            name = ("-" if shift else "+") + axis
            self.set_view(name)
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)

    def shutdown(self):
        """在销毁 HWND 前释放 OpenGL，避免 wglMakeCurrent(handle invalid)。"""
        if self._shutting_down:
            return
        self._shutting_down = True
        self._iren_ready = False
        try:
            if self._orient is not None:
                self._orient.SetEnabled(0)
                try:
                    self._orient.SetInteractor(None)
                except Exception:
                    pass
                self._orient = None
        except Exception:
            pass
        try:
            from vtkmodules.vtkCommonCore import vtkObject
            vtkObject.GlobalWarningDisplayOff()
        except Exception:
            vtkObject = None
        try:
            rw = self.vtk_widget.GetRenderWindow()
            iren = rw.GetInteractor() if rw is not None else None
            if iren is not None:
                try:
                    iren.EnableRenderOff()
                except Exception:
                    pass
            if rw is not None:
                try:
                    rw.Finalize()
                except Exception:
                    pass
        except Exception:
            pass
        if vtkObject is not None:
            try:
                vtkObject.GlobalWarningDisplayOn()
            except Exception:
                pass


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
    context_command = pyqtSignal(str, object)  # (action_key, SimObject|None)

    def __init__(self, model=None, icons=None, parent=None):
        super().__init__(parent)
        self.model = model
        self._icons = icons
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["模型 / 场景/绘图"])
        self.tree.setHeaderHidden(True)
        self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_context)
        self.tree.itemSelectionChanged.connect(self._on_selection)
        self.tree.itemChanged.connect(self._on_item_changed)
        self._filling = False
        lay.addWidget(self.tree)

    def set_model(self, model):
        self.model = model
        self.rebuild()

    def rebuild(self):
        self._filling = True
        self.tree.blockSignals(True)
        try:
            self.tree.clear()
            if self.model is None:
                return
            from star_gui_model import Node  # noqa: F401
            roots = self.model.sim_tree() if hasattr(self.model, "sim_tree") else self.model.tree_roots()
            for root in roots:
                item = self._make_item(root)
                self.tree.addTopLevelItem(item)
            self.tree.expandToDepth(3)
        finally:
            self.tree.blockSignals(False)
            self._filling = False

    def set_check(self, oid, checked):
        """同步复选框且不发 check_changed（撤销/命令回写）。"""
        from PyQt5.QtCore import Qt
        self._filling = True
        self.tree.blockSignals(True)
        try:
            def walk(item):
                if item.data(0, 32) == oid:
                    item.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
                    return True
                for i in range(item.childCount()):
                    if walk(item.child(i)):
                        return True
                return False
            for i in range(self.tree.topLevelItemCount()):
                if walk(self.tree.topLevelItem(i)):
                    break
        finally:
            self.tree.blockSignals(False)
            self._filling = False

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
        if getattr(self, "_filling", False):
            return
        if column != 0 or not (item.flags() & 2):   # ItemIsUserCheckable = 2
            return
        from PyQt5.QtCore import Qt
        oid = item.data(0, 32)
        if oid is not None:
            self.check_changed.emit(oid, item.checkState(0) == Qt.Checked)

    def _icon_key(self, node):
        """按 STAR-CCM+ 树节点类型选图标（几何立方体 / 区域金块 / 场景相机等）。"""
        label = node.label or ""
        cn = node.class_name or ""
        short = cn.split(".")[-1]
        folders = {
            "Geometry": "layer_geometry",
            "Continua": "layer_physics",
            "Regions": "folder_regions",
            "Solvers": "solver",
            "Reports": "report",
            "Plots": "plot",
            "Monitors": "monitor",
            "Scenes": "scene",
            "Tools": "tools",
            "Boundaries": "boundary",
            "Parts": "part",
            "Field Functions": "field",
            "Coordinate Systems": "coord",
            "Units": "units",
            "Tables": "table",
            "Operations": "layer_meshing",
            "Derived Parts": "part",
            "3D-CAD": "layer_geometry",
        }
        if label in folders:
            return folders[label]
        if short == "folder":
            return "folder"
        if short == "Region":
            return "region"
        if short == "Boundary":
            return "boundary"
        if short == "Scene":
            return "scene"
        if short == "PhysicsContinuum":
            return "continuum"
        if "Displayer" in short:
            return "displayer"
        if short == "PartGroup" or label == "Parts":
            return "part"
        if short == "Boundary":
            return "boundary"
        if short.endswith("Solver") or "Solver" in short:
            return "solver"
        if "Plot" in short:
            return "plot"
        if "Monitor" in short:
            return "monitor"
        if "CoordinateSystem" in short:
            return "coord"
        if short.endswith("Function") or "Function" in short:
            return "field"
        if "PartSurface" in short:
            return "surface"
        if "Part" in short or short.endswith("Part"):
            return "part"
        if short == "Simulation":
            return "simulation"
        if short.endswith("Model") or "Model" in short:
            return "model"
        if short == "UnitsManager":
            return "units"
        if "Table" in short:
            return "table"
        layer_map = {
            "cad-geometry": "layer_geometry",
            "meshing": "layer_meshing",
            "physics": "layer_physics",
            "visualization": "layer_visualization",
            "post-processing": "layer_post",
            "solver": "solver",
            "query": "tools",
            "core": "simulation",
        }
        return layer_map.get(node.layer, "unknown")

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

    def current_object(self):
        items = self.tree.selectedItems()
        if not items or self.model is None:
            return None
        oid = items[0].data(0, 32)
        return self.model.object_by_id(oid) if oid is not None else None

    def _on_context(self, pos):
        item = self.tree.itemAt(pos)
        if item is not None:
            self.tree.setCurrentItem(item)
        obj = self.current_object()
        menu = QMenu(self)
        def add(key, label):
            act = menu.addAction(label)
            act.triggered.connect(lambda checked=False, k=key: self.context_command.emit(k, obj))
            return act
        add("edit", "编辑...")
        add("rename", "重命名")
        add("copy", "复制")
        add("paste", "粘贴")
        add("delete", "删除")
        menu.addSeparator()
        add("hide", "隐藏")
        add("show", "显示")
        add("show_only", "仅显示此项")
        add("highlight", "在场景中高亮")
        cn = (obj.class_name if obj is not None else "") or ""
        if "Part" in cn or cn.endswith("Region") or "Boundary" in cn:
            menu.addSeparator()
            add("transform", "变换...")
            add("assign_region", "指定到区域...")
        if "Scene" in cn or "Displayer" in cn:
            menu.addSeparator()
            add("add_displayer", "添加到显示器")
            add("representation", "Representation")
            add("parts_filter", "编辑 Parts 过滤器")
        if "MeshOperation" in cn or (item is not None and item.text(0) == "Operations"):
            menu.addSeparator()
            add("execute_mesh", "执行网格操作")
            add("new_automesh", "新建自动网格操作")
        if "CadModel" in cn or (item is not None and item.text(0) == "3D-CAD"):
            menu.addSeparator()
            add("cad_mode", "进入 3D-CAD")
        menu.exec_(self.tree.viewport().mapToGlobal(pos))


_PROP_RO = {"ClassName", "Parent", "Simulation", "NameManager", "TimeStamp",
            "clientType", "Index"}


def _is_g7_edit(raw):
    """P1：G7 编辑描述符判定（kind/oid/key 齐备的 dict）。"""
    return (isinstance(raw, dict) and "kind" in raw
            and isinstance(raw.get("oid"), int) and "key" in raw)


def _g7_parse_edit(raw, text):
    """P1：按编辑描述符把单元格文本解析回目标值；无法解析返回 None。

    quantity：去单位后缀后按 float / 矢量列表解析（以当前值类型为准）；
    scalar：按当前值类型（bool/int/float）解析；
    option：接受选项名（options 全为字符串时）或枚举序号。
    """
    s = str(text).strip()
    kind = raw.get("kind")
    if kind == "quantity":
        units = raw.get("units") or ""
        if units and s.endswith(units):
            s = s[:-len(units)].strip()
        cur = raw.get("value")
        if isinstance(cur, list):
            s = s.strip("()[]")
            parts = [p for p in (x.strip() for x in s.split(",")) if p]
            if len(parts) != len(cur):
                return None
            try:
                return [type(cur[i])(float(p)) for i, p in enumerate(parts)]
            except (TypeError, ValueError):
                return None
        try:
            return float(s)
        except (TypeError, ValueError):
            return None
    if kind == "scalar":
        cur = raw.get("value")
        try:
            if isinstance(cur, bool):
                return s.lower() in ("1", "true", "yes", "on")
            if isinstance(cur, int):
                return int(float(s))
            return float(s)
        except (TypeError, ValueError):
            return None
    if kind == "option":
        opts = raw.get("options") or []
        if opts and all(isinstance(x, str) for x in opts) and s in opts:
            return opts.index(s)
        try:
            return int(float(s))
        except (TypeError, ValueError):
            return None
    return None


def parse_property_text(raw, text):
    """把单元格文本解析回原类型；无法解析返回 None。"""
    if _is_g7_edit(raw):
        return _g7_parse_edit(raw, text)
    if raw is None:
        return text
    if isinstance(raw, bool):
        return str(text).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(raw, int) and not isinstance(raw, bool):
        try:
            return int(float(text))
        except (TypeError, ValueError):
            return None
    if isinstance(raw, float):
        try:
            return float(text)
        except (TypeError, ValueError):
            return None
    if isinstance(raw, (list, tuple)) and raw and all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in raw):
        parts = [p.strip() for p in str(text).replace("[", "").replace("]", "").split(",")
                 if p.strip()]
        if len(parts) != len(raw):
            return None
        out = []
        for i, p in enumerate(parts):
            try:
                out.append(type(raw[i])(float(p)))
            except (TypeError, ValueError):
                return None
        return out
    if isinstance(raw, str):
        return text
    return None


class PropertiesPanel(QWidget):
    """属性检查器：两列（属性|值），可编辑标量/颜色/矢量。"""

    reference_activated = pyqtSignal(object)
    title_changed = pyqtSignal(str)
    property_edited = pyqtSignal(object, str, object)  # obj, key, new_value

    def __init__(self, model=None, parent=None):
        super().__init__(parent)
        self.model = model
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        bar = QHBoxLayout()
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("筛选属性…")
        self.filter.textChanged.connect(self._apply_filter)
        bar.addWidget(self.filter, 1)
        self.btn_reset = QPushButton("重置")
        self.btn_reset.setFixedWidth(48)
        self.btn_reset.clicked.connect(self._reset_selected)
        bar.addWidget(self.btn_reset)
        self.btn_color = QPushButton("颜色")
        self.btn_color.setFixedWidth(48)
        self.btn_color.clicked.connect(self._pick_color)
        bar.addWidget(self.btn_color)
        lay.addLayout(bar)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["属性", "值"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.setColumnWidth(0, 160)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked |
                                   QAbstractItemView.EditKeyPressed)
        self.table.setSelectionBehavior(self.table.SelectRows)
        self.table.cellDoubleClicked.connect(self._on_double)
        self.table.itemChanged.connect(self._on_item_changed)
        lay.addWidget(self.table)
        self.hint = QLabel("")
        self.hint.setObjectName("propHint")
        self.hint.setStyleSheet("padding: 2px 6px; font-size: 11px;")
        lay.addWidget(self.hint)
        self._current = None
        self._rows = []
        self._filling = False
        self._originals = {}

    def set_model(self, model):
        self.model = model

    def show_object(self, obj):
        self._current = obj
        self.table.setRowCount(0)
        self._rows = []
        if obj is None:
            self.hint.setText("")
            self.title_changed.emit("属性")
            return
        from star_gui_model import friendly_name, short_class
        from semantic_dict import resolve_class
        rows = self.model.properties(obj) if self.model else []
        alias = resolve_class(obj.class_name) if obj.class_name else obj.class_name
        self._rows = rows
        self._originals = {k: v for k, _t, v in rows}
        self._fill_table(rows)
        title = "%s - 属性" % (obj.name or friendly_name(obj))
        self.title_changed.emit(title)
        extra = short_class(obj.class_name)
        if alias and alias != obj.class_name:
            extra += "  (= %s)" % short_class(alias)
        self.hint.setText("id %d · %s" % (obj.id, extra))

    def _fill_table(self, rows):
        self._filling = True
        self.table.setRowCount(0)
        for attr, val, raw in rows:
            row = self.table.rowCount()
            self.table.insertRow(row)
            aitem = QTableWidgetItem(attr)
            aitem.setFlags(aitem.flags() & ~Qt.ItemIsEditable)
            aitem.setToolTip(type(raw).__name__)
            self.table.setItem(row, 0, aitem)
            vitem = QTableWidgetItem(val)
            vitem.setData(34, raw)
            vitem.setToolTip(type(raw).__name__)
            if attr.startswith("G7:") or attr in _PROP_RO or isinstance(raw, dict) or (
                    isinstance(raw, int) and self.model is not None
                    and raw in self.model.objmap and attr not in (
                        "Opacity", "TriangleCount", "VertexCount")):
                vitem.setFlags(vitem.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 1, vitem)
        self._filling = False

    def _apply_filter(self, text):
        needle = (text or "").lower()
        if not needle:
            self._fill_table(self._rows)
            return
        filtered = [r for r in self._rows if needle in r[0].lower() or needle in str(r[1]).lower()]
        self._fill_table(filtered)

    def _on_double(self, row, col):
        if self.model is None:
            return
        item = self.table.item(row, 1)
        raw = item.data(34) if item else None
        if isinstance(raw, int) and raw in self.model.objmap:
            self.reference_activated.emit(self.model.objmap[raw])

    def _on_item_changed(self, item):
        if self._filling or self._current is None or item.column() != 1:
            return
        row = item.row()
        key_item = self.table.item(row, 0)
        if key_item is None:
            return
        key = key_item.text()
        raw = item.data(34)
        parsed = parse_property_text(raw, item.text())
        if parsed is None:
            self._filling = True
            item.setText(str(self._originals.get(key, raw)))
            self._filling = False
            return
        if _is_g7_edit(raw):
            target = self.model.objmap.get(raw["oid"]) if self.model else None
            if target is None:
                return
            self.property_edited.emit(target, raw["key"], parsed)
            return
        item.setData(34, parsed)
        self.property_edited.emit(self._current, key, parsed)

    def _reset_selected(self):
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows or self._current is None:
            return
        row = rows[0].row()
        key_item = self.table.item(row, 0)
        if key_item is None:
            return
        key = key_item.text()
        if key not in self._originals:
            return
        self.property_edited.emit(self._current, key, self._originals[key])
        self.show_object(self._current)

    def _pick_color(self):
        if self._current is None:
            return
        key = None
        raw = None
        for k, _t, v in self._rows:
            if "Color" in k and isinstance(v, (list, tuple)) and len(v) >= 3:
                key, raw = k, v
                break
        if key is None:
            return
        from PyQt5.QtGui import QColor
        c = QColorDialog.getColor(QColor(int(raw[0] * 255), int(raw[1] * 255),
                                         int(raw[2] * 255)), self, "选择颜色")
        if not c.isValid():
            return
        rgb = [c.red() / 255.0, c.green() / 255.0, c.blue() / 255.0]
        if len(raw) > 3:
            rgb.extend(list(raw[3:]))
        self.property_edited.emit(self._current, key, rgb)
        self.show_object(self._current)


class PartsFilterDialog(QDialog):
    """Parts 过滤器：勾选 MeshPart / CadPart / PartSurface，写回 Collector.Keys。"""

    def __init__(self, parts, selected_ids, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Parts 过滤器")
        self.resize(360, 420)
        lay = QVBoxLayout(self)
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.NoSelection)
        selected = set(selected_ids or [])
        for oid, label in parts or []:
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if oid in selected else Qt.Unchecked)
            item.setData(Qt.UserRole, oid)
            self.list.addItem(item)
        lay.addWidget(self.list, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def selected_ids(self):
        out = []
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.checkState() == Qt.Checked:
                oid = item.data(Qt.UserRole)
                if oid is not None:
                    out.append(int(oid))
        return out

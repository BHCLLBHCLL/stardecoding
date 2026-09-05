# -*- coding: utf-8 -*-
"""star_gui.py — STAR-CCM+ .sim 项目查看器 / 编辑器。

布局对齐 STAR-CCM+ 20.02（模型树 / 图形窗口 / 属性 / 输出 / 状态栏），
技术路线对齐 cabdecoding（PyQt5 + VTK + numpy）。
求解运算不实现（菜单保留并禁用）。见 star_gui_parity.md。
"""

import os
import sys
import threading

HEADLESS = os.environ.get("QT_QPA_PLATFORM", "").lower() in ("minimal", "offscreen")
"""无头模式：minimal/offscreen 下不创建 QVTK 视口（QVTK 在无头平台不稳定），
用占位控件代替；3D 渲染正确性由 star_gui_vtk.render_offscreen_png 的纯 VTK
离屏测试覆盖。"""

from PyQt5.QtCore import QObject, QSettings, QSize, QThread, Qt, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QAction, QApplication, QFileDialog, QFrame, QInputDialog, QMainWindow,
    QMessageBox, QSplitter, QTabWidget, QToolBar, QVBoxLayout, QWidget,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from star_gui_icons import AppIcons, icons
from star_gui_panes import (
    MessageWindow, PaneFrame, ProgressPanel, StatusBarHelper, SummaryPane,
)


class LoadWorker(QObject):
    """后台加载 .sim（大文件避免卡 UI）。"""

    finished = pyqtSignal(object, str)   # (SimFile, summary_text)
    failed = pyqtSignal(str, str)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        from sim_parser import SimFile
        try:
            sim = SimFile(self.path)
            lines = [sim.summary()]
            lines.append("")
            fp = sim.version_fingerprint()
            lines.append("版本指纹: banner=%s release=%s 编码=%s 头部=%s" % (
                fp["banner_version"], fp["release"], fp["state_mode"],
                ",".join(fp["header_keys"] or [])))
            chk = sim.check_state_length()
            lines.append("长度自校验: %s (%s)" % ("通过" if chk["ok"] else "失败",
                                                chk["detail"]))
            census, _named = sim.layer_census()
            top = ", ".join("%s=%d" % (k, n) for k, n in census.most_common(8))
            lines.append("语义层: %s" % top)
            if sim.nested_transmits:
                lines.append("嵌套 TRANSMIT 子块: %d" % len(sim.nested_transmits))
            if sim.state_segments:
                lines.append("状态表分段: %d 段" % len(sim.state_segments))
            self.finished.emit(sim, "\n".join(lines))
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.failed.emit(self.path, "%s\n%s" % (exc, traceback.format_exc()))


class SolverController(QObject):
    """P10 求解运行闭环控制器（GUI 侧）。

    把 solver_run.SolverBackend 的 run_loop 驱动在后台线程（python threading），
    GUI 线程通过 pause/step/stop 写入控制标志；运行时事件经 UpdateEvents 监听转成
    Qt 信号（AutoConnection 排队到 GUI 线程），驱动状态按钮、状态栏与残差实时曲线。
    后端用 DemoDiffusionSolver（纯 numpy）读取真实残差/监视器，不依赖 CCM 许可。
    """

    state_changed = pyqtSignal(str)      # RUNNING/PAUSED/STOPPED/COMPLETED/ERROR
    iterated = pyqtSignal(int, dict)     # (iteration, payload)
    finished = pyqtSignal(dict)          # run_loop 返回的 result
    failed = pyqtSignal(str)             # 异常消息

    def __init__(self, parent=None):
        super().__init__(parent)
        from solver_run import DemoDiffusionSolver, SolverBackend, demo_mesh
        self.backend = SolverBackend(solver=DemoDiffusionSolver(*demo_mesh(nx=6)))
        self._thread = None
        self._max_iter = 1_000_000_000
        self._wire_events()

    def _wire_events(self):
        from solver_run import SolverState
        ev = self.backend.events
        ev.register("start", lambda **k: self.state_changed.emit(SolverState.RUNNING))
        ev.register("pause", lambda **k: self.state_changed.emit(SolverState.PAUSED))
        ev.register("resume", lambda **k: self.state_changed.emit(SolverState.RUNNING))
        ev.register("stop", lambda **k: self.state_changed.emit(SolverState.STOPPED))
        ev.register("finish", lambda **k: self.state_changed.emit(SolverState.COMPLETED))
        ev.register("iterate", self._on_iterate)
        ev.register("step", self._on_iterate)

    def _on_iterate(self, iteration, payload, state="working", **kw):
        self.iterated.emit(int(iteration), dict(payload))

    def begin(self):
        """启动/恢复 run_loop：线程未起则起新线程，已挂起则 resume。"""
        from solver_run import SolverState
        if self._thread is not None and self._thread.is_alive():
            if self.backend.state() == SolverState.PAUSED:
                self.backend.resume()
            return
        self._thread = threading.Thread(target=self._work, daemon=True)
        self._thread.start()

    def _work(self):
        try:
            result = self.backend.run_loop(max_iter=self._max_iter)
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))

    def pause(self):
        self.backend.pause()

    def step(self):
        if self.backend.step():
            self.state_changed.emit(self.backend.state())

    def stop(self):
        self.backend.stop()

    def metrics(self):
        return self.backend.metrics()

    def curve_items(self):
        return self.backend.curve_items()

    def report_lines(self):
        return self.backend.report_lines()

    def thread_alive(self):
        return self._thread is not None and self._thread.is_alive()


class StarMainWindow(QMainWindow):
    """主窗口（M0：骨架 + 打开 + 摘要；M1-M6 逐步补齐窗格）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        from star_gui_i18n import tr
        self.tr = tr
        self.setWindowTitle("STAR-CCM+ .sim Viewer / Editor")
        self.resize(1280, 820)
        self.sim = None
        self.sim_path = None
        self._thread = None
        self._worker = None
        self.solver_controller = None
        from star_gui_document import SimDocument
        self.document = SimDocument()
        self.document.bus.on_change = self._sync_edit_actions
        self.document.add_listener(self._on_document_event)
        self._build_actions()
        self._build_ui()
        self._build_menus()
        self._build_toolbar()
        self._wire_editor_ui()
        self._setup_solver_controller()
        self._sync_edit_actions()
        self.msg("STAR-CCM+ .sim Viewer / Editor 就绪 — 文件>打开 加载项目")

    def _setup_solver_controller(self):
        """P10：创建求解运行控制器并接线（状态/迭代/完成/失败 → GUI）。"""
        from solver_run import SolverState
        self.solver_controller = SolverController(self)
        self.solver_controller.state_changed.connect(self._on_solver_state)
        self.solver_controller.iterated.connect(self._on_solver_iterated)
        self.solver_controller.finished.connect(self._on_solver_finished)
        self.solver_controller.failed.connect(self._on_solver_failed)
        self._sync_solution_actions(SolverState.IDLE)

    # ---------------- UI ----------------
    def _build_ui(self):
        """2×2 块状布局：左(树/属性) | 右(3D 场景/输出)，对齐 STAR-CCM+ 停靠窗。"""
        central = QWidget()
        central.setObjectName("workspace")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(0)

        self.split_main = QSplitter(Qt.Horizontal)
        self.split_main.setObjectName("splitMain")
        self.split_main.setChildrenCollapsible(False)
        self.split_main.setHandleWidth(4)

        self.split_left = QSplitter(Qt.Vertical)
        self.split_left.setObjectName("splitLeft")
        self.split_left.setChildrenCollapsible(False)
        self.split_left.setHandleWidth(4)

        from star_gui_panes import PropertiesPanel, SimulationTree
        self.model = None
        self.tree_widget = SimulationTree(icons=icons())
        self.props_widget = PropertiesPanel()
        self.tree_widget.object_selected.connect(self.on_object_selected)
        self.tree_widget.check_changed.connect(self.on_tree_check)
        self.props_widget.reference_activated.connect(self.on_object_selected)
        self.tree_widget.context_command.connect(self.on_tree_context)
        self.props_widget.property_edited.connect(self.on_property_edited)
        self.tree_pane = PaneFrame("模型 / 场景/绘图")
        self.tree_pane.set_body(self.tree_widget)
        self.props_pane = PaneFrame("属性")
        self.props_pane.set_body(self.props_widget)
        self.props_widget.title_changed.connect(self.props_pane.set_title)
        self.split_left.addWidget(self.tree_pane)
        self.split_left.addWidget(self.props_pane)
        self.split_left.setStretchFactor(0, 3)
        self.split_left.setStretchFactor(1, 2)
        self.split_left.setSizes([480, 280])

        self.split_right = QSplitter(Qt.Vertical)
        self.split_right.setObjectName("splitRight")
        self.split_right.setChildrenCollapsible(False)
        self.split_right.setHandleWidth(4)

        self.summary_pane = SummaryPane()
        from star_gui_panes import GraphicsTabs, Star3DViewport
        self.graphics_tabs = GraphicsTabs()
        self.graphics_tabs.add_info_tab(self.summary_pane)
        self.graphics_tabs.currentChanged.connect(self._on_graphics_tab_changed)
        if HEADLESS:
            from PyQt5.QtWidgets import QLabel
            self.viewport = QLabel("3D 视口（无头模式禁用 QVTK）")
            self.viewport.setAlignment(Qt.AlignCenter)
            self.graphics_tabs.add_mesh_tab("3D Mesh", self.viewport)
        else:
            self.viewport = Star3DViewport()
            self.graphics_tabs.add_mesh_tab("3D Mesh", self.viewport)
        self.graphics_block = QFrame()
        self.graphics_block.setObjectName("BlockFrame")
        gbl = QVBoxLayout(self.graphics_block)
        gbl.setContentsMargins(0, 0, 0, 0)
        gbl.setSpacing(0)
        gbl.addWidget(self.graphics_tabs)
        self.graphics_pane = self.graphics_block
        self.split_right.addWidget(self.graphics_block)

        bottom = QTabWidget()
        self.bottom_tabs = bottom
        self.messages = MessageWindow()
        self.progress = ProgressPanel()
        from star_gui_i18n import tr
        from star_gui_plots import PlotPane
        self.plot_pane = PlotPane()
        bottom.addTab(self.messages, tr("Messages"))
        bottom.addTab(self.progress, tr("Progress"))
        bottom.addTab(self.plot_pane, tr("Plots Window"))
        self.output_block = QFrame()
        self.output_block.setObjectName("BlockFrame")
        obl = QVBoxLayout(self.output_block)
        obl.setContentsMargins(0, 0, 0, 0)
        obl.setSpacing(0)
        obl.addWidget(bottom)
        self.split_right.addWidget(self.output_block)
        self.split_right.setStretchFactor(0, 5)
        self.split_right.setStretchFactor(1, 1)
        self.split_right.setSizes([640, 150])

        self.split_main.addWidget(self.split_left)
        self.split_main.addWidget(self.split_right)
        self.split_main.setStretchFactor(0, 0)
        self.split_main.setStretchFactor(1, 1)
        self.split_main.setSizes([300, 980])
        root.addWidget(self.split_main)

        self.status_helper = StatusBarHelper()
        self.statusBar().addWidget(self.status_helper, 1)
        self.set_status("就绪")

    def _build_actions(self):
        from star_gui_i18n import tr
        self.actions = {}
        self._add("File>New", tr("New"), self.cmd_new, "file")
        self._add("File>Open", tr("Open..."), self.open_file, "open", QKeySequence.Open)
        self._add("File>Close", tr("Close"), self.close_sim, "close")
        self._add("File>Reload", tr("Reload"), self.cmd_reload, "open")
        self._add("File>Exit", tr("Exit"), self.close, "close", QKeySequence.Quit)
        self._add("File>Export>STL", tr("Export STL..."), self.cmd_export_stl, "mesh")
        self._add("File>Export>Summary", tr("Export Summary..."), self.cmd_export_summary, "file")
        self._add("File>Export>Report", tr("Export Report (JSON)..."), self.cmd_export_report, "report")
        self._add("File>Save", tr("Save"), self.cmd_save, "save", QKeySequence.Save)
        self._add("File>Save As", tr("Save As..."), self.cmd_save_as, "save")
        self._add("File>Save All", tr("Save All"), lambda: self._nyi("File>Save All"), "save")
        self._add("File>Import>Surface", tr("Import Surface..."), self.cmd_import_surface, "mesh")
        self._add("File>Import>CAD", tr("Import CAD..."), self.cmd_import_cad, "mesh")
        self._add("File>Import>Volume", tr("Import Volume Mesh..."),
                  self.cmd_import_volume, "mesh")
        self._add("Edit>Undo", tr("Undo"), self.cmd_undo, "undo", QKeySequence.Undo)
        self._add("Edit>Redo", tr("Redo"), self.cmd_redo, "undo", QKeySequence.Redo)
        self._add("Edit>Copy", tr("Copy"), self.cmd_copy, "file")
        self._add("Edit>Paste", tr("Paste"), self.cmd_paste, "file")
        self._add("Edit>Delete", tr("Delete"), self.cmd_delete, "close")
        self._add("Edit>Rename", tr("Rename"), self.cmd_rename, "file")
        self._add("Edit>Search", tr("Search Names"), self.cmd_search, "field")
        self._add("Edit>Prev", tr("Previous Selection"), self.cmd_sel_prev, "file")
        self._add("Edit>Next", tr("Next Selection"), self.cmd_sel_next, "file")
        self._add("Mesh>Generate", tr("Generate Surface Mesh"), self.cmd_generate_mesh, "mesh")
        self._add("Mesh>GenerateVolume", tr("Generate Volume Mesh"),
                  self.cmd_generate_volume_mesh, "mesh")
        self._add("Mesh>GeneratePoly", tr("Generate Polyhedral Mesh"),
                  self.cmd_generate_poly_mesh, "mesh")
        self._add("Mesh>GenerateTrimmer", tr("Generate Trimmer Mesh"),
                  self.cmd_generate_trimmer_mesh, "mesh")
        self._add("Mesh>Clear", tr("Clear Generated Meshes"),
                  lambda: self._kernel_nyi("清除已生成网格"), "mesh")
        self._add("Mesh>Scale", tr("Scale Mesh..."), self.cmd_scale_mesh, "mesh")
        self._add("Mesh>Diagnostics", tr("Mesh Diagnostics"), self.cmd_mesh_diag, "ruler")
        self._add("Mesh>Convert2D", tr("Convert to 2D"),
                  lambda: self._kernel_nyi("转换为 2D"), "mesh")
        self._add("Mesh>Repair", tr("Surface Repair"), self.cmd_cad_repair, "mesh")
        self._add("Plot>NYI", tr("Plot"), self.cmd_show_plots, "plot")
        self._add("Scene>New", tr("New Scene"), self.cmd_new_scene, "scene")
        self._add("Scene>AddDisplayer", tr("Add Displayer"), self.cmd_add_displayer, "displayer")
        self._add("Vis>SaveView", tr("Save View"), self.cmd_save_view, "reset")
        self._add("Vis>RestoreView", tr("Restore View"), self.cmd_restore_view, "fit")
        self._add("Solution>Run", tr("Run"), self.cmd_solution_run, "play")
        self._add("Solution>Pause", tr("Pause"), self.cmd_solution_pause, "pause")
        self._add("Solution>Step", tr("Step"), self.cmd_solution_step, "step")
        self._add("Solution>Stop", tr("Stop"), self.cmd_solution_stop, "stop")
        self.actions["Solution>Pause"].setEnabled(False)
        self.actions["Solution>Step"].setEnabled(False)
        self.actions["Solution>Stop"].setEnabled(False)
        self._add("Connection>Server", tr("Connect to Server"),
                  lambda checked=False: self._nyi("Connection>Server"), "server")
        self.actions["Connection>Server"].setEnabled(False)
        self._add("Tools>Fingerprint", tr("Version Fingerprint"), self.cmd_fingerprint, "fingerprint")
        self._add("Tools>Check Length", tr("State Length Check"), self.cmd_check_length, "ruler")
        self._add("Tools>Validate", tr("ClassVersions Validate"), self.cmd_validate, "validate")
        self._add("Tools>Options", tr("Options"), self.cmd_options, "properties")
        self._add("Vis>MeshOn", tr("Mesh On"), lambda: self.cmd_vis_mesh(True), "edges")
        self._add("Vis>MeshOff", tr("Mesh Off"), lambda: self.cmd_vis_mesh(False), "solid")
        self._add("Vis>Derived", tr("Create Derived Part"), self.cmd_create_derived, "part")
        self._add("Vis>Rubber", tr("Rubber Zoom"), self.cmd_rubber_zoom, "fit")
        self._add("Vis>Distance", tr("Measure Distance"), self.cmd_distance, "ruler")
        self._add("Vis>PartsFilter", tr("Edit Parts Filter"), self.cmd_edit_parts_filter, "part")
        self._add("Vis>Scalar", tr("Scalar Coloring"), self.cmd_scalar_color, "field")
        self._add("Window>Plots", tr("Plots Window"), self._toggle_plots, "plot")
        self._add("Window>Cad", tr("3D-CAD Mode"), self.cmd_toggle_cad, "layer_geometry")
        self._add("Cad>Section", tr("CAD Section"), self.cmd_cad_section, "part")
        self._add("Cad>Transform", tr("CAD Transform"), self.cmd_cad_transform, "mesh")
        self.actions["Window>Cad"].setCheckable(True)
        self.actions["Window>Plots"].setCheckable(True)
        self.actions["Window>Plots"].setChecked(True)
        self._add("Scene>Fit", tr("Fit View"), self.cmd_fit, "fit", "Ctrl+F")
        self._add("Scene>Reset", tr("Reset View"), self.cmd_reset, "reset")
        self._add("Scene>Solid", tr("Solid"), self.cmd_solid, "solid")
        self._add("Scene>Wireframe", tr("Wireframe"), self.cmd_toggle_wire, "wire")
        self._add("Scene>Edges", tr("Edges Only"), self.cmd_edges_only, "edges")
        self._add("Scene>Transparency", tr("Transparency"), self.cmd_transparency, "transp")
        for name, label in (("+x", "+X"), ("-x", "-X"), ("+y", "+Y"), ("-y", "-Y"),
                            ("+z", "+Z"), ("-z", "-Z"), ("iso", tr("Isometric"))):
            self._add("Scene>View>%s" % name, label,
                      lambda checked=False, n=name: self.cmd_view(n), "view_%s" % name)
        self._add("Help>About", tr("About"), self.cmd_about, "info")
        self._add("Window>Tree", tr("Simulation Tree"), self._toggle_tree, "tree")
        self._add("Window>Props", tr("Properties"), self._toggle_props, "properties")
        self._add("Window>Output", tr("Output"), self._toggle_output, "messages")
        for key in ("Window>Tree", "Window>Props", "Window>Output"):
            act = self.actions[key]
            act.blockSignals(True)
            act.setCheckable(True)
            act.setChecked(True)
            act.blockSignals(False)

    def _add(self, key, label, slot, icon_key, shortcut=None):
        act = QAction(icons().get(icon_key), label, self)
        if shortcut:
            act.setShortcut(shortcut)
        act.triggered.connect(slot)
        act.setObjectName(key)
        self.actions[key] = act
        return act

    def _build_menus(self):
        from star_gui_i18n import tr
        mbar = self.menuBar()
        def menu(title, keys):
            m = mbar.addMenu(title)
            for key in keys:
                if key is None:
                    m.addSeparator()
                    continue
                act = self.actions.get(key)
                if act:
                    m.addAction(act)
            return m

        file_menu = menu(tr("File") + "(&F)", [
            "File>New", "File>Open", "File>Reload", "File>Close", None,
            "File>Save", "File>Save As", "File>Save All", None,
            "File>Import>Surface", "File>Import>CAD", "File>Import>Volume", None,
            "File>Export>STL", "File>Export>Summary", "File>Export>Report"])
        self._recent_menu = file_menu.addMenu(tr("Recent Files"))
        self._rebuild_recent_menu()
        file_menu.addSeparator()
        file_menu.addAction(self.actions["File>Exit"])
        menu(tr("Edit") + "(&E)", [
            "Edit>Undo", "Edit>Redo", None, "Edit>Copy", "Edit>Paste",
            "Edit>Delete", "Edit>Rename", None, "Edit>Prev", "Edit>Next",
            "Edit>Search"])
        menu(tr("Mesh") + "(&M)", [
            "Mesh>Generate", "Mesh>GenerateVolume", "Mesh>GeneratePoly",
            "Mesh>GenerateTrimmer", "Mesh>Clear", None,
            "Mesh>Scale", "Mesh>Diagnostics", "Mesh>Convert2D", "Mesh>Repair"])
        menu(tr("Scene") + "(&S)", [
            "Scene>New", "Scene>AddDisplayer", None,
            "Scene>Fit", "Scene>Reset", None,
            "Scene>View>+x", "Scene>View>-x", "Scene>View>+y", "Scene>View>-y",
            "Scene>View>+z", "Scene>View>-z", "Scene>View>iso", None,
            "Scene>Solid", "Scene>Wireframe", "Scene>Edges", "Scene>Transparency"])
        menu(tr("Solution") + "(&N)", [
            "Solution>Run", "Solution>Pause", "Solution>Step", "Solution>Stop"])
        menu(tr("Tools") + "(&T)", [
            "Tools>Options", None,
            "Tools>Fingerprint", "Tools>Check Length", "Tools>Validate"])
        menu(tr("Connection") + "(&C)", ["Connection>Server"])
        menu(tr("Plot") + "(&P)", ["Plot>NYI"])
        menu(tr("Window") + "(&W)", [
            "Window>Tree", "Window>Props", "Window>Output", "Window>Plots",
            "Window>Cad"])
        menu(tr("Help") + "(&H)", ["Help>About"])

    def _build_toolbar(self):
        from PyQt5.QtCore import Qt as _Qt
        def make_tb(name):
            tb = QToolBar(name)
            tb.setObjectName(name)
            tb.setMovable(False)
            tb.setIconSize(QSize(22, 22))
            tb.setToolButtonStyle(_Qt.ToolButtonIconOnly)
            self.addToolBar(tb)
            return tb

        self.tb_file = make_tb("File")
        for key in ("File>New", "File>Open", "File>Close", "File>Save"):
            self.tb_file.addAction(self.actions[key])
        self.tb_edit = make_tb("Edit")
        for key in ("Edit>Copy", "Edit>Paste", "Edit>Prev", "Edit>Next"):
            self.tb_edit.addAction(self.actions[key])
        self.tb_mesh = make_tb("MeshGen")
        for key in ("File>Import>CAD", "File>Import>Surface", "Mesh>Repair",
                    "Mesh>Generate", "Mesh>GenerateVolume", "Mesh>Diagnostics"):
            self.tb_mesh.addAction(self.actions[key])
        self.tb_solve = make_tb("Solve")
        for key in ("Solution>Run", "Solution>Pause", "Solution>Step", "Solution>Stop"):
            self.tb_solve.addAction(self.actions[key])
        self.addToolBarBreak()
        self.tb_view = make_tb("View")
        for key in ("Scene>Fit", "Scene>Reset"):
            self.tb_view.addAction(self.actions[key])
        self.tb_view.addSeparator()
        for name in ("+x", "-x", "+y", "-y", "+z", "-z", "iso"):
            self.tb_view.addAction(self.actions["Scene>View>%s" % name])
        self.tb_disp = make_tb("Display")
        for key in ("Scene>Solid", "Scene>Wireframe", "Scene>Edges",
                    "Scene>Transparency", "Vis>MeshOn", "Vis>MeshOff",
                    "Vis>Derived", "Vis>SaveView", "Vis>RestoreView", "Vis>Rubber",
                    "Vis>Distance", "Vis>PartsFilter"):
            self.tb_disp.addAction(self.actions[key])
        self.tb_cad = make_tb("Cad")
        self.tb_cad.addAction(self.actions["Window>Cad"])
        self.tb_cad.addAction(self.actions["Vis>Derived"])
        self.tb_cad.addAction(self.actions["Cad>Section"])
        self.tb_cad.addAction(self.actions["Cad>Transform"])
        self.tb_cad.addAction(self.actions["Mesh>Repair"])
        self.tb_cad.setVisible(False)

    def _toggle_tree(self, on=True):
        self.split_left.setVisible(bool(self.actions["Window>Tree"].isChecked()
                                        or self.actions["Window>Props"].isChecked()))
        self.tree_pane.setVisible(self.actions["Window>Tree"].isChecked())

    def _toggle_props(self, on=True):
        self.split_left.setVisible(bool(self.actions["Window>Tree"].isChecked()
                                        or self.actions["Window>Props"].isChecked()))
        self.props_pane.setVisible(self.actions["Window>Props"].isChecked())

    def _toggle_output(self, on=True):
        block = getattr(self, "output_block", self.bottom_tabs)
        block.setVisible(self.actions["Window>Output"].isChecked())

    def _toggle_plots(self, on=True):
        pane = getattr(self, "plot_pane", None)
        if pane is None:
            return
        vis = self.actions["Window>Plots"].isChecked()
        self.bottom_tabs.setTabVisible(self.bottom_tabs.indexOf(pane), vis) if hasattr(
            self.bottom_tabs, "setTabVisible") else None
        if vis:
            self.bottom_tabs.setCurrentWidget(pane)

    def _wire_editor_ui(self):
        vp = getattr(self, "viewport", None)
        if vp is not None and hasattr(vp, "context_command"):
            vp.context_command.connect(self.on_view_context)

    def _sync_edit_actions(self):
        bus = getattr(self, "document", None)
        if bus is None:
            return
        model = getattr(self, "model", None)
        if model is not None and hasattr(model, "invalidate_g7"):
            model.invalidate_g7()
            pw = getattr(self, "props_widget", None)
            if pw is not None and pw._current is not None:
                pw.show_object(pw._current)
        self.actions["Edit>Undo"].setEnabled(self.document.bus.can_undo())
        self.actions["Edit>Redo"].setEnabled(self.document.bus.can_redo())
        title = "STAR-CCM+ .sim Viewer / Editor"
        if self.sim_path:
            title += " — %s" % self.sim_path
            if self.document.dirty:
                title += " *"
        self.setWindowTitle(title)

    def _on_document_event(self, kind, **kw):
        if kind in ("property", "transform", "created", "deleted", "undeleted"):
            self._refresh_after_edit(kind, **kw)
        if kind == "visibility":
            oid = kw.get("obj_id")
            if oid is not None:
                vis = kw.get("visible", True)
                if hasattr(self.tree_widget, "set_check"):
                    self.tree_widget.set_check(oid, vis)
                self.on_part_visibility(oid, vis)

    def _refresh_after_edit(self, kind, **kw):
        if self.model is None:
            return
        oid = kw.get("obj_id")
        if kind == "property" and oid is not None:
            obj = self.document.object(oid)
            if obj is not None and self.props_widget._current is obj:
                self.props_widget.show_object(obj)
            self._apply_live_property(obj, kw.get("key"), kw.get("value"))
            if kw.get("key") in ("PresentationName", "name"):
                self.tree_widget.rebuild()
            if kw.get("key") in ("Mesh", "Keys", "Collector", "Representation",
                                 "Opacity", "DisplayerColor"):
                if kw.get("key") in ("Mesh", "Keys", "Collector", "Representation"):
                    self._rebuild_graphics()
        if kind == "transform":
            self._apply_doc_transforms()
        if kind in ("created", "deleted", "undeleted"):
            self.tree_widget.rebuild()
            if getattr(self, "plot_pane", None) is not None:
                self.plot_pane.show_sim(self.sim)
            self._rebuild_graphics()

    def _rebuild_graphics(self):
        if HEADLESS or self.sim is None or self.model is None:
            return
        try:
            self._build_3d()
        except Exception as exc:
            self.msg("3D 刷新失败: %s" % exc, "warn")

    def _apply_live_property(self, obj, key, value):
        if obj is None or key is None:
            return
        if key in ("Opacity", "DisplayerColor", "Color", "MeshColor") or "Color" in str(key):
            for vp in self._iter_viewports():
                for _k, name, pid, actor in vp.actors:
                    if pid == obj.id or name == (obj.name or ""):
                        if key == "Opacity":
                            try:
                                actor.GetProperty().SetOpacity(float(value))
                            except Exception:
                                pass
                        elif isinstance(value, (list, tuple)) and len(value) >= 3:
                            actor.GetProperty().SetColor(float(value[0]), float(value[1]),
                                                         float(value[2]))
                if hasattr(vp, "render"):
                    vp.render()

    def _apply_doc_transforms(self):
        from star_gui_vtk import apply_actor_transform
        for vp in self._iter_viewports():
            for _k, _n, pid, actor in vp.actors:
                if pid in getattr(self.document, "baked_parts", set()):
                    continue
                t = self.document.transforms.get(pid)
                if t:
                    apply_actor_transform(actor, t[:3], t[3:6])
            if hasattr(vp, "render"):
                vp.render()

    def _kernel_nyi(self, what):
        self.msg("%s 需要网格/CAD 内核或 STAR-CCM+ 宏，当前禁用。见 star_gui_parity.md" % what,
                 "nyi")

    def _selected_obj(self):
        return self.tree_widget.current_object() if hasattr(self.tree_widget, "current_object") else None

    def _recent_paths(self):
        s = QSettings("stardecoding", "star_gui")
        raw = s.value("recent", [])
        if not raw:
            return []
        if isinstance(raw, str):
            return [raw]
        return [p for p in raw if p]

    def _remember_recent(self, path):
        s = QSettings("stardecoding", "star_gui")
        items = [p for p in self._recent_paths() if p != path]
        items.insert(0, path)
        s.setValue("recent", items[:8])
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self):
        menu = getattr(self, "_recent_menu", None)
        if menu is None:
            return
        menu.clear()
        paths = self._recent_paths()
        if not paths:
            act = menu.addAction("(空)")
            act.setEnabled(False)
            return
        for p in paths:
            menu.addAction(p, lambda checked=False, path=p: self.load_file(path))

    def cmd_solid(self):
        vp = self.current_viewport()
        if vp is not None:
            vp.set_representation("solid")

    # ---------------- 命令 ----------------
    def msg(self, text, level="info"):
        self.messages.log(text, level)

    def set_status(self, text):
        self.status_helper.set_stats(text)

    def _nyi(self, key):
        self.messages.nyi(key)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开 STAR-CCM+ 项目", "",
                                              "STAR-CCM+ (*.sim);;All files (*)")
        if path:
            self.load_file(path)

    def load_file(self, path):
        if self._thread and self._thread.isRunning():
            self.msg("已有文件在加载中", "warn")
            return
        self.progress.set_progress(5, "正在解析 %s ..." % os.path.basename(path))
        self._thread = QThread(self)
        self._worker = LoadWorker(path)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_loaded)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    def _finish_thread(self):
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
            self._thread = None
            self._worker = None

    def _on_loaded(self, sim, summary_text):
        self._finish_thread()
        self.sim = sim
        self.sim_path = sim.path
        self.summary_pane.show_summary(summary_text)
        self.progress.done("已加载 %s" % os.path.basename(sim.path))
        self.set_status("对象 %d · 分区 %d · 数组 %d" % (
            len(sim.objects), len(sim.sections), len(sim.arrays)))
        self.setWindowTitle("STAR-CCM+ .sim Viewer / Editor — %s" % sim.path)
        base = os.path.splitext(os.path.basename(sim.path))[0]
        self.bottom_tabs.setTabText(0, base)
        self.msg("starccm+ viewer  %s" % os.path.basename(sim.path))
        self.msg("已加载 %s" % sim.path)
        self._remember_recent(sim.path)
        self.on_file_loaded()   # M1+ 钩子

    def on_file_loaded(self):
        """文件加载后的后续构建（M1 建树/属性，M2 建 3D）。"""
        from star_gui_model import StarSceneModel
        self.model = StarSceneModel(self.sim)
        self.document.bind(self.sim, self.sim_path)
        self.tree_widget.set_model(self.model)
        self.props_widget.set_model(self.model)
        if getattr(self, "plot_pane", None) is not None:
            self.plot_pane.show_sim(self.sim)
        self.msg("仿真树已构建：%d 个顶层节点" % self.tree_widget.tree.topLevelItemCount())
        self._build_3d()
        self._sync_edit_actions()

    def _build_3d(self):
        """M2/M3：按场景构建 3D 标签页（场景→显示器颜色→视图相机）。

        无头模式下 viewport 是占位控件，仅更新消息，不做 QVTK 操作。
        """
        try:
            from star_gui_vtk import (apply_camera, build_mesh_actors,
                                      build_scene_actors)
            self.progress.set_progress(10, "构建 3D 场景 ...")
            scenes = self.model.scenes() if self.model else []
            self._clear_extra_graphics_tabs()
            if scenes:
                for sc in scenes:
                    from star_gui_panes import Star3DViewport
                    scene_obj = self.sim.objmap.get(sc["id"])
                    actors, cam = build_scene_actors(self.sim, scene_obj)
                    if HEADLESS:
                        from PyQt5.QtCore import Qt
                        from PyQt5.QtWidgets import QLabel
                        vp = QLabel("3D（无头）")
                        vp.setAlignment(Qt.AlignCenter)
                    else:
                        vp = Star3DViewport()
                    self.graphics_tabs.add_mesh_tab(sc["name"] or "Scene", vp)
                    if not HEADLESS:
                        from star_gui_vtk import volume_mesh_actors
                        extra, _vol = volume_mesh_actors(self.sim)
                        if extra:
                            actors = list(actors) + extra
                        vp.set_actors(actors)
                        apply_camera(vp.renderer, cam)
                        vp.picked.connect(self.on_picked)
                        if hasattr(vp, "context_command"):
                            vp.context_command.connect(self.on_view_context)
                self.graphics_tabs.setCurrentIndex(1)
                vp1 = self.graphics_tabs.widget(1)
                if vp1 is not None and hasattr(vp1, "set_actors"):
                    self.viewport = vp1
                self.msg("3D 场景：%d 个（含显示器/视图相机）" % len(scenes))
            else:
                actors = build_mesh_actors(self.sim)
                if HEADLESS:
                    from PyQt5.QtWidgets import QLabel
                    self.viewport = QLabel("3D（无头）")
                    self.viewport.setAlignment(Qt.AlignCenter)
                    self.graphics_tabs.add_mesh_tab("3D Mesh", self.viewport)
                else:
                    from star_gui_panes import Star3DViewport
                    if self.viewport is None or not hasattr(self.viewport, "set_actors"):
                        self.viewport = Star3DViewport()
                        self.graphics_tabs.add_mesh_tab("3D Mesh", self.viewport)
                    self.viewport.set_actors(actors)
                    if hasattr(self.viewport, "picked"):
                        try:
                            self.viewport.picked.connect(self.on_picked)
                        except Exception:
                            pass
                self.msg("3D 网格：%d 个 Part" % len(actors))
            self.progress.done("3D 就绪")
            self._apply_doc_transforms()
            try:
                m = self.sim.extract_mesh()
                nv = 0 if m.get("vertices") is None else len(m["vertices"])
                nf = 0 if m.get("faces") is None else len(m["faces"])
                self.status_helper.set_mesh(nv, nf)
            except Exception:
                pass
        except Exception as exc:  # noqa: BLE001
            self.msg("3D 构建失败: %s" % exc, "warn")

    def _clear_extra_graphics_tabs(self):
        """关掉 Info 以外的 3D 页，先 Finalize OpenGL 再丢控件。"""
        while self.graphics_tabs.count() > 1:
            w = self.graphics_tabs.widget(self.graphics_tabs.count() - 1)
            self.graphics_tabs.removeTab(self.graphics_tabs.count() - 1)
            self._dispose_3d_widget(w)

    def _dispose_3d_widget(self, w):
        if w is None:
            return
        try:
            w.hide()
        except Exception:
            pass
        try:
            if hasattr(w, "shutdown"):
                w.shutdown()
        except Exception:
            pass
        if getattr(self, "viewport", None) is w:
            self.viewport = None
        try:
            w.setParent(None)
            w.deleteLater()
        except Exception:
            pass

    def _on_graphics_tab_changed(self, idx):
        w = self.graphics_tabs.widget(idx)
        if w is not None and hasattr(w, "_on_shown"):
            w._show_tries = 0
            w._on_shown()

    def closeEvent(self, event):
        if self.solver_controller is not None:
            try:
                self.solver_controller.stop()
            except Exception:
                pass
        if not self.confirm_discard_dirty("未保存", "文档已修改，是否保存？"):
            event.ignore()
            return
        try:
            self._clear_extra_graphics_tabs()
        except Exception:
            pass
        super().closeEvent(event)

    def on_object_selected(self, obj):
        self.props_widget.show_object(obj)
        if obj is not None:
            self.set_status("选中 %s %s (id %d, %s)" % (
                obj.class_name or "?", obj.name or "", obj.id, obj.layer))
            self.status_helper.set_mode(obj.name or obj.class_name.split(".")[-1])
            if getattr(self, "document", None) is not None:
                self.document.push_selection(obj.id)
            if obj.class_name == "star.vis.Scene":
                self._activate_scene_tab(obj.name)
        else:
            self.status_helper.set_mode("对象")
        self._highlight_selection(obj)

    def _activate_scene_tab(self, name):
        tabs = getattr(self, "graphics_tabs", None)
        if tabs is None or not name:
            return
        for i in range(tabs.count()):
            if tabs.tabText(i) == name:
                tabs.setCurrentIndex(i)
                return

    def on_picked(self, info, xyz):
        """3D 拾取 → 状态栏坐标 + 树同步选中。测距模式累计两点。"""
        self.status_helper.set_coord(xyz)
        if getattr(self, "_measure_pts", None) is not None:
            self._measure_pts.append(tuple(xyz))
            if len(self._measure_pts) >= 2:
                a, b = self._measure_pts[0], self._measure_pts[1]
                dist = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5
                self.msg("测距: %.6g （(%.4g,%.4g,%.4g) → (%.4g,%.4g,%.4g)）" % (
                    dist, a[0], a[1], a[2], b[0], b[1], b[2]))
                self._measure_pts = None
                self.set_status("测距 %.6g" % dist)
                return
            self.set_status("测距：再拾取第二点")
            return
        pid = info[2] if info and len(info) > 2 else None
        name = info[1] if info and len(info) > 1 else ""
        if pid:
            self.tree_widget.select_object(pid)
            self.set_status("拾取 %s (id %s)  @ (%.4g, %.4g, %.4g)" % (
                name, pid, xyz[0], xyz[1], xyz[2]))

    def _highlight_selection(self, obj):
        """选中 Region/Boundary → 高亮对应 actor。Boundary 用 FaceTypes
        面子块 / PartSurface id，否则退回 Region→Parts。"""
        if obj is None or HEADLESS:
            return
        if obj.class_name not in ("star.common.Region", "star.common.Boundary"):
            return
        names = set()
        ids = set()
        if obj.class_name == "star.common.Boundary":
            from star_gui_model import part_surface_cad_ids
            ids.add(obj.id)
            if obj.name:
                names.add(obj.name)
            psg = self.sim.objmap.get(obj.dict.get("PartSurfaces") or -1)
            for k in (psg.dict.get("Keys") or []) if psg is not None else []:
                s = self.sim.objmap.get(k)
                if s is None:
                    continue
                ids.add(s.id)
                if s.name:
                    names.add(s.name)
            cad = part_surface_cad_ids(self.sim, obj)
            vp = self.current_viewport()
            if vp is not None:
                hit = 0
                for key, name, pid, actor in vp.actors:
                    show = pid in ids or name in names
                    actor.SetVisibility(show)
                    if show:
                        hit += 1
                if hit:
                    vp.render()
                    self.msg("高亮边界 %s（%d 个 actor，FaceTypes=%d）" % (
                        obj.name or obj.id, hit, len(cad)))
                    return
        if obj.class_name == "star.common.Region":
            parts = self.sim.objmap.get(obj.dict.get("Parts")) if obj.dict.get("Parts") else None
            plist = parts.dict.get("Keys") if parts is not None else []
            for k in plist:
                p = self.sim.objmap.get(k)
                if p is not None and p.name:
                    names.add(p.name)
        else:
            parent = self.sim.objmap.get(obj.dict.get("Parent") or -1)
            if parent is not None and parent.class_name == "star.common.Region":
                parts = self.sim.objmap.get(parent.dict.get("Parts") or -1)
                for k in (parts.dict.get("Keys") or []) if parts else []:
                    p = self.sim.objmap.get(k)
                    if p is not None and p.name:
                        names.add(p.name)
        vp = self.current_viewport()
        if vp is None:
            return
        for key, name, pid, actor in vp.actors:
            actor.SetVisibility(name in names or pid in ids)
        vp.render()
        self.msg("高亮 %d 个 Part（Region→Parts；边界优先 FaceTypes）" % len(names))

    def on_tree_check(self, obj_id, checked):
        """树勾选走 VisibilityCommand，避免绕过 Undo。"""
        if obj_id is None or getattr(self, "document", None) is None:
            return
        if self.document.is_visible(obj_id) == bool(checked):
            self.on_part_visibility(obj_id, checked)
            return
        from star_gui_commands import VisibilityCommand
        self.document.execute(VisibilityCommand(obj_id, checked))

    def confirm_discard_dirty(self, title, text):
        """脏文档确认。无头测试直接丢弃以免挂起对话框。"""
        if getattr(self, "document", None) is None or not self.document.dirty:
            return True
        if HEADLESS:
            return True
        ans = QMessageBox.question(
            self, title, text,
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if ans == QMessageBox.Cancel:
            return False
        if ans == QMessageBox.Save:
            return bool(self.cmd_save())
        return True

    def on_part_visibility(self, obj_id, checked):
        """树勾选 → 按 Scene / Displayer / Parts / Part 显隐 3D actor。"""
        if obj_id is None or HEADLESS or self.sim is None:
            return
        obj = self.sim.objmap.get(obj_id)
        if obj is None:
            return
        from star_gui_model import owning_mesh_part
        cn = obj.class_name or ""
        viewports = list(self._iter_viewports())
        if not viewports:
            return

        def apply(pred):
            for vp in viewports:
                for key, name, pid, actor in vp.actors:
                    if pred(key, name, pid):
                        actor.SetVisibility(checked)
                if hasattr(vp, "render"):
                    vp.render()

        if cn == "star.vis.Scene":
            tabs = self.graphics_tabs
            for i in range(tabs.count()):
                if tabs.tabText(i) == (obj.name or ""):
                    w = tabs.widget(i)
                    if hasattr(w, "actors"):
                        for _k, _n, _p, actor in w.actors:
                            actor.SetVisibility(checked)
                        if hasattr(w, "render"):
                            w.render()
            return
        if "Displayer" in cn:
            suf = ":%d" % obj.id
            apply(lambda key, name, pid: (key or "").endswith(suf))
            return
        if cn.endswith("PartGroup") or cn == "star.common.PartGroup":
            for o in self.sim.objects:
                if o.dict.get("Collector") == obj.id and "Displayer" in (o.class_name or ""):
                    suf = ":%d" % o.id
                    apply(lambda key, name, pid, s=suf: (key or "").endswith(s))
            return
        if "PlaneSection" in cn:
            apply(lambda key, name, pid: pid == obj.id or name == (obj.name or ""))
            return
        if "PartSurface" in cn or cn.endswith("Boundary") or cn == "star.common.Boundary":
            apply(lambda key, name, pid: pid == obj.id or name == (obj.name or ""))
            return
        part = owning_mesh_part(self.sim, obj)
        target_id = part.id if part is not None else obj.id
        target_name = (part.name if part is not None else obj.name) or ""
        apply(lambda key, name, pid: pid == target_id or pid == obj.id
              or name == target_name or name == (obj.name or ""))

    def _iter_viewports(self):
        tabs = getattr(self, "graphics_tabs", None)
        if tabs is None:
            return
        for i in range(tabs.count()):
            w = tabs.widget(i)
            if hasattr(w, "actors") and hasattr(w, "set_actors"):
                yield w

    def _on_failed(self, path, err):
        self._finish_thread()
        self.progress.set_progress(0, "加载失败")
        self.msg("加载失败: %s" % path, "error")
        self.msg(err, "error")
        QMessageBox.critical(self, "加载失败", "%s\n%s" % (path, err.splitlines()[-1]))

    def close_sim(self):
        self.sim = None
        self.sim_path = None
        self.summary_pane.show_summary("")
        self.set_status("已关闭")
        self.document.bind(None, None)
        self.model = None
        self.tree_widget.set_model(None)
        self.props_widget.set_model(None)
        self.props_widget.show_object(None)
        try:
            self._clear_extra_graphics_tabs()
        except Exception:
            pass
        self.setWindowTitle("STAR-CCM+ .sim Viewer / Editor")
        if getattr(self, "plot_pane", None) is not None:
            self.plot_pane.show_sim(None)

    def cmd_new(self):
        if not self.confirm_discard_dirty("新建", "文档已修改，是否保存？"):
            return
        self.document.mark_clean()
        self.close_sim()
        self.msg("新建空会话（尚未写入 .sim）")

    def cmd_reload(self):
        if not self.sim_path:
            return self.msg("没有可重新加载的路径", "warn")
        if not self.confirm_discard_dirty("重新加载", "文档已修改，重新加载将丢失未保存改动，是否保存？"):
            return
        path = self.sim_path
        self.document.mark_clean()
        self.load_file(path)

    def cmd_save(self):
        if not self.sim_path:
            return self.cmd_save_as()
        return self._write_sim(self.sim_path)

    def cmd_save_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "另存为", self.sim_path or "untitled.sim",
                                              "STAR-CCM+ (*.sim)")
        if not path:
            return False
        return self._write_sim(path)

    def _write_sim(self, path):
        if self.sim is None:
            self.msg("请先打开文件", "warn")
            return False
        try:
            cam = self._camera_from_viewport()
            scene = self._active_scene_object()
            if cam is not None and scene is not None:
                self.document.persist_view(scene.id, cam)
            from sim_writer import save_sim
            save_sim(self.sim, path, patches=self.document.patches,
                     created=self.document.created, src_path=self.sim_path,
                     array_patches=getattr(self.document, "array_patches", None),
                     deleted=getattr(self.document, "deleted", None))
            self.sim_path = path
            self.document.mark_clean()
            self.document.path = path
            self.msg("已保存 %s" % path)
            self._sync_edit_actions()
            return True
        except Exception as exc:
            self.msg("保存失败: %s" % exc, "error")
            QMessageBox.warning(self, "保存失败", str(exc))
            return False

    def cmd_undo(self):
        if self.document.undo():
            self.msg("撤销 %s" % self.document.bus.peek_redo())
            self._sync_edit_actions()

    def cmd_redo(self):
        if self.document.redo():
            self.msg("重做 %s" % self.document.bus.peek_undo())
            self._sync_edit_actions()

    def cmd_copy(self):
        obj = self._selected_obj()
        if obj is None:
            return
        self.document.clipboard = obj.id
        self.msg("已复制 %s" % (obj.name or obj.id))

    def cmd_paste(self):
        oid = self.document.clipboard
        if oid is None:
            return self.msg("剪贴板为空", "warn")
        from star_gui_commands import CopyObjectCommand
        self.document.execute(CopyObjectCommand(oid))
        self.msg("已粘贴副本")

    def cmd_delete(self):
        obj = self._selected_obj()
        if obj is None:
            return
        from star_gui_commands import DeleteObjectCommand
        self.document.execute(DeleteObjectCommand(obj.id))
        self.msg("已删除 %s（会话）" % (obj.name or obj.id))

    def cmd_rename(self):
        obj = self._selected_obj()
        if obj is None:
            return
        key = "PresentationName" if "PresentationName" in obj.dict else "name"
        old = obj.dict.get(key) or ""
        text, ok = QInputDialog.getText(self, "重命名", "名称:", text=str(old))
        if not ok or not text:
            return
        from star_gui_commands import RenameCommand
        self.document.execute(RenameCommand(obj.id, text, old, key))
        self.tree_widget.rebuild()
        self.tree_widget.select_object(obj.id)

    def cmd_search(self):
        if self.sim is None:
            return
        text, ok = QInputDialog.getText(self, "按名称搜索", "名称包含:")
        if not ok or not text:
            return
        needle = text.lower()
        for o in self.sim.objects:
            if o.name and needle in o.name.lower():
                self.tree_widget.select_object(o.id)
                self.on_object_selected(o)
                return
        self.msg("未找到 %s" % text, "warn")

    def cmd_sel_prev(self):
        oid = self.document.select_prev()
        if oid:
            self.tree_widget.select_object(oid)

    def cmd_sel_next(self):
        oid = self.document.select_next()
        if oid:
            self.tree_widget.select_object(oid)

    def cmd_generate_mesh(self):
        self._try_star_macro("生成表面网格")

    def cmd_generate_volume_mesh(self):
        """生成体网格：用本地 N 波内核跑默认流水线（表面细分→tet→质量→重编号）。

        输入取当前仿真抽取的水密表面；不平整/无网格则提示。结果存 self._volume_mesh_result，
        消息条回报单元数与质量。
        """
        self._volume_mesh_result = None
        if self.sim is None:
            return self.msg("请先打开 .sim", "warn")
        try:
            m = self.sim.extract_mesh()
        except Exception as exc:
            return self.msg("表面抽取失败: %s" % exc, "error")
        vet = (m or {}).get("vertices")
        fac = (m or {}).get("faces")
        if vet is None or fac is None or len(vet) < 4 or len(fac) < 4:
            return self.msg("当前无可用表面网格（可先 导入>表面 一个 STL）", "warn")
        from occ_repair import boundary_edges
        b, _ = boundary_edges(vet, fac)
        if len(b):
            return self.msg("表面不水密（%d 条开放边界），先修复再生成体网格" % len(b), "warn")
        import numpy as np
        try:
            diag = float(np.linalg.norm(np.asarray(vet, float).max(0)
                                        - np.asarray(vet, float).min(0)))
        except Exception:
            diag = 1.0
        cell_size = max(diag / 20.0, 1e-6)
        from mesh_pipeline import default_volume_mesh_pipeline
        pipe = default_volume_mesh_pipeline(vet, fac, cell_size=cell_size,
                                            refine_levels=0)
        cancel = {"off": True}
        dlg = None
        if not HEADLESS:
            from PyQt5.QtWidgets import QProgressDialog
            from PyQt5.QtCore import Qt
            dlg = QProgressDialog("生成体网格…", "取消", 0, len(pipe.stages), self)
            dlg.setWindowModality(Qt.WindowModal)
            dlg.setMinimumDuration(0)
            dlg.canceled.connect(lambda: cancel.__setitem__("off", False))
        vp = (self.graphics_tabs.current_viewport()
              if hasattr(self, "graphics_tabs") else None)

        def progress(i, name, total):
            if dlg is not None:
                dlg.setValue(i)
                dlg.setLabelText("%d/%d %s" % (i, total, name))
            self.msg("网格流水线：%d/%d %s" % (i, total, name))
            if vp is not None and hasattr(vp, "processEvents"):
                vp.processEvents()

        out = pipe.run(progress=progress,
                       canceled=(lambda: not cancel["off"]))
        if dlg is not None:
            dlg.setValue(len(pipe.stages))
        if out.get("cancelled"):
            return self.msg("体网格生成已取消（已完成 %d 段）" % out["stages_ran"], "warn")
        if not out.get("ok"):
            return self.msg("体网格生成失败", "error")
        ctx = out["ctx"]
        Vn, Cn = ctx["volume"]
        q = ctx.get("quality", {})
        self._volume_mesh_result = {"name": self.sim_path,
                                    "vertices": np.asarray(Vn, float),
                                    "cells": np.asarray(Cn, np.int64),
                                    "quality": q}
        vq = (q.get("volume") if isinstance(q, dict) else None) or {}
        amax = (vq.get("edge_aspect", {}).get("max", 0.0)
                if isinstance(vq.get("edge_aspect"), dict) else 0.0)
        msg = "体网格完成：%d 四面体 / %d 节点，负单元=%s，最大边纵横比=%.2f" % (
            len(Cn), len(Vn), vq.get("n_negative", 0), amax)
        return self.msg(msg)

    def _poly_surface_input(self):
        """poly/trimmer 共用：抽取当前水密表面 (V,F)；失败已提示并返回 None。"""
        if self.sim is None:
            self.msg("请先打开 .sim", "warn")
            return None
        try:
            m = self.sim.extract_mesh()
        except Exception as exc:
            self.msg("表面抽取失败: %s" % exc, "error")
            return None
        vet = (m or {}).get("vertices")
        fac = (m or {}).get("faces")
        if vet is None or fac is None or len(vet) < 4 or len(fac) < 4:
            self.msg("当前无可用表面网格（可先 导入>表面 一个 STL）", "warn")
            return None
        from occ_repair import boundary_edges
        b, _ = boundary_edges(vet, fac)
        if len(b):
            self.msg("表面不水密（%d 条开放边界），先修复再生成" % len(b), "warn")
            return None
        return vet, fac

    def cmd_generate_poly_mesh(self):
        """生成多面体网格：tet(Delaunay) → Voronoi 对偶（N3b poly 内核）。

        结果存 self._poly_mesh_result，消息条回报单元数/总体积。
        """
        self._poly_mesh_result = None
        got = self._poly_surface_input()
        if got is None:
            return
        vet, fac = got
        import numpy as np
        try:
            diag = float(np.linalg.norm(np.asarray(vet, float).max(0)
                                        - np.asarray(vet, float).min(0)))
        except Exception:
            diag = 1.0
        try:
            from mesh_poly import poly_mesh
            out = poly_mesh(vet, fac, spacing=max(diag / 8.0, 1e-6))
        except ValueError as exc:
            return self.msg("多面体网格生成失败：%s" % exc, "error")
        except Exception as exc:                      # 内核意外错误也不崩 GUI
            return self.msg("多面体网格生成异常：%s" % exc, "error")
        if not out.get("ok"):
            return self.msg("多面体网格生成失败（无有效单元）", "error")
        self._poly_mesh_result = {
            "name": self.sim_path, "cells": out["poly"]["cells"],
            "n_cells": out["n_cells"], "volume_total": out["volume_total"],
            "quality": out["quality"], "method": out["method"]}
        q = out["quality"]
        return self.msg("多面体网格完成：%d 单元 / 总体积=%.4g，最小单元体积=%.3g，"
                        "平均面数=%.1f（%s）" % (
                            out["n_cells"], out["volume_total"],
                            q.get("volume_min", 0.0), q.get("faces_mean", 0.0),
                            out["method"]))

    def cmd_generate_trimmer_mesh(self):
        """生成 Trimmer 网格：八叉树加密 + 表面切割单元（N3b trimmer 内核）。

        结果存 self._trimmer_mesh_result，消息条回报六面体/切割单元数与体积。
        """
        self._trimmer_mesh_result = None
        got = self._poly_surface_input()
        if got is None:
            return
        vet, fac = got
        import numpy as np
        try:
            diag = float(np.linalg.norm(np.asarray(vet, float).max(0)
                                        - np.asarray(vet, float).min(0)))
        except Exception:
            diag = 1.0
        try:
            from mesh_trimmer import trimmer_mesh
            out = trimmer_mesh(vet, fac, cell_size=max(diag / 8.0, 1e-6))
        except ValueError as exc:
            return self.msg("Trimmer 网格生成失败：%s" % exc, "error")
        except Exception as exc:
            return self.msg("Trimmer 网格生成异常：%s" % exc, "error")
        if not out.get("ok"):
            return self.msg("Trimmer 网格生成失败（无有效单元）", "error")
        self._trimmer_mesh_result = {
            "name": self.sim_path, "cells": out["cells"],
            "n_cells": out["n_cells"], "n_hex": out["n_hex"],
            "n_cut": out["n_cut"], "volume_total": out["volume_total"],
            "quality": out["quality"], "max_depth": out["max_depth"]}
        return self.msg("Trimmer 网格完成：%d 单元（%d 六面体 + %d 切割），"
                        "总体积=%.4g，最深 %d 级" % (
                            out["n_cells"], out["n_hex"], out["n_cut"],
                            out["volume_total"], out["max_depth"]))

    def _find_starccm(self):
        env = os.environ.get("STARCCM_HOME") or ""
        names = ("starccmw.exe", "starccm+.exe", "starccm.exe")
        candidates = []
        if env:
            for n in names:
                candidates.append(os.path.join(env, "star", "bin", n))
                candidates.append(os.path.join(env, n))
        for root in (r"D:\training\starccm", r"C:\Program Files\Siemens"):
            if not os.path.isdir(root):
                continue
            for n in names:
                candidates.append(os.path.join(root, n))
        for p in candidates:
            if p and os.path.isfile(p):
                return p
        return None

    def _try_star_macro(self, what):
        exe = self._find_starccm()
        if exe is None:
            return self._kernel_nyi(what)
        if not self.sim_path:
            return self._kernel_nyi(what)
        if HEADLESS:
            self.msg("%s：无头模式不启动 STAR 宏（将在工作副本上跑）" % what, "nyi")
            return
        ans = QMessageBox.question(
            self, what,
            "将在临时副本上调用\n%s\n不会改写当前打开的教程原件。继续？" % exe)
        if ans != QMessageBox.Yes:
            return self.msg("已取消 %s" % what, "warn")
        try:
            from star_macro import run_star_macro_on_copy, write_generate_mesh_macro
            import tempfile
            tmp = tempfile.mkdtemp(prefix="star_gui_macro_")
            macro = write_generate_mesh_macro(tmp)
            work, code, log = run_star_macro_on_copy(exe, self.sim_path, macro)
            self.msg("%s 结束 code=%s 目录=%s" % (what, code, work))
            if log:
                self.msg(log[-800:])
            out = os.path.join(work, "out.sim")
            if os.path.isfile(out):
                self.load_file(out)
        except Exception as exc:
            self.msg("%s 失败: %s" % (what, exc), "error")

    def cmd_scale_mesh(self):
        obj = self._selected_obj()
        if obj is None:
            return
        from star_gui_model import owning_mesh_part
        part = owning_mesh_part(self.sim, obj) if self.sim else obj
        if part is None:
            return self.msg("请选择一个部件", "warn")
        factor, ok = QInputDialog.getDouble(self, "缩放网格", "比例:", 1.0, 0.001, 1000.0, 4)
        if not ok:
            return
        from star_gui_commands import TransformPartCommand
        self.document.execute(TransformPartCommand(part.id, scale=(factor, factor, factor)))
        self.msg("已缩放 %s × %g" % (part.name or part.id, factor))

    def cmd_mesh_diag(self):
        if self.sim is None:
            return
        try:
            m = self.sim.extract_mesh()
            ok = bool(m.get("consistent"))
            vol = self.sim.extract_volume_mesh()
            self.msg("网格诊断: faces=%s verts=%s consistent=%s flag=%s/%s  volume=%s" % (
                None if m.get("faces") is None else len(m["faces"]),
                None if m.get("vertices") is None else len(m["vertices"]),
                ok, m.get("face_flag"), m.get("vertex_flag"),
                ("%s %s" % (vol.get("kind"), vol.get("count"))) if vol.get("ok")
                else vol.get("reason")))
        except Exception as exc:
            self.msg("网格诊断失败: %s" % exc, "error")

    def cmd_import_surface(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入表面", "", "STL (*.stl);;All (*)")
        if not path:
            return
        self.import_surface_from_path(path)

    def import_surface_from_path(self, path):
        """导入 STL/表面 → MeshPart，顶点/面写入对象图 Imported* 字段。"""
        name = os.path.splitext(os.path.basename(path))[0]
        verts = faces = None
        if os.path.isfile(path):
            try:
                from mesh_io import read_surface
                verts, faces = read_surface(path)
            except Exception as exc:
                self.msg("STL 读取失败: %s" % exc, "warn")
        oid = self._new_imported_mesh_part(name, verts, faces)
        if oid and verts is not None:
            self.msg("已导入 %s：%d 顶点 %d 面（写入 ImportedVertices/Faces）" % (
                name, len(verts), len(faces)))
        elif oid:
            self.msg("已记录导入占位 %s（未读到 STL）" % name, "warn")
        return oid

    def _new_imported_mesh_part(self, name, verts=None, faces=None, extra=None):
        if self.sim is None:
            return None
        src = None
        for o in self.sim.objects:
            if (o.class_name or "").endswith("MeshPart"):
                src = o
                break
        if src is None:
            oid = self.document.create_session_object("star.meshing.MeshPart", name)
        else:
            from star_gui_commands import CopyObjectCommand
            cmd = CopyObjectCommand(src.id)
            self.document.execute(cmd)
            oid = cmd.new_id
            if oid:
                self.document.set_property(oid, "PresentationName", name)
        if oid and verts is not None and faces is not None:
            self.document.set_property(oid, "ImportedVertices", verts)
            self.document.set_property(oid, "ImportedFaces", faces)
            self.document.set_property(oid, "VertexCount", len(verts))
            self.document.set_property(oid, "TriangleCount", len(faces))
            for k, v in (extra or {}).items():
                self.document.set_property(oid, k, v)
            if self.sim is not None:
                self.sim._part_meshes_cache = None
        return oid

    def cmd_import_cad(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "导入 CAD", "", "Surface (*.stl *.obj);;All (*)")
        if not path:
            return
        self.import_surface_from_path(path)
        self.msg("无 Parasolid：CAD 按 STL/OBJ 三角化导入", "warn")

    def cmd_import_volume(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "导入体网格", "",
            "CCM (*.ccm *.ccmg);;All (*)")
        if not path:
            return
        self.import_volume_from_path(path)

    def import_volume_from_path(self, path):
        """导入 CCM 体网格：边界三角化写入 MeshPart（与 STL 导入同一 Persist 路径）。"""
        from ccm_io import ccmio_available, ccmio_unavailable_reason, read_ccm
        if not os.path.isfile(path):
            self.msg("CCM 文件不存在: %s" % path, "warn")
            return None
        if not ccmio_available():
            self.msg("无法导入 CCM：%s" % ccmio_unavailable_reason(), "warn")
            return None
        try:
            mesh = read_ccm(path)
        except Exception as exc:
            self.msg("CCM 读取失败: %s" % exc, "error")
            return None
        verts, faces = mesh.get("vertices") or [], mesh.get("faces") or []
        if not verts or not faces:
            self.msg("CCM 没有可显示的边界面", "warn")
            return None
        name = os.path.splitext(os.path.basename(path))[0]
        oid = self._new_imported_mesh_part(
            name, verts, faces, extra={"CcmCellCount": int(mesh.get("n_cells") or 0)})
        regs = mesh.get("regions") or []
        self.msg("已导入 CCM %s：%d 顶点 %d 三角（边界） %d 体单元 %d 边界区" % (
            os.path.basename(path), len(verts), len(faces),
            int(mesh.get("n_cells") or 0), len(regs)))
        return oid

    def cmd_options(self):
        QMessageBox.information(self, "选项",
                                "语言: 中文\n对标表: star_gui_parity.md\n求解运算: 禁用")

    def cmd_show_plots(self):
        self.actions["Window>Plots"].setChecked(True)
        self._toggle_plots()

    # ---------------- P10 求解运行闭环 ----------------
    def _sync_solution_actions(self, state):
        """按运行态切换 Run/Pause/Step/Stop 可用性。"""
        from solver_run import SolverState
        self.actions["Solution>Run"].setEnabled(
            state != SolverState.RUNNING)
        self.actions["Solution>Pause"].setEnabled(
            state == SolverState.RUNNING)
        self.actions["Solution>Step"].setEnabled(
            state == SolverState.PAUSED)
        self.actions["Solution>Stop"].setEnabled(
            state in (SolverState.RUNNING, SolverState.PAUSED))

    def _on_solver_state(self, state):
        self._sync_solution_actions(state)
        self.set_status("求解器 %s" % state)

    def _on_solver_iterated(self, iteration, payload):
        if self.solver_controller is None:
            return
        items = self.solver_controller.curve_items()
        if getattr(self, "plot_pane", None) is not None:
            self.plot_pane.canvas.set_series(items[:3])
            self.plot_pane.canvas.setVisible(bool(items))
        res = payload.get("residual")
        self.set_status("迭代 %d  残差 %s" % (
            iteration, ("%.4g" % res) if res is not None else "-"))

    def _on_solver_finished(self, result):
        from solver_run import SolverState
        state = result.get("state", SolverState.COMPLETED)
        self._sync_solution_actions(state)
        self.set_status("求解结束: %s" % state)
        self._append_solver_report()

    def _on_solver_failed(self, message):
        from solver_run import SolverState
        self._sync_solution_actions(SolverState.ERROR)
        self.set_status("求解错误")
        self.msg("求解错误: %s" % message, "error")

    def _append_solver_report(self):
        if self.solver_controller is None:
            return
        for line in self.solver_controller.report_lines():
            self.messages.log(line, "info")

    def _reset_solution_plots(self):
        if getattr(self, "plot_pane", None) is not None:
            self.plot_pane.canvas.set_series([])

    def cmd_solution_run(self):
        if self.solver_controller is None:
            return
        if not self.solver_controller.thread_alive():
            self._reset_solution_plots()
            if getattr(self, "plot_pane", None) is not None and self.bottom_tabs:
                self.bottom_tabs.setCurrentWidget(self.plot_pane)
        self.solver_controller.begin()

    def cmd_solution_pause(self):
        if self.solver_controller is not None:
            self.solver_controller.pause()

    def cmd_solution_step(self):
        if self.solver_controller is not None:
            self.solver_controller.step()

    def cmd_solution_stop(self):
        if self.solver_controller is not None:
            self.solver_controller.stop()

    def cmd_toggle_cad(self, on=False):
        show = self.actions["Window>Cad"].isChecked()
        if hasattr(self, "tb_cad"):
            self.tb_cad.setVisible(show)
        self.msg("3D-CAD 模式 %s（外壳；草图/拉伸需几何内核）" % ("开" if show else "关"))

    def cmd_vis_mesh(self, on):
        vp = self.current_viewport()
        if vp is None:
            return
        vp.set_representation("edges" if on else "solid")

    def cmd_create_derived(self):
        if self.sim is None:
            return self.msg("请先打开文件", "warn")
        parent = None
        for o in self.sim.objects:
            if (o.class_name or "").endswith("DerivedPartManager"):
                parent = o.id
                break
        oid = self.document.create_session_object(
            "star.vis.PlaneSection", "Plane Section", parent=parent)
        self.msg("已创建派生零件 Plane Section（会话）")
        if hasattr(self.tree_widget, "select_object"):
            self.tree_widget.select_object(oid)
        return oid

    def cmd_new_scene(self):
        if self.sim is None:
            return
        src = None
        for o in self.sim.objects:
            if o.class_name == "star.vis.Scene":
                src = o
                break
        if src is None:
            return self.document.create_session_object("star.vis.Scene", "Scene")
        from star_gui_commands import CopyObjectCommand
        cmd = CopyObjectCommand(src.id)
        self.document.execute(cmd)
        self.msg("已新建场景（会话）")
        return cmd.new_id

    def cmd_add_displayer(self):
        if self.sim is None:
            return
        src = None
        obj = self._selected_obj()
        if obj is not None and "Displayer" in (obj.class_name or ""):
            src = obj
        if src is None:
            for o in self.sim.objects:
                if "Displayer" in (o.class_name or "") and "Manager" not in (o.class_name or ""):
                    src = o
                    break
        if src is None:
            return self.msg("没有可复制的显示器", "warn")
        from star_gui_commands import CopyObjectCommand
        cmd = CopyObjectCommand(src.id)
        self.document.execute(cmd)
        self.msg("已添加显示器（会话）")
        return cmd.new_id

    def cmd_representation(self):
        obj = self._selected_obj()
        if obj is None or "Displayer" not in (obj.class_name or ""):
            return self.msg("请选择显示器", "warn")
        cur = bool(obj.dict.get("Mesh"))
        from star_gui_commands import SetPropertyCommand
        self.document.execute(SetPropertyCommand(obj.id, "Mesh", (not cur), cur))
        self.msg("Representation Mesh=%s" % (not cur))

    def _active_scene_object(self):
        if self.sim is None:
            return None
        tabs = getattr(self, "graphics_tabs", None)
        name = tabs.tabText(tabs.currentIndex()) if tabs is not None else ""
        for o in self.sim.objects:
            if o.class_name == "star.vis.Scene" and (o.name or "") == name:
                return o
        for o in self.sim.objects:
            if o.class_name == "star.vis.Scene":
                return o
        return None

    def _camera_from_viewport(self, vp=None):
        vp = vp or self.current_viewport()
        if vp is None or not hasattr(vp, "renderer"):
            return None
        cam = vp.renderer.GetActiveCamera()
        return {
            "position": cam.GetPosition(),
            "focal": cam.GetFocalPoint(),
            "view_up": cam.GetViewUp(),
            "parallel_scale": cam.GetParallelScale(),
        }

    def cmd_save_view(self):
        cam = self._camera_from_viewport()
        if cam is None:
            return
        self.document.saved_views["default"] = cam
        scene = self._active_scene_object()
        if scene is not None:
            self.document.persist_view(scene.id, cam)
        self.msg("已保存视图到 CurrentView")

    def cmd_restore_view(self):
        cam = self.document.saved_views.get("default")
        if cam is None and self.sim is not None:
            scene = self._active_scene_object()
            if scene is not None:
                from star_gui_vtk import scene_camera
                cam = scene_camera(self.sim, scene)
        vp = self.current_viewport()
        if cam is None or vp is None or not hasattr(vp, "renderer"):
            return self.msg("没有已保存的视图", "warn")
        from star_gui_vtk import apply_camera
        apply_camera(vp.renderer, cam)
        if hasattr(vp, "render"):
            vp.render()
        self.msg("已恢复视图")

    def cmd_cad_repair(self):
        if self.actions.get("Window>Cad") and not self.actions["Window>Cad"].isChecked():
            self.actions["Window>Cad"].setChecked(True)
            self.cmd_toggle_cad()
        self.msg("三角化修复预览：填洞/缝边需 CAD 内核，当前仅记录请求", "nyi")

    def cmd_cad_section(self):
        if self.actions.get("Window>Cad"):
            self.actions["Window>Cad"].setChecked(True)
            self.cmd_toggle_cad()
        self.cmd_create_derived()
        self.msg("3D-CAD 剖面：已创建会话 Plane Section（三角化预览）")

    def cmd_cad_transform(self):
        if self.actions.get("Window>Cad"):
            self.actions["Window>Cad"].setChecked(True)
            self.cmd_toggle_cad()
        self.cmd_transform()

    def cmd_transform(self):
        obj = self._selected_obj()
        if obj is None or self.sim is None:
            return
        from star_gui_model import owning_mesh_part
        part = owning_mesh_part(self.sim, obj) or obj
        text, ok = QInputDialog.getText(
            self, "变换", "平移 dx,dy,dz  缩放 sx,sy,sz:", text="0,0,0  1,1,1")
        if not ok or not text:
            return
        parts = text.replace(";", " ").split()
        def triple(s, default):
            xs = [p.strip() for p in s.replace(",", " ").split() if p.strip()]
            if len(xs) != 3:
                return default
            try:
                return tuple(float(x) for x in xs)
            except ValueError:
                return default
        translate = triple(parts[0] if parts else "0,0,0", (0.0, 0.0, 0.0))
        scale = triple(parts[1] if len(parts) > 1 else "1,1,1", (1.0, 1.0, 1.0))
        from star_gui_commands import TransformPartCommand
        self.document.execute(TransformPartCommand(part.id, translate=translate, scale=scale))
        self.msg("已变换 %s" % (part.name or part.id))

    def cmd_rubber_zoom(self):
        vp = self.current_viewport()
        if vp is not None and hasattr(vp, "enable_rubber_zoom"):
            vp.enable_rubber_zoom(True)
            self.msg("框选缩放：在 3D 视口拖出矩形后松开（再点适配视图恢复）")
        else:
            self.msg("框选缩放需要 3D 视口")

    def cmd_distance(self):
        self._measure_pts = []
        self.msg("测距：连续拾取两个 3D 点")

    def apply_parts_filter(self, group_id, keys):
        obj = self.document.object(group_id) if self.document else None
        if obj is None:
            return False
        from star_gui_commands import SetPropertyCommand
        self.document.execute(SetPropertyCommand(
            group_id, "Keys", list(keys), list(obj.dict.get("Keys") or [])))
        return True

    def cmd_edit_parts_filter(self):
        if self.sim is None:
            return
        obj = self._selected_obj()
        pg = None
        if obj is not None and (obj.class_name or "").endswith("PartGroup"):
            pg = obj
        elif obj is not None and "Displayer" in (obj.class_name or ""):
            pg = self.sim.objmap.get(obj.dict.get("Collector") or -1)
        if pg is None:
            return self.msg("请选择显示器或 Parts 组", "warn")
        parts = [o for o in self.sim.objects
                 if (o.class_name or "").endswith("MeshPart")
                 or (o.class_name or "").endswith("CadPart")
                 or "PartSurface" in (o.class_name or "")]
        choices = [(p.id, "%s (%d)" % (p.name or "?", p.id)) for p in parts]
        if not choices:
            return self.msg("没有可选零件", "warn")
        if HEADLESS:
            return
        from star_gui_panes import PartsFilterDialog
        dlg = PartsFilterDialog(choices, list(pg.dict.get("Keys") or []), self)
        if dlg.exec_() != dlg.Accepted:
            return
        keys = dlg.selected_ids()
        self.apply_parts_filter(pg.id, keys)
        self.msg("Parts 过滤器 Keys=%s" % keys)

    def cmd_scalar_color(self):
        if self.sim is None:
            return
        from star_gui_vtk import color_actors_by_array, part_meshes
        parts = part_meshes(self.sim)
        nverts = set(int(p["vertices"].shape[0]) for p in parts if p.get("vertices") is not None)
        nfaces = set(int(p.get("triangles") or 0) for p in parts)
        picked = None
        for i, a in enumerate(self.sim.arrays):
            if a.get("type") not in ("Float8", "Float4"):
                continue
            n = a.get("count") or 0
            if n in nverts or n in nfaces:
                picked = (i, a, n in nverts)
                break
        if picked is None:
            self.msg("无解场数组，标量着色禁用", "nyi")
            return
        idx, _a, on_pts = picked
        data = self.sim.array_data(idx)
        colored = 0
        if not HEADLESS:
            for vp in self._iter_viewports():
                colored += color_actors_by_array(vp.actors, data, on_points=on_pts)
                if hasattr(vp, "render"):
                    vp.render()
        self.msg("标量着色：数组[%d] n=%d → %d 个 actor" % (
            idx, int(data.size if hasattr(data, "size") else len(data)), colored))

    def cmd_official_lut(self):
        """G8：按场景官方参数渲染——PredefinedLookupTable 色表 + 场全局范围。"""
        if self.sim is None:
            return
        s8 = self.sim.extract_scene_display()
        if not s8.get("ok"):
            return self.msg("场景解码：%s" % s8.get("reason"), "nyi")
        entry = None
        for s in s8["scenes"]:
            for d in s["displayers"]:
                if d.get("colormap"):
                    entry = (s, d)
                    break
            if entry:
                break
        if entry is None:
            return self.msg("场景无 ScalarDisplayer 颜色映射", "nyi")
        scene, disp = entry
        cm = disp["colormap"]
        from star_gui_vtk import color_actors_by_array, lut_from_colormap
        lut = lut_from_colormap(cm.get("values"), cm.get("alphas"))
        if lut is None:
            return self.msg("颜色映射断点非法（%r）" % cm.get("name"), "nyi")
        sf = self.sim.extract_solution_fields()
        data = None
        fname = (disp.get("field") or {}).get("name", "")
        if sf.get("ok") and fname:
            base = fname.split(":")[0].strip().lower()
            want_mag = "magnitude" in fname.lower()
            for f in sf["fields"]:
                tag = f["name"].lower()
                if base and base in tag and (not want_mag or "magnitude" in tag):
                    data = sf["data"].get(f["name"])
                    break
        if data is None:
            return self.msg("未找到解场数据匹配 %r（色表 %r 已就绪）" % (
                fname, cm.get("name")), "nyi")
        rng = (disp.get("field") or {}).get("range")
        lo = hi = None
        if isinstance(rng, list) and len(rng) == 2:
            lo, hi = float(rng[0]), float(rng[1])
        if lo is None:
            import numpy as np
            arr = np.asarray(data, dtype=np.float64).reshape(-1)
            lo, hi = float(arr.min()), float(arr.max())
        lut.SetTableRange(lo, hi)
        colored = 0
        if not HEADLESS:
            for vp in self._iter_viewports():
                colored += color_actors_by_array(
                    vp.actors, data, on_points=False, lut=lut)
                if hasattr(vp, "render"):
                    vp.render()
        self.msg("官方色表 %r（场 %r 范围 %.4g..%.4g）→ %d 个 actor（场景 %r）" % (
            cm.get("name"), fname, lo, hi, colored, scene["name"]))

    def on_property_edited(self, obj, key, value):
        if obj is None:
            return
        from star_gui_commands import SetPropertyCommand
        self.document.execute(SetPropertyCommand(obj.id, key, value, obj.dict.get(key)))

    def on_tree_context(self, key, obj):
        dispatch = {
            "edit": lambda: self.on_object_selected(obj),
            "rename": self.cmd_rename,
            "copy": self.cmd_copy,
            "paste": self.cmd_paste,
            "delete": self.cmd_delete,
            "hide": lambda: self._ctx_vis(obj, False),
            "show": lambda: self._ctx_vis(obj, True),
            "show_only": lambda: self._ctx_show_only(obj),
            "highlight": lambda: self._highlight_selection(obj),
            "transform": self.cmd_transform,
            "assign_region": lambda: self.cmd_assign_region(obj),
            "execute_mesh": lambda: self._try_star_macro("执行网格操作"),
            "new_automesh": lambda: self._kernel_nyi("新建自动网格"),
            "cad_mode": lambda: (self.actions["Window>Cad"].setChecked(True),
                                 self.cmd_toggle_cad()),
            "add_displayer": self.cmd_add_displayer,
            "representation": self.cmd_representation,
            "parts_filter": self.cmd_edit_parts_filter,
        }
        fn = dispatch.get(key)
        if fn:
            fn()

    def _ctx_vis(self, obj, visible):
        if obj is None:
            return
        from star_gui_commands import VisibilityCommand
        self.document.execute(VisibilityCommand(obj.id, visible))

    def _ctx_show_only(self, obj):
        if obj is None:
            return
        ids = []
        for vp in self._iter_viewports():
            for _k, _n, pid, _a in vp.actors:
                if pid is not None:
                    ids.append(pid)
        if not ids:
            ids = [obj.id]
        from star_gui_commands import ShowOnlyCommand
        self.document.execute(ShowOnlyCommand(obj.id, ids))

    def cmd_assign_region(self, obj=None):
        obj = obj or self._selected_obj()
        if obj is None or self.sim is None:
            return
        regions = [o for o in self.sim.objects if o.class_name == "star.common.Region"]
        if not regions:
            return self.msg("没有区域", "warn")
        names = [r.name or str(r.id) for r in regions]
        name, ok = QInputDialog.getItem(self, "指定到区域", "区域:", names, 0, False)
        if not ok:
            return
        region = regions[names.index(name)]
        pg = self.sim.objmap.get(region.dict.get("Parts") or -1)
        if pg is None:
            return
        keys = list(pg.dict.get("Keys") or [])
        if obj.id not in keys:
            keys.append(obj.id)
            from star_gui_commands import SetPropertyCommand
            self.document.execute(SetPropertyCommand(pg.id, "Keys", keys, list(pg.dict.get("Keys") or [])))
        self.msg("已将 %s 指定到 %s" % (obj.name or obj.id, name))

    def on_view_context(self, key):
        if key == "fit":
            return self.cmd_fit()
        if key == "reset":
            return self.cmd_reset()
        if key.startswith("view:"):
            return self.cmd_view(key.split(":", 1)[1])
        if key == "solid":
            return self.cmd_solid()
        if key == "wire":
            return self.cmd_toggle_wire()
        if key == "edges":
            return self.cmd_edges_only()
        if key == "transp":
            return self.cmd_transparency()
        if key == "mesh_on":
            return self.cmd_vis_mesh(True)
        if key == "mesh_off":
            return self.cmd_vis_mesh(False)
        if key == "copy_image":
            vp = self.current_viewport()
            if vp is not None and hasattr(vp, "copy_image_to_clipboard"):
                vp.copy_image_to_clipboard()
                self.msg("图像已复制")
            return
        if key == "align_normal":
            vp = self.current_viewport()
            if vp is not None:
                vp.set_view("+z")
            return
        if key == "derived":
            return self.cmd_create_derived()
        if key == "distance":
            return self.cmd_distance()
        if key == "rubber":
            return self.cmd_rubber_zoom()

    def cmd_fingerprint(self):
        if not self.sim:
            return self.msg("请先打开文件", "warn")
        fp = self.sim.version_fingerprint()
        self.msg("fingerprint banner=%s release=%s mode=%s hdr=%s" % (
            fp["banner_version"], fp["release"], fp["state_mode"],
            ",".join(fp["header_keys"] or [])))

    def cmd_check_length(self):
        if not self.sim:
            return self.msg("请先打开文件", "warn")
        chk = self.sim.check_state_length()
        self.msg("length check: %s" % chk["detail"])

    def _require_sim(self):
        if self.sim is None:
            self.msg("请先打开文件", "warn")
            return None
        return self.sim

    def cmd_export_stl(self):
        sim = self._require_sim()
        if sim is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出 STL", "mesh.stl", "STL (*.stl)")
        if path:
            self.export_stl_to(path)

    def export_stl_to(self, path):
        sim = self._require_sim()
        if sim is None:
            return
        try:
            obj = self._selected_obj()
            if obj is not None and (obj.class_name or "") == "star.vis.Scene":
                from mesh_io import export_scene_stl
                export_scene_stl(sim, obj, path)
                self.msg("已按场景导出 STL: %s (%s)" % (path, obj.name or obj.id))
                self.progress.done("STL 导出完成")
                return
            if obj is not None:
                from star_gui_model import owning_mesh_part
                part = owning_mesh_part(sim, obj)
                if part is not None:
                    from mesh_io import export_part_stl
                    export_part_stl(sim, part.id, path)
                    self.msg("已按 Part 导出 STL: %s (%s)" % (path, part.name or part.id))
                    self.progress.done("STL 导出完成")
                    return
            sim.export_stl(path)
            self.msg("STL 已导出: %s" % path)
            self.progress.done("STL 导出完成")
        except Exception as exc:  # noqa: BLE001
            self.msg("STL 导出失败: %s" % exc, "error")

    def cmd_export_summary(self):
        sim = self._require_sim()
        if sim is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出摘要", "summary.txt", "Text (*.txt)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(sim.summary())
            self.msg("摘要已导出: %s" % path)

    def cmd_export_report(self):
        sim = self._require_sim()
        if sim is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出报告", "report.json", "JSON (*.json)")
        if path:
            import json
            with open(path, "w", encoding="utf-8") as f:
                json.dump(sim.semantic_report(), f, indent=1, default=str)
            self.msg("报告已导出: %s" % path)

    def current_viewport(self):
        """当前 Graphics 标签中的 3D 视口（场景切换后 Fit/线框应对此窗口生效）。"""
        tabs = getattr(self, "graphics_tabs", None)
        if tabs is not None and hasattr(tabs, "current_viewport"):
            vp = tabs.current_viewport()
            if vp is not None:
                return vp
        vp = getattr(self, "viewport", None)
        if vp is not None and hasattr(vp, "fit_view"):
            return vp
        return None

    def cmd_fit(self):
        vp = self.current_viewport()
        if vp is not None:
            vp.fit_view()

    def cmd_reset(self):
        vp = self.current_viewport()
        if vp is not None:
            vp.reset_view()

    def cmd_toggle_wire(self):
        vp = self.current_viewport()
        if vp is None:
            return
        self._wire = not getattr(self, "_wire", False)
        vp.set_representation("wireframe" if self._wire else "solid")

    def cmd_edges_only(self):
        vp = self.current_viewport()
        if vp is None:
            return
        if getattr(vp, "_rep_mode", "") == "edges":
            vp.set_representation("solid")
        else:
            vp.set_representation("edges")

    def cmd_view(self, name):
        vp = self.current_viewport()
        if vp is not None and hasattr(vp, "set_view"):
            vp.set_view(name)

    def cmd_transparency(self):
        vp = self.current_viewport()
        if vp is not None and hasattr(vp, "toggle_transparency"):
            vp.toggle_transparency()

    def cmd_validate(self):
        if not self.sim:
            return self.msg("请先打开文件", "warn")
        v = self.sim.validate_class_versions()
        self.msg("classversions: %d classes, %d matched, %d/%d totals" % (
            v["expected_classes"], v["matched"], v["expected_total"], v["actual_total"]))

    def cmd_about(self):
        QMessageBox.about(self, "About",
                          "STAR-CCM+ .sim Viewer / Editor\nPyQt5 + VTK\n"
                          "查看 + 编辑几何/网格操作/场景/属性；求解运算禁用。\n"
                          "数据层: sim_parser.py · 回写: sim_writer.py\n"
                          "对标: star_gui_parity.md")


def cli_main(argv=None):
    """--cli 无窗口模式：解析 + 导出（复用 sim_parser 能力）。"""
    import argparse
    import json
    ap = argparse.ArgumentParser(description="STAR-CCM+ .sim 查看器（CLI 模式）")
    ap.add_argument("file", help=".sim 文件路径")
    ap.add_argument("--export-dir", default="", help="导出目录（数组/摘要/报告/STL）")
    ap.add_argument("--stl", default="", help="导出网格 STL 到指定路径")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    from sim_parser import SimFile
    sim = SimFile(args.file)
    print(sim.summary())
    if args.export_dir:
        os.makedirs(args.export_dir, exist_ok=True)
        sim.export(args.export_dir)
        with open(os.path.join(args.export_dir, "semantic_report.json"), "w",
                  encoding="utf-8") as f:
            json.dump(sim.semantic_report(), f, indent=1, default=str)
        print("已导出到 %s" % args.export_dir)
    if args.stl:
        sim.export_stl(args.stl)
        print("STL 已导出到 %s" % args.stl)
    return 0


def _apply_theme(app):
    """应用 STAR-CCM+ 浅色块状主题（Fusion 以便菜单栏吃到青绿 QSS）。"""
    try:
        app.setStyle("Fusion")
    except Exception:
        pass
    qss = os.path.join(os.path.dirname(os.path.abspath(__file__)), "star_gui_theme.qss")
    if os.path.exists(qss):
        with open(qss, encoding="utf-8") as f:
            app.setStyleSheet(f.read())


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="STAR-CCM+ .sim 项目查看器")
    ap.add_argument("file", nargs="?", help=".sim 文件路径")
    ap.add_argument("--cli", action="store_true", help="无窗口 CLI 模式")
    ap.add_argument("--export-dir", default="", help="--cli 导出目录")
    ap.add_argument("--stl", default="", help="--cli 导出 STL 路径")
    args = ap.parse_args(argv)

    if args.cli:
        return cli_main([args.file] +
                        (["--export-dir", args.export_dir] if args.export_dir else []) +
                        (["--stl", args.stl] if args.stl else []))

    _install_qt_message_filter()
    app = QApplication(sys.argv)
    _apply_theme(app)
    win = StarMainWindow()
    win.show()
    if args.file:
        win.load_file(args.file)
    return app.exec_()


def _install_qt_message_filter():
    """过滤 Windows 上无害的 EUDC 字体警告。"""
    try:
        from PyQt5.QtCore import qInstallMessageHandler, qt_message_handler
    except Exception:
        return

    def handler(mode, context, message):
        msg = str(message)
        if "EUDC" in msg and "font" in msg.lower():
            return
        qt_message_handler(mode, context, message)

    qInstallMessageHandler(handler)


if __name__ == "__main__":
    sys.exit(main())

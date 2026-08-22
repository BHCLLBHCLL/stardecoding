# -*- coding: utf-8 -*-
"""star_gui.py — STAR-CCM+ .sim 项目查看器（M0 骨架版）。

布局对齐 STAR-CCM+ 2025（Simulation tree / Graphics window / Properties /
Output / Status bar），技术路线对齐 cabdecoding（PyQt5 + VTK + numpy）。

用法:
    python star_gui.py                    # 空窗口
    python star_gui.py adjointWing_start.sim   # 打开 .sim

里程碑（star_gui_plan.md）:
    M0 骨架+打开+摘要  [本版本]   M1 树+属性   M2 3D 网格
    M3 场景/视图      M4 导出+CLI  M5 几何     M6 主题/测试
"""

import os
import sys
import threading

HEADLESS = os.environ.get("QT_QPA_PLATFORM", "").lower() in ("minimal", "offscreen")
"""无头模式：minimal/offscreen 下不创建 QVTK 视口（QVTK 在无头平台不稳定），
用占位控件代替；3D 渲染正确性由 star_gui_vtk.render_offscreen_png 的纯 VTK
离屏测试覆盖。"""

from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QAction, QApplication, QFileDialog, QMainWindow, QMessageBox, QSplitter,
    QTabWidget, QToolBar, QVBoxLayout, QWidget,
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


class StarMainWindow(QMainWindow):
    """主窗口（M0：骨架 + 打开 + 摘要；M1-M6 逐步补齐窗格）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("STAR-CCM+ .sim Viewer (star_gui)")
        self.resize(1280, 820)
        self.sim = None
        self.sim_path = None
        self._thread = None
        self._worker = None
        self._build_actions()
        self._build_ui()
        self._build_menus()
        self._build_toolbar()
        self.msg("STAR-CCM+ .sim Viewer 就绪 — File>Open 打开项目")

    # ---------------- UI ----------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(3, 3, 3, 3)

        self.split_main = QSplitter()          # 左(树/属性) | 右(图形+底部)
        self.split_left = QSplitter()          # 树 | 属性（M1 填充）
        self.split_left.setOrientation(1)      # 垂直

        # M1：仿真树 + 属性面板
        from star_gui_panes import PropertiesPanel, SimulationTree
        self.model = None
        self.tree_widget = SimulationTree(icons=icons())
        self.props_widget = PropertiesPanel()
        self.tree_widget.object_selected.connect(self.on_object_selected)
        self.tree_widget.check_changed.connect(self.on_part_visibility)
        self.props_widget.reference_activated.connect(self.on_object_selected)
        self.tree_pane = PaneFrame("Simulation Tree")
        self.tree_pane.set_body(self.tree_widget)
        self.props_pane = PaneFrame("Properties")
        self.props_pane.set_body(self.props_widget)
        self.split_left.addWidget(self.tree_pane)
        self.split_left.addWidget(self.props_pane)
        self.split_left.setSizes([420, 260])

        right = QSplitter()
        right.setOrientation(1)                # 垂直：图形区 / 底部输出
        self.summary_pane = SummaryPane()
        from star_gui_panes import GraphicsTabs, Star3DViewport
        self.graphics_tabs = GraphicsTabs()
        self.graphics_tabs.add_info_tab(self.summary_pane)
        if HEADLESS:
            from PyQt5.QtCore import Qt
            from PyQt5.QtWidgets import QLabel
            self.viewport = QLabel("3D 视口（无头模式禁用 QVTK）")
            self.viewport.setAlignment(Qt.AlignCenter)
            self.graphics_tabs.add_mesh_tab("3D Mesh", self.viewport)
        else:
            self.viewport = Star3DViewport()
            self.graphics_tabs.add_mesh_tab("3D Mesh", self.viewport)
        self.graphics_pane = PaneFrame("Graphics Window")
        self.graphics_pane.set_body(self.graphics_tabs)
        right.addWidget(self.graphics_pane)

        bottom = QTabWidget()
        self.messages = MessageWindow()
        self.progress = ProgressPanel()
        bottom.addTab(self.messages, "Messages")
        bottom.addTab(self.progress, "Progress")
        right.addWidget(bottom)
        right.setSizes([560, 170])

        self.split_main.addWidget(self.split_left)
        self.split_main.addWidget(right)
        self.split_main.setSizes([380, 900])
        root.addWidget(self.split_main)

        self.status_helper = StatusBarHelper()
        self.statusBar().addWidget(self.status_helper, 1)
        self.set_status("就绪")

    def _build_actions(self):
        self.actions = {}
        self._add("File>Open", "Open...", self.open_file, "open", QKeySequence.Open)
        self._add("File>Close", "Close", self.close_sim, "file")
        self._add("File>Exit", "Exit", self.close, "file", QKeySequence.Quit)
        self._add("File>Export>STL", "Export STL...", self.cmd_export_stl, "mesh")
        self._add("File>Export>Summary", "Export Summary...", self.cmd_export_summary, "file")
        self._add("File>Export>Report", "Export Report (JSON)...", self.cmd_export_report, "report")
        for path, label in [
            ("File>Save", "Save"),
            ("File>Save As", "Save As..."),
            ("Edit>NYI", "Edit"),
            ("Mesh>NYI", "Mesh"),
            ("Plot>NYI", "Plot"),
        ]:
            self._add(path, label, lambda checked=False, p=path: self._nyi(p), "unknown")
        self._add("Tools>Fingerprint", "Version Fingerprint", self.cmd_fingerprint, "info")
        self._add("Tools>Check Length", "State Length Check", self.cmd_check_length, "info")
        self._add("Tools>Validate", "ClassVersions Validate", self.cmd_validate, "info")
        self._add("Scene>Fit", "Fit View", self.cmd_fit, "fit")
        self._add("Scene>Reset", "Reset View", self.cmd_reset, "reset")
        self._add("Scene>Wireframe", "Wireframe / Solid", self.cmd_toggle_wire, "mesh")
        self._add("Scene>Edges", "Edges Only", self.cmd_edges_only, "mesh")
        self._add("Help>About", "About", self.cmd_about, "info")

    def _add(self, key, label, slot, icon_key, shortcut=None):
        act = QAction(icons().get(icon_key), label, self)
        if shortcut:
            act.setShortcut(shortcut)
        act.triggered.connect(slot)
        act.setObjectName(key)
        self.actions[key] = act
        return act

    def _build_menus(self):
        mbar = self.menuBar()
        for title, keys in [
            ("&File", ["File>Open", "File>Close", None, "File>Exit",
                       None, "File>Save", "File>Save As",
                       None, "File>Export>STL", "File>Export>Summary"]),
            ("&Edit", ["Edit>NYI"]),
            ("&Mesh", ["Mesh>NYI"]),
            ("&Scene", ["Scene>Fit", "Scene>Reset", "Scene>Wireframe"]),
            ("&Plot", ["Plot>NYI"]),
            ("&Tools", ["Tools>Fingerprint", "Tools>Check Length", "Tools>Validate"]),
            ("&Help", ["Help>About"]),
        ]:
            menu = mbar.addMenu(title)
            for key in keys:
                if key is None:
                    menu.addSeparator()
                    continue
                act = self.actions.get(key)
                if act:
                    menu.addAction(act)

    def _build_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        for key in ("File>Open", "File>Close", None, "Tools>Fingerprint"):
            if key is None:
                tb.addSeparator()
                continue
            tb.addAction(self.actions[key])
        self.addToolBar(tb)

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
        self.setWindowTitle("STAR-CCM+ .sim Viewer — %s" % sim.path)
        self.msg("已加载 %s" % sim.path)
        self.on_file_loaded()   # M1+ 钩子

    def on_file_loaded(self):
        """文件加载后的后续构建（M1 建树/属性，M2 建 3D）。"""
        from star_gui_model import StarSceneModel
        self.model = StarSceneModel(self.sim)
        self.tree_widget.set_model(self.model)
        self.props_widget.set_model(self.model)
        self.msg("仿真树已构建：%d 个顶层节点" % self.tree_widget.tree.topLevelItemCount())
        self._build_3d()

    def _build_3d(self):
        """M2/M3：按场景构建 3D 标签页（场景→显示器颜色→视图相机）。

        无头模式下 viewport 是占位控件，仅更新消息，不做 QVTK 操作。
        """
        try:
            from star_gui_vtk import (apply_camera, build_mesh_actors,
                                      build_scene_actors, scene_background)
            self.progress.set_progress(10, "构建 3D 场景 ...")
            scenes = self.model.scenes() if self.model else []
            # 清掉默认 mesh 标签页
            while self.graphics_tabs.count() > 1:
                w = self.graphics_tabs.widget(self.graphics_tabs.count() - 1)
                self.graphics_tabs.removeTab(self.graphics_tabs.count() - 1)
                try:
                    w.close()
                except Exception:
                    pass
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
                        bg = scene_background(self.sim, scene_obj)
                        vp.renderer.SetBackground(*bg["solid"])
                        vp.set_actors(actors)
                        apply_camera(vp.renderer, cam)
                    self.graphics_tabs.add_mesh_tab(sc["name"] or "Scene", vp)
                self.graphics_tabs.setCurrentIndex(1)
                self.msg("3D 场景：%d 个（含显示器/视图相机）" % len(scenes))
            else:
                actors = build_mesh_actors(self.sim)
                if not HEADLESS:
                    self.viewport.set_actors(actors)
                self.msg("3D 网格：%d 个 Part" % len(actors))
            self.progress.done("3D 就绪")
        except Exception as exc:  # noqa: BLE001
            self.msg("3D 构建失败: %s" % exc, "warn")

    def on_object_selected(self, obj):
        self.props_widget.show_object(obj)
        if obj is not None:
            self.set_status("选中 %s %s (id %d, %s)" % (
                obj.class_name or "?", obj.name or "", obj.id, obj.layer))
        self._highlight_selection(obj)

    def _highlight_selection(self, obj):
        """选中 Region/Boundary → 高亮其所属 Part 的网格（边界→面片映射未做，
        按 Region→Parts 链接高亮；其余 Part 降透明度）。"""
        if obj is None or HEADLESS:
            return
        if obj.class_name not in ("star.common.Region", "star.common.Boundary"):
            return
        # Region → Parts 名称集合
        names = set()
        if obj.class_name == "star.common.Region":
            rep = {"regions": [{"parts": []}]}
            parts = self.sim.objmap.get(obj.dict.get("Parts")) if obj.dict.get("Parts") else None
            plist = parts.dict.get("Keys") if parts is not None else []
            for k in plist:
                p = self.sim.objmap.get(k)
                if p is not None and p.name:
                    names.add(p.name)
        else:  # Boundary：属于某 Region，高亮该 Region 的 Part
            parent = self.sim.objmap.get(obj.dict.get("Parent") or -1)
            if parent is not None and parent.class_name == "star.common.Region":
                parts = self.sim.objmap.get(parent.dict.get("Parts") or -1)
                for k in (parts.dict.get("Keys") or []) if parts else []:
                    p = self.sim.objmap.get(k)
                    if p is not None and p.name:
                        names.add(p.name)
        for key, name, pid, actor in self.viewport.actors:
            visible = name in names
            actor.SetVisibility(visible)
        self.viewport.render()
        self.msg("高亮 %d 个 Part（Region→Parts 链接；边界→面片映射未做）" % len(names))

    def on_part_visibility(self, obj_id, checked):
        """树勾选 Part → 显隐对应 3D actor。"""
        if obj_id is None or HEADLESS:
            return
        obj = self.sim.objmap.get(obj_id)
        if obj is None or not obj.name:
            return
        for key, name, pid, actor in self.viewport.actors:
            if name == obj.name:
                actor.SetVisibility(checked)
        self.viewport.render()

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
        self.setWindowTitle("STAR-CCM+ .sim Viewer")

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

    def cmd_fit(self):
        if self.viewport:
            self.viewport.fit_view()

    def cmd_reset(self):
        if self.viewport:
            self.viewport.reset_view()

    def cmd_toggle_wire(self):
        self._wire = not getattr(self, "_wire", False)
        if self.viewport:
            self.viewport.set_representation("wireframe" if self._wire else "solid")

    def cmd_edges_only(self):
        """边线模式：隐藏表面 actor，显示边线 actor。"""
        if HEADLESS:
            return
        self._edges_only = not getattr(self, "_edges_only", False)
        for key, name, pid, actor in self.viewport.actors:
            is_edges = key.startswith("edges:")
            actor.SetVisibility(is_edges if self._edges_only else True)
        self.viewport.render()

    def cmd_validate(self):
        if not self.sim:
            return self.msg("请先打开文件", "warn")
        v = self.sim.validate_class_versions()
        self.msg("classversions: %d classes, %d matched, %d/%d totals" % (
            v["expected_classes"], v["matched"], v["expected_total"], v["actual_total"]))

    def cmd_about(self):
        QMessageBox.about(self, "About",
                          "STAR-CCM+ .sim Viewer\nPyQt5 + VTK\n"
                          "数据层: sim_parser.py（21 文件语料验证）")


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

    app = QApplication(sys.argv)
    win = StarMainWindow()
    win.show()
    if args.file:
        win.load_file(args.file)
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

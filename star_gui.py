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

        # M0 占位：树/属性窗格（M1 替换为真实组件）
        self.tree_pane = PaneFrame("Simulation Tree（M1）")
        self.props_pane = PaneFrame("Properties（M1）")
        self.split_left.addWidget(self.tree_pane)
        self.split_left.addWidget(self.props_pane)
        self.split_left.setSizes([420, 260])

        right = QSplitter()
        right.setOrientation(1)                # 垂直：图形区 / 底部输出
        self.summary_pane = SummaryPane()
        self.graphics_pane = PaneFrame("Graphics Window（M2 起提供 3D 场景）")
        self.graphics_pane.set_body(self.summary_pane)
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
        for path, label in [
            ("File>Export>STL", "Export STL..."),
            ("File>Export>Summary", "Export Summary..."),
            ("File>Save", "Save"),
            ("File>Save As", "Save As..."),
            ("Edit>NYI", "Edit"),
            ("Mesh>NYI", "Mesh"),
            ("Scene>NYI", "Scene"),
            ("Plot>NYI", "Plot"),
        ]:
            self._add(path, label, lambda checked=False, p=path: self._nyi(p), "unknown")
        self._add("Tools>Fingerprint", "Version Fingerprint", self.cmd_fingerprint, "info")
        self._add("Tools>Check Length", "State Length Check", self.cmd_check_length, "info")
        self._add("Tools>Validate", "ClassVersions Validate", self.cmd_validate, "info")
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
            ("&Scene", ["Scene>NYI"]),
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
        """子类/后续里程碑扩展点（M1 建树、M2 建 3D）。"""
        pass

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


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="STAR-CCM+ .sim 项目查看器")
    ap.add_argument("file", nargs="?", help=".sim 文件路径")
    args = ap.parse_args(argv)

    app = QApplication(sys.argv)
    win = StarMainWindow()
    win.show()
    if args.file:
        win.load_file(args.file)
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""S 波 S5：Mesh>清除 与 Mesh>转 2D（会话/显示层，诚实标注边界）。

覆盖：
  清除：丢弃会话生成结果（volume/poly/trimmer）+ 标记 session_meshes_cleared + 提示不改文件
  转 2D：无 3D 视图（无头）或视图无 actor 时如实拒绝；有视图时沿最小跨度轴压平并置 display_2d
  注册：两个动作已从 _kernel_nyi 桩改为真实命令
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest  # noqa: E402

SIM = os.path.join(ROOT, "adjointWing_start.sim")


@pytest.fixture(scope="module")
def app():
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _window(app):
    from star_gui import StarMainWindow
    win = StarMainWindow()
    win.show()
    win.load_file(SIM)
    t0 = time.time()
    while win.sim is None and time.time() - t0 < 30:
        app.processEvents()
        time.sleep(0.02)
    assert win.sim is not None
    return win


def test_mesh_actions_are_real_commands():
    import star_gui
    src = open(os.path.join(ROOT, "star_gui.py"), encoding="utf-8").read()
    assert "self.cmd_clear_mesh" in src and "self.cmd_convert_2d" in src
    assert "self._kernel_nyi(\"清除已生成网格\")" not in src
    assert "self._kernel_nyi(\"转换为 2D\")" not in src
    assert hasattr(star_gui.StarMainWindow, "cmd_clear_mesh")
    assert hasattr(star_gui.StarMainWindow, "cmd_convert_2d")


def test_clear_mesh_drops_session_results_and_marks_state(app):
    win = _window(app)
    try:
        win._volume_mesh_result = {"points": [[0, 0, 0]], "name": "x"}
        win._poly_mesh_result = {"points": [[0, 0, 0]]}
        win.cmd_clear_mesh()
        assert win._volume_mesh_result is None and win._poly_mesh_result is None
        assert getattr(win.document, "session_meshes_cleared", False) is True
        text = win.messages.view.toPlainText()
        assert "已清除本会话网格" in text and "对象图与 .sim 未改动" in text
    finally:
        win.document.mark_clean()
        win.close()


def test_convert_2d_is_honest_without_view(app):
    win = _window(app)
    try:
        win.cmd_convert_2d()
        text = win.messages.view.toPlainText()
        assert ("未执行" in text) or ("已转 2D（显示层）" in text)
        if "未执行" in text:
            assert not getattr(win.document, "display_2d", False)
    finally:
        win.document.mark_clean()
        win.close()


def test_convert_2d_flattens_when_view_available(app):
    win = _window(app)
    try:
        vp = getattr(win, "viewport", None)
        if vp is None or not hasattr(vp, "renderer") or not getattr(vp, "actors", None):
            pytest.skip("无头/无 actor：走诚实拒绝路径（另一用例覆盖）")
        win.cmd_convert_2d()
        assert getattr(win.document, "display_2d", False) is True
        text = win.messages.view.toPlainText()
        assert "已转 2D（显示层）" in text and "维度未改" in text
    finally:
        win.document.mark_clean()
        win.close()

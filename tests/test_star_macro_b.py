# -*- coding: utf-8 -*-
"""P10 B 路线 star_macro 扩展：Solve/Initialize/Step 宏模板 + 运行日志逐行回流。

用 FakeProc 模拟子进程 stdout，避免依赖真实 STAR-CCM+ 可执行文件。
临时目录按仓库约定用 tempfile.mkdtemp（不用 pytest tmp_path fixture）。
"""
import os
import shutil
import tempfile

import pytest

import star_macro as sm


# -- 宏写入 ---------------------------------------------------------------
def test_write_macro_creates_file_and_content():
    tmp = tempfile.mkdtemp(prefix="star_p10_")
    try:
        path = sm.write_macro(tmp, "solve_copy.java", sm.SOLVE_MACRO_JAVA)
        assert os.path.isfile(path)
        assert os.path.basename(path) == "solve_copy.java"
        text = open(path, encoding="ascii").read()
        assert "class solve_copy" in text
        assert "solver.run()" in text
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_write_solve_initialize_step_macros():
    tmp = tempfile.mkdtemp(prefix="star_p10_")
    try:
        pairs = [
            (sm.write_solve_macro(tmp), "solve_copy.java", "class solve_copy"),
            (sm.write_initialize_macro(tmp), "initialize_copy.java", "class initialize_copy"),
            (sm.write_step_macro(tmp), "step_copy.java", "class step_copy"),
        ]
        for path, name, tag in pairs:
            assert os.path.basename(path) == name
            assert tag in open(path, encoding="ascii").read()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_write_generate_mesh_macro_backward_compat():
    tmp = tempfile.mkdtemp(prefix="star_p10_")
    try:
        path = sm.write_generate_mesh_macro(tmp)
        assert os.path.basename(path) == "generate_mesh_copy.java"
        text = open(path, encoding="ascii").read()
        assert "class generate_mesh_copy" in text
        assert "generateVolumeMesh" in text
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# -- 流式回流 --------------------------------------------------------------
class FakeStream:
    def __init__(self, lines):
        self._lines = list(lines)

    def __iter__(self):
        return iter(self._lines)


class FakeProc:
    def __init__(self, lines, code=0):
        self.stdout = FakeStream(lines)
        self._code = code

    def poll(self):
        return self._code

    def wait(self):
        return self._code

    def kill(self):
        self._code = -9


def _make_inputs(tmp):
    src = os.path.join(tmp, "src.sim")
    with open(src, "w", encoding="ascii") as fh:
        fh.write("SIMCOOKIE")
    macro = sm.write_solve_macro(tmp)
    exe = os.path.join(tmp, "fake_star.exe")
    open(exe, "w", encoding="ascii").close()
    return src, macro, exe


def test_run_star_macro_stream_replays_lines(monkeypatch):
    lines = ["[0] init", "residual 1 0.5", "[1] solve_copy: done"]
    monkeypatch.setattr(sm.subprocess, "Popen", lambda cmd, **kw: FakeProc(lines))
    tmp = tempfile.mkdtemp(prefix="star_p10_")
    src, macro, exe = _make_inputs(tmp)
    seen = []
    try:
        work, code, got = sm.run_star_macro_stream(
            exe, src, macro, on_line=seen.append, timeout=3)
        assert code == 0
        assert got == lines
        assert seen == lines
        assert os.path.isdir(work)
        shutil.rmtree(work, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_star_macro_on_copy_backward_compat(monkeypatch):
    lines = ["line a", "line b"]
    monkeypatch.setattr(sm.subprocess, "Popen", lambda cmd, **kw: FakeProc(lines))
    tmp = tempfile.mkdtemp(prefix="star_p10_")
    src, macro, exe = _make_inputs(tmp)
    try:
        work, code, log = sm.run_star_macro_on_copy(exe, src, macro, timeout=3)
        assert code == 0
        assert log == "line a\nline b"
        assert os.path.isdir(work)
        shutil.rmtree(work, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_star_macro_stream_missing_exe():
    tmp = tempfile.mkdtemp(prefix="star_p10_")
    src, macro, _exe = _make_inputs(tmp)
    try:
        with pytest.raises(IOError):
            sm.run_star_macro_stream("no_such_star.exe", src, macro)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_star_macro_stream_timeout(monkeypatch):
    class BlockingStream:
        def __iter__(self):
            return self._gen()

        def _gen(self):
            import time
            while True:
                time.sleep(0.05)

    class HangingProc(FakeProc):
        def __init__(self):
            self.stdout = BlockingStream()
            self._code = None

        def poll(self):
            return None

        def wait(self):
            return None

    monkeypatch.setattr(sm.subprocess, "Popen", lambda cmd, **kw: HangingProc())
    tmp = tempfile.mkdtemp(prefix="star_p10_")
    src, macro, exe = _make_inputs(tmp)
    try:
        with pytest.raises(TimeoutError):
            sm.run_star_macro_stream(exe, src, macro, timeout=0.1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

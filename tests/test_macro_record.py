# -*- coding: utf-8 -*-
"""A1：Java 宏录制器 —— 命令→Java 映射、CommandBus 钩子录制、渲染/回放。

回放用 FakeProc 模拟子进程 stdout（与 P10 同法），临时目录按仓库约定用
tempfile.mkdtemp（不用 pytest tmp_path fixture）。
"""
import os
import shutil
import tempfile

import pytest

import macro_record as mr
import star_macro as sm
from sim_parser import SimFile
from star_gui_commands import (CopyObjectCommand, DeleteObjectCommand,
                               RenameCommand, SetPropertyCommand,
                               ShowOnlyCommand, TransformPartCommand,
                               VisibilityCommand)
from star_gui_document import SimDocument

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM = os.path.join(ROOT, "adjointWing_start.sim")


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


def _region(sim):
    return next(o for o in sim.objects
                if o.class_name == "star.common.Region" and o.name)


def _cmdop(name, cls, **args):
    """构造 command 类 MacroOp（`class` 是关键字，须以字典注入）。"""
    d = dict(args)
    d["class"] = cls
    return mr.MacroOp("command", name, **d)


# -- 命令 -> Java 映射表 ---------------------------------------------------
def test_command_names_and_modes():
    names = mr.command_names()
    assert len(names) == 7
    for n in ("RenameCommand", "SetPropertyCommand", "VisibilityCommand"):
        assert mr.command_mode(n) == "verified"
    for n in ("ShowOnlyCommand", "TransformPartCommand",
              "DeleteObjectCommand", "CopyObjectCommand"):
        assert mr.command_mode(n) == "best-effort"
    assert mr.command_mode("NotACommand") is None
    assert mr.command_mode(RenameCommand(1, "x")) == "verified"


def test_op_from_command_uses_doc_object():
    sim = SimFile(SIM)
    doc = SimDocument(sim, SIM)
    region = _region(sim)
    cmd = RenameCommand(region.id, "New Domain", region.name)
    op = mr.op_from_command(cmd, doc)
    assert op.kind == "command"
    assert op.name == "rename"
    assert op.args["obj_id"] == region.id
    assert op.args["class"] == "star.common.Region"
    assert op.args["target_name"] == region.name
    assert op.args["value"] == "New Domain"


def test_op_from_command_unregistered_returns_none():
    class OtherCommand(object):
        def __init__(self):
            self.obj_id = 1

    assert mr.op_from_command(OtherCommand()) is None


def test_command_java_verified_rename_and_set():
    op = _cmdop("rename", "star.common.Region", obj_id=1,
                target_name="Fluid Domain", key="PresentationName",
                value="Wing Domain")
    imports, lines = mr.command_java(op, 0)
    assert "import star.common.Region;" in imports
    assert 'Region obj0 = sim.getRegionManager().getRegion("Fluid Domain");' in lines
    assert 'obj0.setPresentationName("Wing Domain");' in lines

    op2 = _cmdop("set_property", "star.common.Region", obj_id=1,
                 target_name="Fluid Domain", key="SomeFlag", value=7)
    _imports, lines2 = mr.command_java(op2, 3)
    assert 'obj3.setSomeFlag(7);' in lines2


def test_command_java_visibility_true_false():
    op = _cmdop("set_visibility", "star.vis.Scene", obj_id=1,
                target_name="Mesh Scene 1", visible=True)
    _imports, lines = mr.command_java(op, 0)
    assert 'Scene obj0 = sim.getSceneManager().getScene("Mesh Scene 1");' in lines
    assert "obj0.setVisible(true);" in lines

    op2 = _cmdop("set_visibility", "star.vis.Scene", obj_id=1,
                 target_name="Mesh Scene 1", visible=False)
    _imports, lines2 = mr.command_java(op2, 0)
    assert "obj0.setVisible(false);" in lines2


def test_command_java_best_effort_and_unmapped():
    tr = _cmdop("transform", "star.common.Part", obj_id=1,
                target_name="Small Block", translate=(1, 0, 0), scale=(1, 1, 1))
    _imp, lines = mr.command_java(tr, 0)
    assert lines[0].startswith("// [best-effort] transform")

    for nm in ("delete", "copy", "show_only"):
        op = _cmdop(nm, "star.common.Region", obj_id=1, target_name="Fluid Domain")
        _imp, ls = mr.command_java(op, 0)
        assert ls[0].startswith("// [best-effort] %s" % nm)

    empty = _cmdop("rename", "star.common.Region", obj_id=1,
                   target_name="", value="X")
    _imp, ls2 = mr.command_java(empty, 0)
    assert "[unmapped command]" in ls2[0]


# -- CommandBus 钩子录制 ---------------------------------------------------
def test_recorder_attach_bus_records_and_toggle():
    sim = SimFile(SIM)
    doc = SimDocument(sim, SIM)
    rec = mr.make_recorder(doc.bus, class_name="rec_macro")
    assert rec.recording is True
    assert doc.bus.on_execute == rec._on_execute

    region = _region(sim)
    doc.execute(RenameCommand(region.id, "Renamed", region.name))
    assert len(rec) == 1

    rec.stop()
    doc.execute(SetPropertyCommand(region.id, "PresentationName", "Again", "Renamed"))
    assert len(rec) == 1

    rec.start()
    doc.execute(SetPropertyCommand(region.id, "PresentationName", "Third", "Again"))
    assert len(rec) == 2

    rec.detach()
    assert doc.bus.on_execute is None


def test_recorder_record_command_and_operation():
    sim = SimFile(SIM)
    doc = SimDocument(sim, SIM)
    rec = mr.MacroRecorder()
    region = _region(sim)
    op = rec.record_command(VisibilityCommand(region.id, False), doc)
    assert op is not None and op.name == "set_visibility"
    assert rec.operations() == []

    rec.record_operation("generate_surface_mesh")
    rec.record_operation("initialize")
    rec.record_operation("run")
    rec.record_operation("export_scene_hardcopy", scene="Mesh Scene 1",
                         file="s.png")
    assert rec.operations() == ["generate_surface_mesh", "initialize", "run",
                                "export_scene_hardcopy"]
    assert rec.operations("solver") == ["initialize", "run"]
    assert rec.categories() == {"meshing", "solver", "post"}


# -- 渲染 / 落盘 -----------------------------------------------------------
def test_recorder_to_java_renders_public_api():
    rec = mr.MacroRecorder(class_name="rec_op", save_name="out2.sim")
    rec.record_operation("generate_surface_mesh")
    rec.record_operation("enable_model", continuum="Physics 1",
                         model_class="star.flow.SegregatedFlowModel")
    rec.record_operation("run")
    java = rec.to_java()
    assert "public class rec_op extends StarMacro" in java
    assert "Simulation sim = getActiveSimulation();" in java
    assert "mpc.generateSurfaceMesh();" in java
    assert "import star.flow.SegregatedFlowModel;" in java
    assert "pc.enable(star.flow.SegregatedFlowModel.class);" in java
    assert "solver.run();" in java
    assert 'sim.saveState(resolvePath("out2.sim"));' in java


def test_recorder_to_java_no_save_and_unmapped():
    rec = mr.MacroRecorder(do_save=False)
    rec.record_operation("run")
    assert "saveState" not in rec.to_java()

    rec2 = mr.MacroRecorder()
    rec2.record_operation("nope_op")
    assert "[unmapped operation] nope_op" in rec2.to_java()

    rec3 = mr.MacroRecorder()
    assert "(no recorded operations)" in rec3.to_java()


def test_recorder_save_writes_ascii_java():
    tmp = tempfile.mkdtemp(prefix="star_rec_")
    try:
        rec = mr.MacroRecorder(class_name="saved_macro", save_name="out3.sim")
        rec.record_operation("run")
        path = rec.save(tmp)
        assert os.path.isfile(path)
        assert os.path.basename(path) == "saved_macro.java"
        text = open(path, encoding="ascii").read()
        assert "class saved_macro" in text
        assert "solver.run();" in text
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# -- 回放（复用宏桥） ------------------------------------------------------
def test_recorder_replay_delegates_to_macro_bridge(monkeypatch):
    lines = ["[0] initialize", "solve_copy: done"]
    monkeypatch.setattr(sm.subprocess, "Popen", lambda cmd, **kw: FakeProc(lines))
    tmp = tempfile.mkdtemp(prefix="star_rec_")
    try:
        src = os.path.join(tmp, "src.sim")
        with open(src, "w", encoding="ascii") as fh:
            fh.write("SIMCOOKIE")
        exe = os.path.join(tmp, "star.exe")
        open(exe, "w", encoding="ascii").close()

        rec = mr.MacroRecorder()
        rec.record_operation("run")
        work, code, got = rec.replay(exe, src, timeout=3)
        try:
            assert code == 0
            assert got == lines
            assert os.path.isdir(work)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

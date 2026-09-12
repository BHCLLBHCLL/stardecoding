# -*- coding: utf-8 -*-
"""X1 纯逻辑：备份(~) / AutoSave(@N) 命名轮转 / CHECKPOINT 触发 / 模板 .simt。"""
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from star_gui_session import (AutoSavePolicy, autosave_path, backup_path,
                              consume_checkpoint, checkpoint_triggered,
                              is_template, list_autosaves, make_backup,
                              next_autosave_index, rotate_autosaves,
                              template_path)


def test_backup_path_and_make_backup():
    tmp = tempfile.mkdtemp(prefix="star_session_")
    try:
        src = os.path.join(tmp, "intake.sim")
        assert make_backup(src) is None              # 目标不存在 → 无备份可做
        with open(src, "wb") as f:
            f.write(b"payload")
        assert backup_path(src) == src + "~"
        bp = make_backup(src)
        assert bp == src + "~"
        with open(bp, "rb") as f:
            assert f.read() == b"payload"            # 备份保留覆盖前内容
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_autosave_naming_listing_rotation():
    tmp = tempfile.mkdtemp(prefix="star_session_")
    try:
        base = os.path.join(tmp, "intake.sim")
        assert autosave_path(base, 1) == os.path.join(tmp, "intake@1.sim")
        assert list_autosaves(base) == []
        assert next_autosave_index(base) == 1
        for n in (1, 2, 3):
            with open(autosave_path(base, n), "wb") as f:
                f.write(b"x")
        assert [n for n, _p in list_autosaves(base)] == [1, 2, 3]
        assert next_autosave_index(base) == 4
        removed = rotate_autosaves(base, keep=2)
        assert [os.path.basename(p) for p in removed] == ["intake@1.sim"]
        assert [n for n, _p in list_autosaves(base)] == [2, 3]
        assert rotate_autosaves(base, keep=0)         # keep=0 → 全删
        assert list_autosaves(base) == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_template_extension():
    assert template_path("intake.sim") == "intake.simt"
    assert template_path(os.path.join("a", "b.sim")) == os.path.join("a", "b.simt")
    assert is_template("x.SIMT") and is_template("x.simt")
    assert not is_template("x.sim") and not is_template(None) and not is_template("")


def test_checkpoint_trigger_file():
    tmp = tempfile.mkdtemp(prefix="star_session_")
    try:
        trig = os.path.join(tmp, "stop.trigger")
        assert not checkpoint_triggered(trig)
        assert not consume_checkpoint(trig)           # 不存在 → False
        assert not consume_checkpoint("")             # 空路径安全
        with open(trig, "w") as f:
            f.write("1")
        assert checkpoint_triggered(trig)
        assert consume_checkpoint(trig) is True       # 存在 → 消费并删除
        assert not os.path.exists(trig)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_autosave_policy_snapshot_and_roundtrip():
    tmp = tempfile.mkdtemp(prefix="star_session_")
    try:
        wrote = []

        def write(dest):
            wrote.append(dest)
            with open(dest, "wb") as f:
                f.write(b"1")

        p = AutoSavePolicy(enabled=True, interval_sec=60, keep=1)
        base = os.path.join(tmp, "run.sim")
        d1 = p.snapshot(base, write)
        p.snapshot(base, write)
        assert [os.path.basename(x) for x in wrote] == ["run@1.sim", "run@2.sim"]
        assert os.path.basename(d1) == "run@1.sim"
        assert [n for n, _x in list_autosaves(base)] == [2]   # keep=1 仅留最近
        assert p.snapshot("", write) is None                  # 无路径安全
        assert AutoSavePolicy.from_dict(p.to_dict()).to_dict() == p.to_dict()
        assert AutoSavePolicy(interval_sec=0, keep=-5).interval_sec == 1
        assert AutoSavePolicy(interval_sec=0, keep=-5).keep == 0
        d = AutoSavePolicy.from_dict(None)
        assert d.enabled is False and d.trigger == ""
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

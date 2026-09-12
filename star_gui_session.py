# -*- coding: utf-8 -*-
"""star_gui_session.py — 会话生命周期：Save All / 备份(~) / AutoSave(@N) / CHECKPOINT / 模板 .simt。

对齐 STAR-CCM+ 20.02 的会话行为：
- 覆盖写盘前把既有目标文件备份为 `path~`（就地备份，只留一份）；
- AutoSave 以 `<base>@N.sim` 递增快照命名（如 `intake@1.sim`），可设保留份数轮转；
- CHECKPOINT 触发文件：监视路径上出现触发文件即执行一次断点保存并消费该文件；
- 模板以 `.simt` 扩展名承载 .sim 内容（新建时另存为 .sim）。

本模块为纯逻辑（无 Qt），便于无头单测；GUI 侧只做定时器与菜单接线。
"""

import os
import shutil

BACKUP_SUFFIX = "~"
TEMPLATE_EXT = ".simt"
AUTOSAVE_SEP = "@"
DEFAULT_INTERVAL_SEC = 300
DEFAULT_KEEP = 3


def backup_path(path):
    """就地备份路径：`intake.sim` → `intake.sim~`。"""
    return path + BACKUP_SUFFIX


def make_backup(path):
    """覆盖写前把既有文件备份为 `path~`；文件不存在返回 None（无备份可做）。"""
    if not path or not os.path.exists(path):
        return None
    dest = backup_path(path)
    try:
        shutil.copy2(path, dest)
    except OSError:
        return None
    return dest


def autosave_path(path, index):
    """AutoSave 快照路径：`intake.sim` + 1 → `intake@1.sim`。"""
    base, ext = os.path.splitext(path)
    return "%s%s%d%s" % (base, AUTOSAVE_SEP, int(index), ext or ".sim")


def list_autosaves(path):
    """同目录下 `base@N.sim` 快照，按 N 升序返回 [(n, fullpath), ...]。"""
    base, ext = os.path.splitext(path)
    ext = ext or ".sim"
    folder = os.path.dirname(os.path.abspath(path))
    prefix = os.path.basename(base) + AUTOSAVE_SEP
    out = []
    if not os.path.isdir(folder):
        return out
    for name in os.listdir(folder):
        if not name.startswith(prefix) or not name.endswith(ext):
            continue
        mid = name[len(prefix):len(name) - len(ext)]
        if mid.isdigit():
            out.append((int(mid), os.path.join(folder, name)))
    out.sort()
    return out


def next_autosave_index(path):
    """下一个可用快照序号（无既有则 1）。"""
    items = list_autosaves(path)
    return (items[-1][0] + 1) if items else 1


def rotate_autosaves(path, keep=DEFAULT_KEEP):
    """保留最近 keep 个快照，删除更早的，返回被删除路径列表。"""
    items = list_autosaves(path)
    drop = items if keep <= 0 else items[:-keep]
    removed = []
    for _n, full in drop:
        try:
            os.remove(full)
            removed.append(full)
        except OSError:
            pass
    return removed


def template_path(path):
    """模板路径：`intake.sim` → `intake.simt`。"""
    return os.path.splitext(path)[0] + TEMPLATE_EXT


def is_template(path):
    return os.path.splitext(path or "")[1].lower() == TEMPLATE_EXT


def checkpoint_triggered(trigger_path):
    """触发文件存在即视为收到断点请求。"""
    return bool(trigger_path) and os.path.exists(trigger_path)


def consume_checkpoint(trigger_path):
    """检测到触发文件则删除并返回 True（一次性断点触发）。"""
    if not checkpoint_triggered(trigger_path):
        return False
    try:
        os.remove(trigger_path)
    except OSError:
        pass
    return True


class AutoSavePolicy(object):
    """AutoSave / CHECKPOINT 策略：启用状态 / 间隔 / 保留份数 / 触发文件路径。"""

    def __init__(self, enabled=False, interval_sec=DEFAULT_INTERVAL_SEC,
                 keep=DEFAULT_KEEP, trigger=""):
        self.enabled = bool(enabled)
        self.interval_sec = max(1, int(interval_sec))
        self.keep = max(0, int(keep))
        self.trigger = trigger or ""

    def to_dict(self):
        return {"enabled": self.enabled, "interval_sec": self.interval_sec,
                "keep": self.keep, "trigger": self.trigger}

    @classmethod
    def from_dict(cls, data):
        data = data or {}
        return cls(enabled=data.get("enabled", False),
                   interval_sec=data.get("interval_sec", DEFAULT_INTERVAL_SEC),
                   keep=data.get("keep", DEFAULT_KEEP),
                   trigger=data.get("trigger", ""))

    def snapshot(self, sim_path, write_fn):
        """写下一个 @N 快照并轮转；write_fn(dest) 负责实际写盘。返回快照路径。"""
        if not sim_path:
            return None
        dest = autosave_path(sim_path, next_autosave_index(sim_path))
        write_fn(dest)
        rotate_autosaves(sim_path, self.keep)
        return dest

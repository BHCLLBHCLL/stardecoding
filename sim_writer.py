# -*- coding: utf-8 -*-
"""把会话补丁写回 .sim：只改对象图 repr 行，不重写数组/状态表。"""

import os
import shutil


def format_repr(value):
    """写出 parse_repr 能读的 Python 字面量。"""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(int(value))
    if isinstance(value, float):
        return repr(float(value))
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(format_repr(x) for x in value) + "]"
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            parts.append("%s: %s" % (format_repr(k), format_repr(v)))
        return "{" + ", ".join(parts) + "}"
    return repr(value)


def object_line(obj):
    return format_repr(obj.dict)


def _line_span(blob, start):
    """对象行 [start, end) 含换行。"""
    nl = blob.find(b"\n", start)
    if nl < 0:
        return start, len(blob)
    return start, nl + 1


def apply_patches_to_blob(blob, sim, patches, created=None, encoding="latin-1"):
    """按对象 .line 偏移替换 repr 行。从后往前改，避免偏移漂移。"""
    jobs = []
    for oid, bag in (patches or {}).items():
        obj = sim.objmap.get(oid)
        if obj is None or not bag:
            continue
        if getattr(obj, "line", -1) < 0:
            continue
        start, end = _line_span(blob, int(obj.line))
        text = object_line(obj)
        raw = (text + "\n").encode(encoding, errors="replace")
        jobs.append((start, end, raw))
    jobs.sort(key=lambda j: j[0], reverse=True)
    out = bytearray(blob)
    for start, end, raw in jobs:
        out[start:end] = raw
    return bytes(out)


def save_sim(sim, dest_path, patches=None, created=None, src_path=None):
    """复制原文件并替换已改对象行。新对象若无 file offset 则不写入（会话级）。"""
    src = src_path or sim.path
    if not src or not os.path.isfile(src):
        raise IOError("没有可复制的源 .sim")
    dest_path = os.path.abspath(dest_path)
    src = os.path.abspath(src)
    if dest_path != src:
        shutil.copy2(src, dest_path)
    with open(dest_path, "rb") as f:
        blob = f.read()
    new_blob = apply_patches_to_blob(blob, sim, patches or {}, created)
    with open(dest_path, "wb") as f:
        f.write(new_blob)
    return dest_path

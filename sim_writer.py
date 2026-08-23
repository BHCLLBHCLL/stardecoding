# -*- coding: utf-8 -*-
"""把会话补丁写回 .sim：替换已有对象图 repr 行，并可在 ClassVersions 前插入新对象。"""

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


def remap_value(value, mapping):
    """把会话临时 id 换成插入后的图序号 id。"""
    if not mapping:
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in mapping:
        return mapping[value]
    if isinstance(value, list):
        return [remap_value(x, mapping) for x in value]
    if isinstance(value, tuple):
        return [remap_value(x, mapping) for x in value]
    if isinstance(value, dict):
        return {k: remap_value(v, mapping) for k, v in value.items()}
    return value


def object_line(obj, mapping=None):
    d = remap_value(obj.dict, mapping) if mapping else obj.dict
    return format_repr(d)


def _line_span(blob, start):
    """对象行 [start, end) 含换行。"""
    nl = blob.find(b"\n", start)
    if nl < 0:
        return start, len(blob)
    return start, nl + 1


def _classversions(sim):
    for o in sim.objects:
        if o.class_name == "ClassVersions" and getattr(o, "line", -1) >= 0:
            return o
    return None


def created_id_mapping(sim, created):
    """新对象写入 ClassVersions 之前，占用其旧 id 起的连续号。"""
    pending = [oid for oid, obj in (created or {}).items()
               if obj is not None and getattr(obj, "line", -1) < 0]
    if not pending:
        return {}
    cv = _classversions(sim)
    if cv is None:
        return {}
    pending.sort()
    return {old: cv.id + i for i, old in enumerate(pending)}


def apply_patches_to_blob(blob, sim, patches, created=None, encoding="latin-1"):
    """替换已改对象行，再把 line<0 的新对象插到 ClassVersions 之前。"""
    mapping = created_id_mapping(sim, created)
    jobs = []
    for oid, bag in (patches or {}).items():
        obj = sim.objmap.get(oid)
        if obj is None or not bag:
            continue
        if getattr(obj, "line", -1) < 0:
            continue
        start, end = _line_span(blob, int(obj.line))
        text = object_line(obj, mapping)
        raw = (text + "\n").encode(encoding, errors="replace")
        jobs.append((start, end, raw))
    jobs.sort(key=lambda j: j[0], reverse=True)
    out = bytearray(blob)
    cv = _classversions(sim)
    insert_at = int(cv.line) if cv is not None else None
    for start, end, raw in jobs:
        out[start:end] = raw
        if insert_at is not None and start < insert_at:
            insert_at += len(raw) - (end - start)
    out = _insert_created(out, sim, created, mapping, encoding, insert_at)
    return bytes(out)


def _insert_created(blob, sim, created, mapping, encoding, insert_at=None):
    pending = [(oid, created[oid]) for oid in sorted(created or {})
               if created[oid] is not None and getattr(created[oid], "line", -1) < 0]
    if not pending:
        return blob
    cv = _classversions(sim)
    if cv is None:
        return blob
    if insert_at is None:
        insert_at = int(cv.line)
    chunks = []
    for oid, obj in pending:
        raw = (object_line(obj, mapping) + "\n").encode(encoding, errors="replace")
        chunks.append((oid, obj, raw))
    block = b"".join(c[2] for c in chunks)
    out = bytearray(blob)
    out[insert_at:insert_at] = block
    pos = insert_at
    for oid, obj, raw in chunks:
        obj.line = pos
        pos += len(raw)
    cv.line = insert_at + len(block)
    return out


def save_sim(sim, dest_path, patches=None, created=None, src_path=None):
    """复制原文件、替换补丁行，并插入新对象图行。"""
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

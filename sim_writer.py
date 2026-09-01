# -*- coding: utf-8 -*-
"""把会话补丁写回 .sim：对象行替换/插入 + 数组载荷覆盖/追加。"""

import os
import re
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
    """写出对象行；Imported* 改走数组块，不写进 repr。"""
    d = remap_value(obj.dict, mapping) if mapping else obj.dict
    if isinstance(d, dict) and (d.get("ImportedVertices") or d.get("ImportedFaces")):
        d = dict(d)
        d.pop("ImportedVertices", None)
        d.pop("ImportedFaces", None)
    return format_repr(d)


def _line_span(blob, start):
    nl = blob.find(b"\n", start)
    if nl < 0:
        return start, len(blob)
    return start, nl + 1


def _classversions(sim):
    for o in sim.objects:
        if o.class_name == "ClassVersions" and getattr(o, "line", -1) >= 0:
            return o
    return None


def created_id_mapping(sim, created, skip_ids=None):
    skip = set(skip_ids or [])
    pending = [oid for oid, obj in (created or {}).items()
               if oid not in skip
               and obj is not None and getattr(obj, "line", -1) < 0]
    if not pending:
        return {}
    cv = _classversions(sim)
    if cv is None:
        return {}
    pending.sort()
    return {old: cv.id + i for i, old in enumerate(pending)}


def encode_float8_payload(vertices):
    import numpy as np
    return np.asarray(vertices, dtype="<f8").reshape(-1).tobytes()


def encode_float8_block(vertices):
    import numpy as np
    arr = np.asarray(vertices, dtype="<f8").reshape(-1)
    hdr = "{'ClassName': 'Array', 'Type': 'Float8', 'nElements': %d, 'sizeof<T>': 8}\n" % (
        int(arr.size),)
    return hdr.encode("latin-1") + arr.tobytes()


def encode_unsigned4_block(faces):
    import numpy as np
    arr = np.asarray(faces, dtype="<u4").reshape(-1)
    hdr = "{'ClassName': 'Array', 'Type': 'Unsigned4', 'nElements': %d, 'sizeof<T>': 4}\n" % (
        int(arr.size),)
    return hdr.encode("latin-1") + arr.tobytes()


def collect_imported_array_blocks(sim, skip_ids=None):
    skip = set(skip_ids or [])
    blocks = []
    for o in sim.objects:
        if o.id in skip:
            continue
        verts = (o.dict or {}).get("ImportedVertices")
        faces = (o.dict or {}).get("ImportedFaces")
        if not verts or not faces:
            continue
        blocks.append(encode_float8_block(verts))
        blocks.append(encode_unsigned4_block(faces))
    return blocks


def apply_array_payload_patches(blob, sim, array_patches):
    if not array_patches:
        return blob
    out = bytearray(blob)
    for idx, data in array_patches.items():
        if idx is None or idx < 0 or idx >= len(sim.arrays):
            continue
        a = sim.arrays[idx]
        start = int(a["start"])
        old = a.get("data") or b""
        raw = data if isinstance(data, (bytes, bytearray)) else bytes(data)
        if len(raw) != len(old):
            continue
        out[start:start + len(old)] = raw
        a["data"] = bytes(raw)
    return bytes(out)


def _patch_state_position(blob, new_pos):
    nl = blob.find(b"\n")
    if nl < 0:
        return blob, 0
    head = blob[:nl].decode("latin-1")
    m = re.search(r"'StatePosition':\s*\d+L?", head)
    if not m:
        return blob, 0
    new = "'StatePosition': %dL" % int(new_pos)
    new_head = head[:m.start()] + new + head[m.end():]
    delta = len(new_head) - len(head)
    return new_head.encode("latin-1") + blob[nl:], delta


def insert_array_blocks(blob, sim, blocks):
    """先改头部 StatePosition（含位数变化），再在 StarVersion 前插入数组块。"""
    if not blocks:
        return blob, 0
    insert_at = int((sim.header or {}).get("StatePosition") or 0)
    if insert_at <= 0:
        return blob, 0
    payload = b"".join(blocks)
    new_pos = insert_at + len(payload)
    for _ in range(4):
        _trial, hdelta = _patch_state_position(blob, new_pos)
        expected = insert_at + hdelta + len(payload)
        if expected == new_pos:
            break
        new_pos = expected
    blob, hdelta = _patch_state_position(blob, new_pos)
    insert_at2 = insert_at + hdelta
    out = bytearray(blob)
    out[insert_at2:insert_at2] = payload
    shift = len(payload) + hdelta
    if sim.header is not None:
        sim.header["StatePosition"] = new_pos
    for o in sim.objects:
        if getattr(o, "line", -1) >= insert_at:
            o.line += shift
    for a in sim.arrays:
        st = int(a.get("start") or 0)
        if st <= 0:
            continue
        a["start"] = st + (shift if st >= insert_at else hdelta)
    return bytes(out), shift


def apply_patches_to_blob(blob, sim, patches, created=None, encoding="latin-1",
                          skip_ids=None):
    skip = set(skip_ids or [])
    mapping = created_id_mapping(sim, created, skip)
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
    out = _insert_created(out, sim, created, mapping, encoding, insert_at, skip)
    return bytes(out)


def _insert_created(blob, sim, created, mapping, encoding, insert_at=None,
                    skip_ids=None):
    skip = set(skip_ids or [])
    pending = [(oid, created[oid]) for oid in sorted(created or {})
               if oid not in skip
               and created[oid] is not None and getattr(created[oid], "line", -1) < 0]
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


def save_sim(sim, dest_path, patches=None, created=None, src_path=None,
             array_patches=None, new_arrays=None, deleted=None):
    """复制原文件 → 覆盖数组 → 追加导入数组 → 补丁对象行 → 插入新对象。"""
    src = src_path or sim.path
    if not src or not os.path.isfile(src):
        raise IOError("没有可复制的源 .sim")
    dest_path = os.path.abspath(dest_path)
    src = os.path.abspath(src)
    if dest_path != src:
        shutil.copy2(src, dest_path)
    with open(dest_path, "rb") as f:
        blob = f.read()
    skip = set(deleted or [])
    blob = apply_array_payload_patches(blob, sim, array_patches or {})
    blocks = list(new_arrays or [])
    if not blocks:
        blocks = collect_imported_array_blocks(sim, skip_ids=skip)
    blob, _shift = insert_array_blocks(blob, sim, blocks)
    blob = apply_patches_to_blob(blob, sim, patches or {}, created, skip_ids=skip)
    with open(dest_path, "wb") as f:
        f.write(blob)
    return dest_path


def compare_object_graph(path_a, path_b, keys=None):
    """本解析器差分：同 id 对象指定字段是否一致。"""
    from sim_parser import SimFile
    keys = keys or ("PresentationName", "name", "Opacity", "DisplayerColor",
                    "Mesh", "Keys", "ParallelScale")
    a, b = SimFile(path_a), SimFile(path_b)
    diffs = []
    ids = set(a.objmap) & set(b.objmap)
    for oid in sorted(ids):
        da, db = a.objmap[oid].dict, b.objmap[oid].dict
        for k in keys:
            if da.get(k) != db.get(k) and (k in da or k in db):
                diffs.append((oid, k, da.get(k), db.get(k)))
    return {"ok": not diffs, "diffs": diffs[:50], "n": len(diffs)}


def try_official_resave(src_sim, dest_dir):
    """有 starccmw 时用 resave_sim.java 重存；否则返回 skipped。"""
    exe = os.environ.get("STARCCM_HOME")
    java = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resave_sim.java")
    if not exe or not os.path.isfile(java):
        return {"status": "skipped", "reason": "no STARCCM_HOME or resave_sim.java"}
    cand = os.path.join(exe, "star", "bin", "starccmw.exe")
    if not os.path.isfile(cand):
        return {"status": "skipped", "reason": "starccmw.exe not found"}
    return {"status": "ready", "exe": cand, "macro": java}

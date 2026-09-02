# -*- coding: utf-8 -*-
"""把会话补丁写回 .sim：对象行替换/插入 + 数组载荷覆盖/追加 + 变长替换/删除。"""

import os
import re
import shutil
import bisect


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


def apply_array_ops(blob, sim, ops):
    """W1：已有数组块变长替换 / 删除 + 全量重定位。

    ops: list of dict
      {'op': 'replace', 'index': i, 'count': int, 'payload': bytes}
      {'op': 'delete',  'index': i}
    返回 (new_blob, summary)。只改动目标数组块（header 行 + 载荷）；
    其余字节/对象行逐字保留，重定位全部后续偏移，并重算头部 StatePosition
    （含位数变化）；同步原地更新 sim.arrays / 对象 line / header。
    主状态表数组（sim.state_text 所在）变长编辑拒绝（其自身链属 W2/G 波）。
    """
    sim_state = sim.state_text.encode("latin-1") if sim.state_text else None
    first_nl = blob.find(b"\n")
    head = blob[:first_nl]
    rest0 = first_nl + 1
    m = re.search(rb"'StatePosition':\s*(\d+)", head)
    orig_pos = int(m.group(1)) if m else 0
    # 记录每个 op 的原区块及其替换字节（绝对坐标）
    ev = []
    for op in ops:
        idx = int(op["index"])
        a = sim.arrays[idx]
        orig_data = a.get("data") or b""
        if sim_state is not None and bytes(orig_data) == sim_state:
            raise ValueError(
                "主状态表数组(%d)变长编辑超出 W1 范围（留 W2/G 波安全编辑）" % idx)
        abs_start = int(a["header_start"])
        abs_end = int(a["start"]) + len(orig_data)
        orig_hlen = int(a["start"]) - int(a["header_start"])
        if op["op"] == "delete":
            new_block = b""
            new_hlen = 0
        else:
            n = int(op["count"])
            hdr_text = blob[abs_start:int(a["start"])].decode("latin-1")
            new_hdr = re.sub(r"'nElements':\s*\d+",
                             "'nElements': %d" % n, hdr_text)
            new_block = new_hdr.encode("latin-1") + bytes(op.get("payload") or b"")
            new_hlen = len(new_hdr.encode("latin-1"))
        ev.append({"abs_start": abs_start, "abs_end": abs_end,
                   "new_block": new_block,
                   "delta": len(new_block) - (abs_end - abs_start),
                   "new_hlen": new_hlen, "op": op["op"], "index": idx,
                   "count": int(op["count"]) if op["op"] == "replace" else None})
    # 拼接（倒序应用，保持原偏移有效）
    out = bytearray(blob)
    for e in sorted(ev, key=lambda z: z["abs_start"], reverse=True):
        out[e["abs_start"]:e["abs_end"]] = e["new_block"]
    # 数组移位累积函数：shift(ref) = 起点 < ref 的编辑 delta 之和
    ev_sorted = sorted(ev, key=lambda z: z["abs_start"])
    _starts = [e["abs_start"] for e in ev_sorted]
    _cum = [0]
    for e in ev_sorted:
        _cum.append(_cum[-1] + e["delta"])

    def shift_at(ref):
        return _cum[bisect.bisect_left(_starts, ref)]

    deleted = {e["index"] for e in ev if e["op"] == "delete"}
    replaced = {e["index"] for e in ev if e["op"] == "replace"}
    new_hlen = {e["index"]: e["new_hlen"] for e in ev}
    orig_hlen = {a["index"]: (int(a["start"]) - int(a["header_start"]))
                 for a in sim.arrays}
    final_pos = orig_pos + shift_at(orig_pos)
    for _ in range(8):
        new_head = re.sub(
            rb"'StatePosition':\s*\d+", b"'StatePosition': %d" % final_pos, head)
        hdelta = len(new_head) - len(head)
        cand = orig_pos + shift_at(orig_pos) + hdelta
        if cand == final_pos:
            break
        final_pos = cand
    new_head = re.sub(
        rb"'StatePosition':\s*\d+", b"'StatePosition': %d" % final_pos, head)
    new_blob = new_head + b"\n" + bytes(out[rest0:])
    # ---- 原地一致化 sim ----
    new_head_len = len(new_head) + 1

    def final_off(x):
        if x >= rest0:
            return new_head_len + (x - rest0) + shift_at(x)
        return x

    keep = []
    for a in sim.arrays:
        idx = a["index"]
        if idx in deleted:
            continue
        nd = dict(a)
        nhs = final_off(int(a["header_start"]))
        nd["header_start"] = nhs
        nd["start"] = nhs + (new_hlen[idx] if idx in replaced else orig_hlen[idx])
        if idx in replaced:
            op = next(e for e in ev if e["index"] == idx)
            nd["count"] = int(op["count"])
            nd["data"] = bytes(op["new_block"])[new_hlen[idx]:]
            nd["dict"] = dict(a["dict"])
            nd["dict"]["nElements"] = int(op["count"])
        keep.append(nd)
    for nj, a in enumerate(keep):
        a["index"] = nj
    sim.arrays = keep
    if sim.header is not None:
        sim.header["StatePosition"] = final_pos
    for o in sim.objects:
        if getattr(o, "line", -1) >= rest0:
            o.line = final_off(int(o.line))
    summary = [{"index": e["index"], "op": e["op"], "delta": e["delta"],
                "abs_range": (e["abs_start"], e["abs_end"])} for e in ev_sorted]
    return new_blob, {"edits": summary, "old_state_position": orig_pos,
                      "new_state_position": final_pos}


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


def _line_text(blob, start):
    end = blob.find(b"\n", start)
    if end < 0:
        end = len(blob)
    return blob[start:end], start, end


def get_name_manager(sim):
    """W4：定位 NameManager 标记对象（原始文件为空标记 {'ClassName':'NameManager'}）。"""
    for o in sim.objects:
        if o.class_name == "NameManager" and getattr(o, "line", -1) >= 0:
            return o
    return None


def write_name_manager(blob, sim, object_id=None, names=None):
    """W4：NameManager 写入（保守）。

    - 无 NameManager 对象：原样返回（info['present']=False）。
    - 仅当变更**等宽**（改写既有 'ObjectId' 数值、或等宽覆盖既有名字值）时原位写回；
      任何会改变对象行长度的新增（首次给空 NameManager 加字段）一律跳过并如实上报，
      避免影响后续偏移/状态表（W4 不做全量重定位）。
    返回 (new_blob, info)。
    """
    from sim_parser import parse_repr
    nm = get_name_manager(sim)
    if nm is None:
        return blob, {"present": False, "changed": False, "reason": "no NameManager"}
    text, s0, s1 = _line_text(blob, int(nm.line))
    d = parse_repr(text.decode("latin-1"))
    orig = dict(d)
    if isinstance(object_id, int):
        cur = d.get("ObjectId")
        if isinstance(cur, int) and len(str(object_id)) == len(str(cur)):
            d["ObjectId"] = object_id
        else:
            return blob, {"present": True, "changed": False,
                          "reason": "ObjectId 变长（新增/位数变化）超出 W4 等宽范围"}
    if names is not None:
        return blob, {"present": True, "changed": False,
                      "reason": "names 名字映射大变长写入超出 W4（留全量重定位/W1）"}
    if d == orig:
        return blob, {"present": True, "changed": False, "reason": "无变更"}
    out = bytearray(blob)
    new_raw = (format_repr(d) + "\n").encode("latin-1", errors="replace")
    out[s0:s1 + 1] = new_raw
    return bytes(out), {"present": True, "changed": True,
                        "width": len(new_raw) - (s1 - s0)}


def maintain_class_versions(blob, sim, created):
    """W4：创建对象后维护尾部 ClassVersions 统计（'Versions': {类名: 实例数}）。

    对每个新对象的类，把 Versions 对应计数累加新实例数；类不存在则新增。
    ClassVersions 是文件最后一段，整行重写不影响任何后续偏移/调用方已持有行号。
    返回 (new_blob, {类名: 增量})。无 created 或无 ClassVersions 时原样返回。
    """
    from sim_parser import parse_repr
    if not created:
        return blob, {}
    cv = _classversions(sim)
    if cv is None:
        return blob, {}
    from collections import Counter
    delta = Counter()
    for o in created.values():
        if o is not None:
            cn = getattr(o, "class_name", None) or (o.dict or {}).get("ClassName")
            if cn and cn != "ClassVersions":
                delta[cn] += 1
    if not delta:
        return blob, {}
    text, s0, s1 = _line_text(blob, int(cv.line))
    d = parse_repr(text.decode("latin-1"))
    versions = d.get("Versions")
    if not isinstance(versions, dict):
        return blob, dict(delta)
    versions = dict(versions)  # 浅拷贝，保留原键序
    for cn, n in sorted(delta.items()):
        versions[cn] = int(versions.get(cn, 0)) + n
    d["Versions"] = versions
    new_raw = (format_repr(d) + "\n").encode("latin-1", errors="replace")
    out = bytearray(blob)
    out[s0:s1 + 1] = new_raw             # 只替换 ClassVersions 行（尾部），不影响任何后续偏移
    return bytes(out), dict(delta)


def save_sim(sim, dest_path, patches=None, created=None, src_path=None,
             array_patches=None, new_arrays=None, deleted=None,
             maintain_versions=True):
    """复制原文件 → 覆盖数组 → 追加导入数组 → 补丁对象行 → 插入新对象。
    maintain_versions：插入新对象后同步维护尾部 ClassVersions 统计（W4）。
    """
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
    cv_delta = {}
    if maintain_versions:
        blob, cv_delta = maintain_class_versions(blob, sim, created)
    with open(dest_path, "wb") as f:
        f.write(blob)
    try:
        sim.class_versions_delta = cv_delta
    except Exception:
        pass
    try:
        # W5：写侧引用白名单审计（非致命，供 GUI/测试提示悬空引用）
        sim.write_reference_issues = audit_write_references(sim, patches, created, deleted)
    except Exception:
        sim.write_reference_issues = []
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


def _ref_one_ok(x, objmap, created_ids, deleted):
    if isinstance(x, bool) or not isinstance(x, int):
        return False
    if x in deleted:
        return False
    return x in objmap or x in created_ids


def _ref_ok(value, objmap, created_ids, deleted):
    if value is None:
        return True
    if isinstance(value, (list, tuple)):
        return all(_ref_one_ok(v, objmap, created_ids, deleted) for v in value)
    return _ref_one_ok(value, objmap, created_ids, deleted)


def audit_write_references(sim, patches=None, created=None, deleted=None):
    """W5：写侧引用白名单审计 —— semantic_dict DOWN/UP 白名单扩展到写侧。

    对被改对象逐属性用 attr_direction 分类：
      down（含 'Keys'）→ 引用集合，值须为 int 列表且每个都能解析到现存对象或本次
         创建（经 created_id_mapping）；指向已删除对象视为悬空。
      up → 标量对象引用，值须为 int/None 且能解析（同上）。
      非引用（数值/枚举/名字等 NON_REF）不检查。
    返回值 issues: [ {oid, key, value, direction, problem} ]；空=全部引用合法可写。
    本审计非致命：仅如实验证/提示，save_sim 不因它失败（写入侧保守但大胆）。
    """
    from semantic_dict import attr_direction
    mapping = created_id_mapping(sim, created)
    created_ids = set(mapping) | set(mapping.values())
    deleted = set(deleted or [])
    issues = []
    for oid, bag in (patches or {}).items():
        if not isinstance(bag, dict):
            continue
        for key, value in bag.items():
            if key == "Keys":
                direction = "down"
            else:
                direction = attr_direction(key)
            if direction not in ("down", "up"):
                continue
            if direction == "down":
                ok = _ref_ok(value, sim.objmap, created_ids, deleted)
            elif value is None:
                ok = True
            elif isinstance(value, bool) or not isinstance(value, int):
                # up 属性值非 int（如枚举/字符串）→ 非引用编辑，跳过避免误报
                continue
            else:
                ok = (value in sim.objmap or value in created_ids) and value not in deleted
            if not ok:
                issues.append({"oid": oid, "key": key, "value": value,
                               "direction": direction,
                               "problem": "引用目标不可解析或指向已删除对象"})
    return issues


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

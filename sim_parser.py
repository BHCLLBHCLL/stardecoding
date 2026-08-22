# -*- coding: utf-8 -*-
"""
STAR-CCM+ .sim (TRANSMIT) 项目文件解析器
========================================

解析 Siemens STAR-CCM+ 的 .sim 项目文件（"TRANSMIT FILE" 格式）。无需安装/启动
STAR-CCM+，纯 Python 3 标准库实现（numpy 可选，用于导出数组）。

文件结构（见 README.md 的详细说明）:

  偏移 0        {'Binary': ..., 'ClassName': 'STAR', 'StatePosition': N, ...}
                （Python repr 字典头，StatePosition 指向"状态区"）
  之后          {'ClassName': 'Array', 'Type': 'Character1', 'nElements': M, ...}
                + M 字节的 ASCII 状态表（STAR-CORE 状态序列化，行宽 80 自动换行，
                换行可能切断单词，解析时先合并再按空白切分）
  之后          32 个类型化数组块（顶点坐标、索引表等，little-endian 二进制）
  偏移 N        StarVersion / loadedLibraries repr 字典
  之后          modeller Python 对象图：每行一个 {'ClassName': ..., attr: ...}
                repr 字典（Simulation / Manager / Region / Scene / Part ...），
                对象 id = 图中序号 + 2，'Parent'/'Keys' 用 id 相互引用

用法:
  python sim_parser.py file.sim --summary            # 总体概览
  python sim_parser.py file.sim --sections           # 各分区（repr 字典/数组块）
  python sim_parser.py file.sim --arrays             # 数组表（类型/元素数/预览）
  python sim_parser.py file.sim --state              # STAR-CORE 状态表记录流
  python sim_parser.py file.sim --objects            # 对象图（id/ClassName/name）
  python sim_parser.py file.sim --tree               # 按 Parent 重构的对象树
  python sim_parser.py file.sim --export <目录>       # 导出数组(.npy/.csv)+JSON

作者: stardecoding 项目 (D:/training/caedecoder/stardecoding)
"""
import argparse
import json
import os
import re
import sys

try:
    import numpy as _np
except ImportError:
    _np = None


# ----------------------------------------------------------------------------
# 1. Python repr 值解析器（该格式用 repr() 风格写出，含 Python2 的 L 后缀、
#    Java 风格 true/false、嵌套 dict/list/tuple）
# ----------------------------------------------------------------------------
class ReprParser:
    """解析 modeller 写出的 Python repr 风格文本值。"""

    def __init__(self, s):
        self.s = s
        self.i = 0
        self.n = len(s)

    def _ws(self):
        while self.i < self.n and self.s[self.i] in " \t\r\n":
            self.i += 1

    def _err(self, msg):
        raise ValueError("%s at pos %d near %r" % (msg, self.i, self.s[self.i:self.i + 40]))

    def parse(self):
        self._ws()
        v = self._value()
        self._ws()
        if self.i != self.n:
            self._err("trailing data")
        return v

    def _value(self):
        self._ws()
        c = self.s[self.i]
        if c == "{":
            return self._dict()
        if c == "[":
            return self._list()
        if c == "(":
            return self._tuple()
        if c in "'" + chr(34):
            return self._string()
        if c == "<":
            return self._opaque()
        m = re.match(r"[A-Za-z_]\w*", self.s[self.i:])
        if m:
            w = m.group(0)
            self.i += len(w)
            self._ws()
            if w in ("true", "True"):
                return True
            if w in ("false", "False"):
                return False
            if w in ("null", "None", "nil"):
                return None
            if self.i < self.n and self.s[self.i] == "(":
                return {"__opaque__": w + self._raw_parens()}
            self._err("bare word " + w)
        m = re.match(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?L?", self.s[self.i:])
        if m:
            t = m.group(0)
            self.i += len(t)
            if t.endswith("L"):
                return int(t[:-1])
            if "." in t or "e" in t.lower() or "E" in t:
                return float(t)
            return int(t)
        self._err("cannot parse value")

    def _dict(self):
        self.i += 1
        d = {}
        self._ws()
        if self.s[self.i] == "}":
            self.i += 1
            return d
        while True:
            self._ws()
            k = self._value()
            self._ws()
            if self.s[self.i] != ":":
                self._err("expected :")
            self.i += 1
            d[k] = self._value()
            self._ws()
            c = self.s[self.i]
            if c == ",":
                self.i += 1
                continue
            if c == "}":
                self.i += 1
                return d
            self._err("expected , or }")

    def _list(self):
        self.i += 1
        items = []
        self._ws()
        if self.s[self.i] == "]":
            self.i += 1
            return items
        while True:
            items.append(self._value())
            self._ws()
            c = self.s[self.i]
            if c == ",":
                self.i += 1
                continue
            if c == "]":
                self.i += 1
                return items
            self._err("expected , or ]")

    def _tuple(self):
        self.i += 1
        items = []
        self._ws()
        if self.s[self.i] == ")":
            self.i += 1
            return tuple(items)
        while True:
            items.append(self._value())
            self._ws()
            c = self.s[self.i]
            if c == ",":
                self.i += 1
                continue
            if c == ")":
                self.i += 1
                return tuple(items)
            self._err("expected , or )")

    def _string(self):
        q = self.s[self.i]
        self.i += 1
        out = []
        while True:
            if self.i >= self.n:
                self._err("unterminated string")
            c = self.s[self.i]
            if c == "\\":
                nxt = self.s[self.i + 1]
                if nxt == "n":
                    out.append("\n")
                elif nxt == "t":
                    out.append("\t")
                elif nxt == "r":
                    out.append("\r")
                else:
                    out.append(nxt)
                self.i += 2
                continue
            if c == q:
                self.i += 1
                return "".join(out)
            out.append(c)
            self.i += 1

    def _raw_parens(self):
        j = self.i
        depth = 0
        while j < self.n:
            if self.s[j] == "(":
                depth += 1
            elif self.s[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        raw = self.s[self.i:j + 1]
        self.i = j + 1
        return raw

    def _opaque(self):
        j = self.s.index(">", self.i)
        raw = self.s[self.i:j + 1]
        self.i = j + 1
        return {"__opaque__": raw}


def parse_repr(s):
    return ReprParser(s).parse()


# ----------------------------------------------------------------------------
# 2. 分区遍历器：整个文件 = 一系列"repr 字典 + 可选二进制载荷"的分区
# ----------------------------------------------------------------------------
TYPE_SIZE = {
    "Character1": 1, "Character": 1, "Byte": 1,
    "Integer4": 4, "Unsigned4": 4, "Integer": 4, "Unsigned": 4,
    "Float8": 8, "Double": 8, "Float": 4,
}

_WS = b" \t\r\n"


def _skip_ws(blob, pos):
    while pos < len(blob) and blob[pos:pos + 1] in _WS:
        pos += 1
    return pos


def walk_sections(blob, start=0, max_sections=200000):
    """遍历分区。返回 [(dict, payload_bytes_or_None, start, payload_start)]。

    文件是线性结构：repr 字典行与数组载荷依次排列，不需要按 StatePosition 跳转
    （StatePosition 只是指向 StarVersion 状态的绝对偏移，与读取顺序一致）。
    """
    out = []
    pos = _skip_ws(blob, start)
    for _ in range(max_sections):
        if pos >= len(blob):
            break
        s0 = pos
        try:
            nl = blob.index(b"\n", pos)
        except ValueError:
            nl = len(blob)
        line = blob[pos:nl].decode("latin-1").strip()
        pos = nl + 1
        if not line:
            continue  # 空行/填充行
        d = parse_repr(line)
        payload = None
        ps = pos
        if isinstance(d, dict) and d.get("ClassName") == "Array":
            nbytes = d["nElements"] * TYPE_SIZE.get(d.get("Type", ""), d.get("sizeof<T>", 1))
            payload = blob[pos:pos + nbytes]
            pos += nbytes
            pos = _skip_ws(blob, pos)
        out.append((d, payload, s0, ps))
    return out


# ----------------------------------------------------------------------------
# 3. STAR-CORE 状态表解析
# ----------------------------------------------------------------------------
NAME_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_\-/+.]*?)(\d+)$")
FMT_RE = re.compile(r"^([A-Za-z]+)(\d*)$")
NUM_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")
BITMAP_RE = re.compile(r"^([FT]*F[FT]*)(\d+)$")
FMT_LETTERS = set("ABCDITVZSuld")  # 状态表格式字母（其余大写/小写字母属于名称）


def unwrap_table(table_lines):
    """按 80 列折行规则重建无换行的表文本。

    折行规则（由逐字节检查 .sim 样例推得）:
      - 行尾是字母、下一行以数字开头     -> 折行吞掉了空格，补 " "（如 "T"|"1"）
      - 行尾是字母、下一行以 _ 开头      -> 同上补 " "（如 "SDL/TYSA"|"_COLOUR79"）
      - 行尾是完整 "?数字" 指针且下一行以数字开头 -> 补 " "
      - 行尾是数字、下一行以字母开头且不是 [eE][+-]?数字 的续写 -> 补 " "
      - 其余情况（行内断词、空格保留在下一行行首）-> 直接拼接
    """
    out = []
    for k, ln in enumerate(table_lines):
        out.append(ln)
        if k >= len(table_lines) - 1:
            break
        nxt = table_lines[k + 1]
        if not ln or not nxt:
            continue
        insert = ""
        if ln[-1].isalpha() and (nxt[0].isdigit() or nxt[0] == "_"):
            insert = " "
        elif re.search(r"\?\d+$", ln) and nxt[0].isdigit():
            insert = " "
        elif ln[-1].isdigit() and nxt[0].isalpha() and not re.match(r"^[eE][+-]?\d", nxt):
            insert = " "
        out.append(insert)
    return "".join(out)


def parse_state_table(text):
    """把状态表文本解析为记录流（无损：每个 token 都被归类）。

    状态表 = 4 行魔数块 + 1 行 T51 banner + 80 列折行的记录表。
    返回 (tokens, records, magic, banner)。
    """
    lines = text.split("\n")
    magic = lines[:4]
    banner = lines[4] if len(lines) > 4 else ""
    table_lines = lines[5:]

    table = unwrap_table(table_lines)
    tokens = table.split()
    records = []
    i = 0
    n = len(tokens)

    def next_non_number(j):
        while j < n:
            t = tokens[j]
            if NUM_RE.match(t) or t.startswith("+-"):
                j += 1
                continue
            return j
        return n

    def classify(tok):
        if NUM_RE.match(tok):
            return "int" if re.fullmatch(r"[+-]?\d+", tok) else "float"
        if tok.startswith("+-"):
            return "element"
        if tok.startswith("?"):
            return "pointer"
        if BITMAP_RE.match(tok):
            return "bitmap"
        m = FMT_RE.match(tok)
        if m and all(c in FMT_LETTERS for c in m.group(1)):
            return "fmt"
        if NAME_RE.match(tok):
            return "name"
        return "other"

    while i < n:
        tok = tokens[i]
        kind = classify(tok)

        if kind == "name":
            m = NAME_RE.match(tok)
            name, sid = m.group(1), int(m.group(2))
            rec = {"kind": "named", "name": name, "id": sid, "flags": None,
                   "version": None, "fmt": None, "value": None, "values": [],
                   "token_index": i}
            i += 1
            if i < n and NUM_RE.match(tokens[i]) and re.fullmatch(r"[+-]?\d+", tokens[i]):
                rec["flags"] = int(tokens[i]); i += 1
            # 可选 version：后一个 token 是整数且再后一个是格式 token
            if (i + 1 < n and re.fullmatch(r"[+-]?\d+", tokens[i])
                    and classify(tokens[i + 1]) == "fmt"):
                rec["version"] = int(tokens[i]); i += 1
            if i < n and classify(tokens[i]) == "fmt":
                m2 = FMT_RE.match(tokens[i])
                rec["fmt"] = m2.group(1)
                rec["value"] = int(m2.group(2)) if m2.group(2) else None
                i += 1
            j = next_non_number(i)
            while i < j:
                t = tokens[i]
                if t.startswith("+-"):
                    rec["values"].append({"kind": "element", "value": _num(t[2:])})
                else:
                    rec["values"].append({"kind": classify(t), "value": _num(t)})
                i += 1
            records.append(rec)

        elif kind == "fmt":
            m = FMT_RE.match(tok)
            rec = {"kind": "anonymous", "name": None, "id": None, "flags": None,
                   "version": None, "fmt": m.group(1),
                   "value": int(m.group(2)) if m.group(2) else None,
                   "values": [], "token_index": i}
            i += 1
            j = next_non_number(i)
            while i < j:
                t = tokens[i]
                if t.startswith("+-"):
                    rec["values"].append({"kind": "element", "value": _num(t[2:])})
                else:
                    rec["values"].append({"kind": classify(t), "value": _num(t)})
                i += 1
            records.append(rec)

        elif kind == "pointer":
            rec = {"kind": "pointer", "ref": int(tok[1:]), "values": [], "token_index": i}
            i += 1
            j = next_non_number(i)
            while i < j:
                t = tokens[i]
                if t.startswith("+-"):
                    rec["values"].append({"kind": "element", "value": _num(t[2:])})
                else:
                    rec["values"].append({"kind": classify(t), "value": _num(t)})
                i += 1
            records.append(rec)

        elif kind == "element":
            records.append({"kind": "element", "value": _num(tok[2:]), "values": [], "token_index": i})
            i += 1

        elif kind == "bitmap":
            m = BITMAP_RE.match(tok)
            records.append({"kind": "bitmap", "bits": m.group(1), "value": int(m.group(2)),
                            "values": [], "token_index": i})
            i += 1

        elif kind in ("int", "float"):
            # 游离数值（不属于任何记录的数据区）— 归入一个"裸数据"记录
            j = next_non_number(i)
            records.append({"kind": "data", "values": [
                {"kind": classify(t), "value": _num(t)} for t in tokens[i:j]],
                "token_index": i})
            i = j

        else:
            # 裸字符串值（如 "SDL/TYSA"）或未识别的 token
            records.append({"kind": "string" if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\-/+.]*", tok)
                            else "other",
                            "token": tok, "token_index": i})
            i += 1

    # banner 记录放在最前（"T51 : TRANSMIT FILE created by ..."）
    bm = re.match(r"^([A-Za-z]+)(\d+)\s*:\s*(.*)$", banner.strip())
    if bm:
        records.insert(0, {
            "kind": "banner", "fmt": bm.group(1), "value": int(bm.group(2)),
            "message": bm.group(3), "token_index": -1})
    return tokens, records, magic, banner


def _num(t):
    try:
        return int(t) if re.fullmatch(r"[+-]?\d+", t) else float(t)
    except ValueError:
        return t


# ----------------------------------------------------------------------------
# 4. 对象图
# ----------------------------------------------------------------------------
class SimObject:
    __slots__ = ("id", "dict", "line")

    def __init__(self, oid, d, line):
        self.id = oid
        self.dict = d
        self.line = line

    @property
    def class_name(self):
        return self.dict.get("ClassName")

    @property
    def name(self):
        # 顶层对象用 'name'，其余对象用 'PresentationName'
        return self.dict.get("name") or self.dict.get("PresentationName")

    def __repr__(self):
        return "<SimObject id=%d %s name=%r>" % (self.id, self.class_name, self.name)


def build_object_graph(sections, objects_start):
    """从对象区（repr 字典行）构建对象表。id = 图中序号 + 2。"""
    objs = []
    graph_idx = 0
    for d, payload, s0, ps in sections:
        if s0 < objects_start:
            continue
        objs.append(SimObject(graph_idx + 2, d, s0))
        graph_idx += 1
    return objs


def resolve_ref(value, objmap):
    if isinstance(value, int) and value in objmap:
        return objmap[value]
    return None


def build_tree(objs):
    """按所有权引用构建对象树。

    所有权边（按优先级，先到先得）:
      1. 显式 'Parent' 属性（Parent == 自身 id 或非法 id 表示根）;
      2. 出现在某个对象 'Keys' 列表中的对象 → 该对象的子节点;
      3. 被某个对象 'NameManager' 属性引用的 NameManager 对象 → 该对象的子节点。
    其余对象（Dimensions / Units / 事件对象等通过语义属性相互引用）保持游离，
    由调用方自行归类。
    """
    objmap = {o.id: o for o in objs}
    parent_of = {}
    for o in objs:
        p = o.dict.get("Parent")
        if isinstance(p, int) and p in objmap and p != o.id:
            parent_of[o.id] = p
    for o in objs:
        keys = o.dict.get("Keys")
        if isinstance(keys, list):
            for k in keys:
                if isinstance(k, int) and k in objmap and k not in parent_of:
                    parent_of[k] = o.id
    for o in objs:
        nm = o.dict.get("NameManager")
        if isinstance(nm, int) and nm in objmap and nm not in parent_of:
            parent_of[nm] = o.id
    roots = [o for o in objs if o.id not in parent_of]
    children = {o.id: [] for o in objs}
    for cid, pid in parent_of.items():
        children.setdefault(pid, []).append(objmap[cid])
    for k in children:
        children[k].sort(key=lambda o: o.id)
    return roots, children


def print_tree(roots, children, indent=0, max_depth=12, show_nm_leaves=False):
    for o in roots:
        kids = children.get(o.id, [])
        if o.class_name == "NameManager" and not kids and not show_nm_leaves:
            continue
        tag = ""
        if o.name is not None:
            tag = " name=%r" % o.name
        print("  " * indent + "[%d] %s%s" % (o.id, o.class_name, tag))
        if indent < max_depth:
            for c in kids:
                print_tree([c], children, indent + 1, max_depth, show_nm_leaves)


# ----------------------------------------------------------------------------
# 5. 顶层 SimFile
# ----------------------------------------------------------------------------
class SimFile:
    """解析后的 .sim 文件。"""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            blob = f.read()
        self.container_entry = None
        if blob[:2] == b"PK":
            # ZIP 容器：取唯一/最大的条目作为主载荷（压缩版 .sim 变体）
            import io as _io
            import zipfile as _zipfile
            zf = _zipfile.ZipFile(_io.BytesIO(blob))
            names = [n for n in zf.namelist() if not n.endswith("/")]
            if names:
                entry = names[0] if len(names) == 1 else max(
                    names, key=lambda n: zf.getinfo(n).file_size)
                self.container_entry = entry
                blob = zf.read(entry)
            zf.close()
        self.blob = blob
        self.sections = walk_sections(self.blob)
        self.header = self.sections[0][0]
        self.state_position = self.header.get("StatePosition")
        self.arrays = []
        self.state_text = None
        self.state_magic = []
        self.state_banner = ""
        self.tokens = []
        self.records = []
        self.objects = []
        self.roots = []
        self.children = {}
        self.objmap = {}

        for d, payload, s0, ps in self.sections:
            if isinstance(d, dict) and d.get("ClassName") == "Array":
                self.arrays.append({
                    "index": len(self.arrays),
                    "type": d.get("Type"),
                    "count": d.get("nElements"),
                    "start": ps,
                    "data": payload,
                    "dict": d,
                })
                if d.get("Type") == "Character1" and self.state_text is None:
                    self.state_text = payload.decode("latin-1")
                    self.tokens, self.records, self.state_magic, self.state_banner = \
                        parse_state_table(self.state_text)

        # 对象图：从 Simulation 区（第一个 star.common.Simulation）开始
        obj_start = None
        for d, payload, s0, ps in self.sections:
            if isinstance(d, dict) and d.get("ClassName") == "star.common.Simulation":
                obj_start = s0
                break
        if obj_start is not None:
            self.objects = build_object_graph(self.sections, obj_start)
            self.objmap = {o.id: o for o in self.objects}
            self.roots, self.children = build_tree(self.objects)

    # ---- 便捷访问 ----
    def object_by_id(self, oid):
        return self.objmap.get(oid)

    def array_data(self, index):
        """解码数组（numpy 可用时返回 ndarray，否则返回 list/bytes）。"""
        a = self.arrays[index]
        t = a["type"]
        if _np is not None and t in ("Float8", "Float", "Unsigned4", "Integer4"):
            dt = {"Float8": "<f8", "Float": "<f4", "Unsigned4": "<u4", "Integer4": "<i4"}[t]
            return _np.frombuffer(a["data"], dtype=dt)
        if t == "Character1":
            return a["data"].decode("latin-1")
        return a["data"]

    # ---- 汇总 ----
    def summary(self):
        L = []
        L.append("文件: %s" % self.path)
        L.append("大小: %d 字节" % len(self.blob))
        L.append("头部: %s" % {k: v for k, v in self.header.items() if k != "Binary"})
        L.append("Binary 标记: %r" % self.header.get("Binary"))
        L.append("StatePosition: %d" % self.state_position)
        L.append("分区数: %d" % len(self.sections))
        L.append("数组块: %d" % len(self.arrays))
        L.append("状态表: %d 字符 / %d token / %d 记录" % (
            len(self.state_text or ""), len(self.tokens), len(self.records)))
        if self.state_magic:
            L.append("魔数块: %r" % self.state_magic)
        if self.state_banner:
            L.append("Banner: %s" % self.state_banner.strip())
        L.append("对象图: %d 个对象 (id 2..%d)" % (len(self.objects),
                                                  len(self.objects) + 1 if self.objects else 0))
        if self.roots:
            main_roots = [r for r in self.roots if self.children.get(r.id)]
            loose = [r for r in self.roots if not self.children.get(r.id)]
            L.append("对象树根（有子树）:")
            for r in main_roots:
                L.append("  [%d] %s%s" % (r.id, r.class_name,
                                          (" name=%r" % r.name) if r.name else ""))
            if loose:
                from collections import Counter
                cn = Counter(o.class_name for o in loose)
                L.append("游离对象（无 Parent/Keys/NameManager 归属）: %d 个, 类别: %s" % (
                    len(loose), ", ".join("%s x%d" % kv for kv in cn.most_common(8))))
        return "\n".join(L)

    def export(self, outdir):
        os.makedirs(outdir, exist_ok=True)
        for a in self.arrays:
            t = a["type"]
            if t == "Character1":
                with open(os.path.join(outdir, "state_table.txt"), "w", encoding="latin-1") as f:
                    f.write(self.state_text)
                continue
            data = self.array_data(a["index"])
            base = os.path.join(outdir, "array_%02d_%s_n%d" % (a["index"], t, a["count"]))
            if _np is not None and hasattr(data, "dtype"):
                _np.save(base + ".npy", data)
                _np.savetxt(base + ".csv", data, delimiter=",", fmt="%r")
            else:
                with open(base + ".raw", "wb") as f:
                    f.write(a["data"])
        with open(os.path.join(outdir, "objects.json"), "w", encoding="utf-8") as f:
            json.dump([{"id": o.id, "line": o.line, "obj": o.dict} for o in self.objects],
                      f, indent=1, default=str)
        with open(os.path.join(outdir, "state_records.json"), "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=1)
        with open(os.path.join(outdir, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(self.summary())
        return outdir


# ----------------------------------------------------------------------------
# 6. CLI
# ----------------------------------------------------------------------------
def _fmt_rec(rec, objmap=None):
    if rec["kind"] == "named":
        s = "named  %-24s id=%-4s flags=%s ver=%s fmt=%-14s val=%s" % (
            rec["name"], rec["id"], rec["flags"], rec["version"], rec["fmt"], rec["value"])
        if rec["values"]:
            vals = rec["values"]
            shown = ", ".join(str(v["value"]) for v in vals[:8])
            s += "  [%d values: %s%s]" % (len(vals), shown, " ..." if len(vals) > 8 else "")
        return s
    if rec["kind"] == "anonymous":
        s = "anon   fmt=%-14s val=%s" % (rec["fmt"], rec["value"])
        if rec["values"]:
            vals = rec["values"]
            shown = ", ".join(str(v["value"]) for v in vals[:8])
            s += "  [%d values: %s%s]" % (len(vals), shown, " ..." if len(vals) > 8 else "")
        return s
    if rec["kind"] == "pointer":
        target = ""
        if objmap and rec["ref"] in objmap:
            o = objmap[rec["ref"]]
            target = " -> %s%s" % (o.class_name, (" name=%r" % o.name) if o.name else "")
        s = "ptr    ?%-4d%s" % (rec["ref"], target)
        if rec["values"]:
            vals = rec["values"]
            shown = ", ".join(str(v["value"]) for v in vals[:8])
            s += "  [%d values: %s%s]" % (len(vals), shown, " ..." if len(vals) > 8 else "")
        return s
    if rec["kind"] == "element":
        return "element %s" % rec["value"]
    if rec["kind"] == "bitmap":
        return "bitmap %s value=%s" % (rec["bits"], rec["value"])
    if rec["kind"] == "data":
        vals = rec["values"]
        shown = ", ".join(str(v["value"]) for v in vals[:8])
        return "data   [%d values: %s%s]" % (len(vals), shown, " ..." if len(vals) > 8 else "")
    if rec["kind"] == "banner":
        return "banner %s%d : %s" % (rec["fmt"], rec["value"], rec["message"])
    return "%s %s" % (rec["kind"], rec.get("token", ""))


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="STAR-CCM+ .sim (TRANSMIT) 文件解析器")
    ap.add_argument("file", help=".sim 文件路径")
    ap.add_argument("--summary", action="store_true", help="总体概览")
    ap.add_argument("--sections", action="store_true", help="列出全部分区")
    ap.add_argument("--arrays", action="store_true", help="列出数组表")
    ap.add_argument("--state", action="store_true", help="输出状态表记录流")
    ap.add_argument("--objects", action="store_true", help="列出对象图")
    ap.add_argument("--tree", action="store_true", help="按 Parent 输出对象树")
    ap.add_argument("--export", metavar="DIR", help="导出到目录（数组 .npy/.csv + JSON）")
    ap.add_argument("--max-records", type=int, default=0, help="--state 最多输出的记录数（0=全部）")
    args = ap.parse_args(argv)

    sim = SimFile(args.file)

    if args.summary or not any([args.sections, args.arrays, args.state,
                                args.objects, args.tree, args.export]):
        print(sim.summary())

    if args.sections:
        print("\n== 分区 ==")
        for i, (d, payload, s0, ps) in enumerate(sim.sections):
            cls = d.get("ClassName") if isinstance(d, dict) else "?"
            extra = ""
            if payload is not None:
                extra = "  payload %d 字节" % len(payload)
            print("%4d @%-8d %-40s%s" % (i, s0, cls, extra))

    if args.arrays:
        print("\n== 数组表 ==")
        for a in sim.arrays:
            data = sim.array_data(a["index"])
            if _np is not None and hasattr(data, "dtype"):
                preview = ", ".join("%r" % v for v in data[:6].tolist())
            elif isinstance(data, str):
                preview = data[:40]
            else:
                preview = repr(data[:24])
            print("%3d  %-10s n=%-6d @%-8d  [%s ...]" % (
                a["index"], a["type"], a["count"], a["start"], preview))

    if args.state:
        print("\n== STAR-CORE 状态表（%d 记录）==" % len(sim.records))
        for k, rec in enumerate(sim.records):
            if args.max_records and k >= args.max_records:
                print("... 共 %d 条，其余略" % len(sim.records))
                break
            print("%5d  %s" % (k, _fmt_rec(rec, sim.objmap)))

    if args.objects:
        print("\n== 对象图（%d）==" % len(sim.objects))
        for o in sim.objects:
            print("%5d  %-60s %s" % (o.id, o.class_name,
                                     ("name=%r" % o.name) if o.name else ""))

    if args.tree:
        print("\n== 对象树 ==")
        for r in sim.roots:
            if sim.children.get(r.id):
                print_tree([r], sim.children)

    if args.export:
        sim.export(args.export)
        print("\n已导出到 %s" % args.export)


if __name__ == "__main__":
    main()

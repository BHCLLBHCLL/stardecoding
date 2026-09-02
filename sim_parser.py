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
import struct as _struct
import sys

try:
    import numpy as _np
except ImportError:
    _np = None

try:
    from semantic_dict import layer_of, resolve_class, attr_direction, LAYER_CN
except ImportError:  # 允许单文件独立运行
    def layer_of(cn):
        return "unknown"

    def resolve_class(cn):
        return cn

    def attr_direction(name):
        return None

    LAYER_CN = {"unknown": "未分类"}


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
NAME_RE2 = re.compile(r"^([A-Za-z0-9_][A-Za-z0-9_\-/+.]*[A-Za-z_])(\d+)$")  # 允许数字开头（如 3DX_NON_PERSISTENT_ID14）
FMT_RE = re.compile(r"^([A-Za-z]+)(\d*)$")
NUM_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")
BITMAP_RE = re.compile(r"^([FT]+)(\d+)$")
BITMAP2_RE = re.compile(r"^([FTPB?]+)(\d+)$")  # B/P 位图 + 内嵌 ?? 指针链（如 BBBBBBBB????0）
FMT_LETTERS = set("ABCDIJLTVZSuld")  # 状态表格式字母（其余大写/小写字母属于名称）


def parse_pointer(tok):
    """解析 '?' 指针 token（支持嵌套与浮点引用）：?5, ?24, ?-.0376, ??0, ????0 ..."""
    rest = tok[1:]
    if rest.startswith("?"):
        return {"nested": parse_pointer(rest)}
    if re.fullmatch(r"[+-]?\d+", rest):
        return int(rest)
    if NUM_RE.match(rest):
        return float(rest)
    return rest  # 原样保留


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


KNOWN_BIN_FMTS = {
    "A", "CI", "CZ", "DI", "dA", "dI", "dZ", "dCCZ", "uI", "lCCCDCCDI",
    "CCCCCCCA", "CCCCCCCCCA", "CCCCCCCCCCCCCDI", "CCCCCCCCCCCCCCCCCCCCCCCA",
    "S", "V", "Z", "T", "J", "L", "F", "Q", "G", "D", "TTT",
}


def decode_binary_values(raw_hex):
    """尽力解码二进制数值流：把原始字节区拆为 2 字节整数与 8 字节双精度交替段。

    二进制状态表的数值流（T/Q/G/V 块等）混合存放 2 字节小端整数与 8 字节小端
    双精度，且无显式类型标记。本函数按"8 字节对齐的有限浮点"扫描双精度段，
    其余按 2 字节整数解释，返回 [(kind, offset, value), ...] 与置信统计。
    """
    import struct as _struct
    if not raw_hex:
        return [], {"int_runs": 0, "double_runs": 0, "double_values": 0, "bytes": 0}
    blob = bytes.fromhex(raw_hex.replace(" ", ""))
    n = len(blob)
    out = []
    stats = {"int_runs": 0, "double_runs": 0, "double_values": 0, "bytes": n}
    i = 0
    while i < n:
        # 尝试在当前位置找一段连续的合法双精度（至少 2 个连续才认定为 double run）
        j = i
        while j + 8 <= n:
            d = _struct.unpack_from("<d", blob, j)[0]
            if not (d == 0.0 or 1e-30 <= abs(d) <= 1e30):
                break
            j += 8
        run = (j - i) // 8
        if run >= 2 and j - i >= 16:
            stats["double_runs"] += 1
            stats["double_values"] += run
            for k in range(run):
                out.append(("double", i + 8 * k, _struct.unpack_from("<d", blob, i + 8 * k)[0]))
            i = j
        else:
            stats["int_runs"] += 1
            while i + 1 < n:
                out.append(("int", i, _struct.unpack_from("<h", blob, i)[0]))
                i += 2
            if i < n:
                out.append(("byte", i, blob[i]))
                i += 1
    return out, stats


def parse_state_table_binary(text):
    """二进制状态表完整文法（G9，语料逆向验证，4 个 binary 样本逐字节往返）。

    结构（长度前缀文法）：
      banner：'PS' 变体前缀 + "TRANSMIT FILE created by modeller version <v>"
              + NUL 填充 + 长度前缀 SCH 串 + NUL；
      记录流：
        named      := [len][name] [id 3B = id<<8] [flags 1B] [version 1B iff id==0]
                      [fmt run] [value 1B] [stream]
        anonymous  := [fmt run] [value 1B] [stream]
        data       := 原始字节（非记录头）
      名字 = 1 字节长度 L（0<L<128）+ L 个名字字节（lattice/mesh/owner 等）；
      id 3 字节大端、低字节恒 0（= id<<8，mesh 1006 存 `03 ee 00`）；
      flags 真实记录 <=2、fmt 非空、id 低字节为 0——任一不合即流数据假阳性；
      记录尾判定：fmt 后若为合法名字长度前缀（下一记录）则无 value/stream；
                  否则取 1 字节 value + 原始流（吞到下一记录头）。
      每字节归属唯一记录（无损）；serialize_binary_records 可逐字节还原。
    返回 (tokens, records, magic, banner)。
    """
    lines = text.split("\n")
    bi = next((k for k, l in enumerate(lines[:60])
               if "TRANSMIT FILE" in l and k > 0), None)
    if bi is None:
        bi = 4
    magic = lines[:bi]
    body_all = text.encode("latin-1")
    bm = re.search(rb"TRANSMIT FILE created by modeller version (\d+).{0,24}?SCH_([A-Za-z0-9_]+)",
                   body_all)
    banner = ""
    if bm:
        banner = "T51 : TRANSMIT FILE created by modeller version %s SCH_%s" % (
            bm.group(1).decode("latin-1"), bm.group(2).decode("latin-1"))
        body = body_all[bm.end():]
    else:
        body = body_all

    FMT_BYTES = set(b"ABCDIJLTVZSuld")
    NAME_BYTES = set(
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")
    n = len(body)
    records = []
    tokens = []

    def name_at(p):
        """p 处若为合法名字记录头（长度前缀+名字+id+flags+[version]+fmt）
        且 id 低字节 0、flags<=2、fmt 非空，返回字段 dict；否则 None。"""
        if p >= n:
            return None
        L = body[p]
        if not (0 < L < 128) or p + 1 + L > n:
            return None
        chunk = body[p + 1:p + 1 + L]
        if not all(x in NAME_BYTES for x in chunk):
            return None
        i = p + 1 + L
        if i + 4 > n:
            return None
        idd = int.from_bytes(body[i:i + 3], "big") >> 8
        if body[i + 2] != 0:   # 真实记录 id 低字节恒 0（id<<8 编码）
            return None
        i += 3
        flags = body[i]
        i += 1
        if flags > 2:
            return None
        version = None
        if idd == 0:
            if i >= n:
                return None
            version = body[i]
            i += 1
        j = i
        while j < n and body[j] in FMT_BYTES:
            j += 1
        fmt = body[i:j].decode("latin-1")
        if not fmt:
            return None
        return {"name": chunk.decode("latin-1"), "id": idd, "flags": flags,
                "version": version, "fmt": fmt, "next": j}

    def fmt_at(p):
        if p >= n or body[p] not in FMT_BYTES:
            return None, p
        j = p
        while j < n and body[j] in FMT_BYTES:
            j += 1
        return body[p:j].decode("latin-1"), j

    def next_head(p):
        nm = name_at(p)
        if nm:
            return "named", nm
        fm, j = fmt_at(p)
        if fm:
            return "anonymous", {"fmt": fm, "next": j}
        return None, None

    i = 0
    while i < n:
        start = i
        kind, hd = next_head(i)
        if kind is None:
            j = i
            while j < n and body[j] not in FMT_BYTES and body[j] not in NAME_BYTES:
                j += 1
            if j == i:
                j = i + 1
            tokens.append("<bytes:%d>" % (j - i))
            records.append({"kind": "data", "name": None, "id": None,
                            "flags": None, "version": None, "fmt": None,
                            "value": None, "values": [],
                            "token_index": len(tokens) - 1,
                            "raw": body[start:j].hex(" ")})
            i = j
            continue
        if kind == "named":
            rec = {"kind": "named", "name": hd["name"], "id": hd["id"],
                   "flags": hd["flags"], "version": hd["version"],
                   "fmt": hd["fmt"], "value": None, "values": [],
                   "token_index": len(tokens), "raw": None}
            tokens.append(hd["name"])
            i = hd["next"]
        else:
            rec = {"kind": "anonymous", "name": None, "id": None,
                   "flags": None, "version": None, "fmt": hd["fmt"],
                   "value": None, "values": [], "token_index": len(tokens),
                   "raw": None}
            tokens.append(hd["fmt"])
            i = hd["next"]
        # 记录尾：fmt 后若为下一名字长度前缀则无 value/stream
        if name_at(i):
            records.append(rec)
            continue
        value = body[i]
        i += 1
        j = i
        while j < n and next_head(j)[0] is None:
            j += 1
        rec["value"] = value
        rec["raw"] = body[i:j].hex(" ") if j > i else None
        records.append(rec)
        i = j

    if banner:
        records.insert(0, {"kind": "banner", "name": None, "id": None,
                           "flags": None, "version": None, "fmt": "T",
                           "value": 51, "values": [], "token_index": -1,
                           "raw": None,
                           "message": banner.split(": ", 1)[-1]})
    # 对带 raw 的记录追加尽力数值解码（2 字节整 + 8 字节双精度交替段）
    for rec in records:
        if rec.get("raw"):
            vals, stats = decode_binary_values(rec["raw"])
            rec["values_decoded"] = vals
            rec["decode_stats"] = stats
    return tokens, records, magic, banner


def serialize_binary_records(records):
    """按完整文法重新序列化二进制记录流 → 逐字节还原（G9 可逆性验收）。

    与 parse_state_table_binary 互逆：serialize(parse(body)) == body。
    banner 记录（kind='banner'）不参与写出（其字节属表头区）。
    """
    out = bytearray()
    for r in records:
        if r.get("kind") == "banner":
            continue
        out += _binary_record_bytes(r)
    return bytes(out)


def _binary_record_bytes(r):
    """单条二进制记录的原始字节（嵌套于 serialize_binary_records 的重组逻辑）。"""
    out = bytearray()
    if r.get("kind") == "data":
        if r.get("raw"):
            out += bytes.fromhex(r["raw"])
        return bytes(out)
    if r.get("kind") == "named":
        nb = r["name"].encode("latin-1")
        out += bytes([len(nb)]) + nb
        out += (r["id"] << 8).to_bytes(3, "big")
        out += bytes([r["flags"]])
        if r["id"] == 0:
            out += bytes([r["version"]])
    out += r["fmt"].encode("latin-1")
    if r.get("value") is not None:
        out += bytes([r["value"]])
    if r.get("raw"):
        out += bytes.fromhex(r["raw"])
    return bytes(out)


def binary_record_spans(records):
    """每条记录在记录流中的字节区间 [start, end)（W2 差分定位用）。

    与 serialize_binary_records 的拼接顺序一致；banner 记录（不参与写出）区间为 None。
    """
    spans = []
    pos = 0
    for r in records:
        if r.get("kind") == "banner":
            spans.append(None)
            continue
        n = len(_binary_record_bytes(r))
        spans.append((pos, pos + n))
        pos += n
    return spans


def edit_binary_state_records(state_text, edits):
    """W2 状态表安全编辑：只动已确证记录（G1/G9 产物），不碰尾部/魔数/其他记录。

    edits: [{''index'': 记录序号, 字段...}]，字段可选 name/id/flags/version，
          且仅允许**等宽**编辑（name 等长、id 3B、flags 1B、id==0 的 version 1B）；
          任何变长编辑（改 name 长度、id<->0 增删 version 字节）抛 ValueError（留给 W1 全量重定位）。
    返回 (new_state_bytes, changed_spans)：
      new_state_bytes  = magic + banner + 打补丁后的记录流（latin-1 bytes）；
      changed_spans    = 相对记录流的字节区间列表（差分验证：仅这些字节变动）。
    """
    body_all = state_text.encode("latin-1")
    bm = re.search(rb"TRANSMIT FILE created by modeller version (\d+).{0,24}?SCH_([A-Za-z0-9_]+)",
                   body_all)
    offset = bm.end() if bm else 0
    _tok, records, _magic, _ban = parse_state_table_binary(state_text)
    patched = [dict(r) for r in records]
    targets = []
    for e in edits:
        idx = int(e["index"])
        if idx < 0 or idx >= len(patched):
            raise ValueError("记录序号越界: %d" % idx)
        r = patched[idx]
        if r.get("kind") != "named":
            raise ValueError("记录 %d 非 named（kind=%r），不可安全改头字段" % (idx, r.get("kind")))
        p = {"index": idx, "kind": "named", "span_from": r}
        if "name" in e:
            nv = e["name"].encode("latin-1")
            if len(nv) != len(r["name"].encode("latin-1")):
                raise ValueError("改名变长（等长 %d -> %d）留给 W1" % (len(r["name"]), len(nv)))
            r["name"] = e["name"]
        if "id" in e:
            nv = int(e["id"])
            if nv < 0:
                raise ValueError("id 非负: %d" % nv)
            if (r["id"] == 0) != (nv == 0):
                raise ValueError("id 在 0 与非 0 间切换需增删 version 字节（变长，留给 W1）")
            r["id"] = nv
        if "flags" in e:
            nv = int(e["flags"])
            if not (0 <= nv <= 2):
                raise ValueError("flags 越界（须<=2）: %d" % nv)
            r["flags"] = nv
        if "version" in e:
            if r["id"] != 0:
                raise ValueError("非 id==0 记录无 version 字节")
            r["version"] = int(e["version"])
        targets.append(idx)
    ser = serialize_binary_records(patched)
    # 差分验证：除目标记录外逐字节不变（含 banner/magic 前缀）
    spans = binary_record_spans(records)
    orig_body = body_all[offset:]
    changed = sorted(sp for (i, sp) in enumerate(spans)
                     if sp is not None and i in targets)
    spans2 = binary_record_spans(patched)
    for i, sp in enumerate(spans):
        if sp is None or i in targets:
            continue
        a, b = sp
        a2, b2 = spans2[i]
        if orig_body[a:b] != ser[a2:b2]:
            raise AssertionError("非目标记录 %d 被意外改动" % i)
    new_state = body_all[:offset] + ser
    return new_state, changed


def verify_binary_state_edit(state_text, new_bytes, edits):
    """重解析新状态表并断言：仅目标记录的头字段按 edits 变化，其余记录逐字一致。"""
    _t, records, _m, _b = parse_state_table_binary(state_text)
    _t2, records2, _m2, _b2 = parse_state_table_binary(new_bytes.decode("latin-1"))
    if len(records) != len(records2):
        raise AssertionError("记录数变化 %d -> %d" % (len(records), len(records2)))
    targets = set(int(e["index"]) for e in edits)
    diffs = 0
    for i, (r, r2) in enumerate(zip(records, records2)):
        a = _binary_record_bytes(r)
        b = _binary_record_bytes(r2)
        if i in targets:
            if a != b:
                diffs += 1
        elif a != b:
            raise AssertionError("非目标记录 %d 序列化后改变" % i)
    if diffs != len(targets):
        raise AssertionError("目标记录变动数不符: %d/%d" % (diffs, len(targets)))
    return True


def parse_state_table(text):
    """把状态表文本解析为记录流（无损：每个 token 都被归类）。

    状态表 = 魔数块（外层 4 行 / 嵌套 3 行）+ 1 行 T51 banner + 80 列折行的记录表。
    自动识别 ASCII / 二进制编码变体。
    返回 (tokens, records, magic, banner)。
    """
    lines = text.split("\n")
    # banner 行 = 包含 "TRANSMIT FILE" 的行（外层与嵌套块行数不同，动态定位；
    # 多 id 魔数可能占几十行，故扫描窗口放宽到 60 行）
    bi = next((k for k, l in enumerate(lines[:60])
               if "TRANSMIT FILE" in l and k > 0), None)
    if bi is None:
        bi = 4
    magic = lines[:bi]
    banner = lines[bi] if bi < len(lines) else ""
    table_lines = lines[bi + 1:]

    # 二进制变体检测：记录区包含 NUL 字节
    if "\x00" in "\n".join(table_lines[:20]):
        return parse_state_table_binary(text)

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
        if BITMAP_RE.match(tok) or BITMAP2_RE.match(tok):
            return "bitmap"
        m = FMT_RE.match(tok)
        if m and all(c in FMT_LETTERS for c in m.group(1)):
            return "fmt"
        if NAME_RE.match(tok) or NAME_RE2.match(tok):
            return "name"
        return "other"

    while i < n:
        tok = tokens[i]
        kind = classify(tok)

        if kind == "name":
            m = NAME_RE.match(tok) or NAME_RE2.match(tok)
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
            rec = {"kind": "pointer", "ref": parse_pointer(tok), "values": [], "token_index": i}
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
            m = BITMAP_RE.match(tok) or BITMAP2_RE.match(tok)
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

    @property
    def layer(self):
        return layer_of(self.class_name or "")

    @property
    def resolved_class(self):
        return resolve_class(self.class_name or "")

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
    """按所有权引用构建对象树（结合语义字典的引用方向）。

    边（按优先级，先到先得）:
      1. 显式 'Parent' 属性（Parent == 自身 id 或非法 id 表示根）;
      2. 'Keys' 列表项 → 该对象的子节点;
      3. 'NameManager' 引用 → 该对象的子节点;
      4. 语义字典（semantic_dict.attr_direction）:
         - 'down' 属性（管理器/复数容器）→ 属性值是子节点;
         - 'up' 属性（Parent/Simulation/Region/Units...）→ 被引用对象是父节点。
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
    # 语义字典边（先 down 后 up，先到先得）
    for o in objs:
        if o.class_name == "ClassVersions":
            continue  # 尾部统计对象，值不是引用
        for attr, v in o.dict.items():
            direction = attr_direction(attr)
            if direction is None:
                continue
            if attr == "Dimensions" and isinstance(v, list):
                continue  # 列表形式 = 量纲指数向量（数据）；仅整数值是 Dimensions 对象引用
            cands = []
            if isinstance(v, int):
                cands.append(v)
            elif isinstance(v, list):
                cands.extend(x for x in v if isinstance(x, int))
            for c in cands:
                if c not in objmap or c == o.id or c in parent_of:
                    continue
                if direction == "down":
                    parent_of[c] = o.id
                else:  # up：被引用对象是父
                    parent_of[o.id] = c
                    break
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


def compact_face_indices(faces, nverts):
    """把面顶点编号映射到 [0, nverts)。

    STAR-CCM+ 面索引常见三种形态:
      - 0 基连续: min=0, max=n-1
      - 1 基连续: min=1, max=n（或 max<=n）
      - 分块偏移: min=k, max=k+n-1（如 genericHelicopter Wind Tunnel 的 k=195）

    旧逻辑一律按 1 基减 1 再 clip 到 [0, n-1]。偏移块的超界下标会被钉到最后一个
    顶点，3D 上外框拧成星形。返回 (faces0, ok)。
    """
    if _np is None or faces is None or int(nverts) <= 0:
        return faces, False
    f = _np.asarray(faces, dtype=_np.int64)
    if f.size == 0:
        return f, True
    lo, hi = int(f.min()), int(f.max())
    n = int(nverts)
    span = hi - lo + 1
    if span == n:
        return f - lo, True
    if lo >= 1 and hi <= n:
        return f - 1, True
    if lo >= 0 and hi < n:
        return f, True
    if lo >= 1:
        f = f - 1
    return _np.clip(f, 0, n - 1), False


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
        self.state_mode = "ascii"
        self.state_magic = []
        self.state_banner = ""
        self.tokens = []
        self.records = []
        self.nested_transmits = []   # 嵌套 TRANSMIT 子块（Character1 数组）
        self.state_segments = []   # 多 id 魔数声明的表内分段（字节区间）
        self.embedded_states = []  # 表内嵌 "CD-adapco_STAR-CCM+_ID<对象id>" 标记
        self.objects = []
        self.roots = []
        self.children = {}
        self.objmap = {}
        self._storage_pos = {}   # G3：存储键(数组 s0/ps) -> 数组记录 缓存

        for d, payload, s0, ps in self.sections:
            if isinstance(d, dict) and d.get("ClassName") == "Array":
                self.arrays.append({
                    "index": len(self.arrays),
                    "type": d.get("Type"),
                    "count": d.get("nElements"),
                    "start": ps,
                    "header_start": s0,
                    "data": payload,
                    "dict": d,
                })
                if d.get("Type") == "Character1":
                    ctext = payload.decode("latin-1")
                    if self.state_text is None:
                        self.state_text = ctext
                        self.tokens, self.records, self.state_magic, self.state_banner = \
                            parse_state_table(self.state_text)
                        self.state_mode = "binary" if "\x00" in \
                            "\n".join(self.state_text.split("\n")[5:20]) else "ascii"
                        self._parse_state_segments()
                    elif "TRANSMIT FILE created by" in ctext:
                        # 嵌套 TRANSMIT 子块（每块一个子状态表）
                        try:
                            nt_tok, nt_rec, nt_magic, nt_banner = parse_state_table(ctext)
                            self.nested_transmits.append({
                                "array_index": len(self.arrays),
                                "count": d.get("nElements"),
                                "magic": nt_magic, "banner": nt_banner.strip(),
                                "records": nt_rec,
                            })
                        except Exception as _e:
                            self.nested_transmits.append({
                                "array_index": len(self.arrays),
                                "count": d.get("nElements"),
                                "magic": [], "banner": "", "records": [],
                                "error": repr(_e),
                            })

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
        self._find_embedded_states()  # 需要 objmap，放在对象图构建之后

    # ---- 便捷访问 ----
    def object_by_id(self, oid):
        return self.objmap.get(oid)

    # ---- 边界 → 面片映射 ----
    def _parts_with_mesh(self):
        """[(name, part_id, face_array_index, patch_array, faces)]，按 part 分组。

        每个 part 有自己的面索引数组（count==TriangleCount×3）与每面 patch id 数组
        （Integer4、count==面数、小范围 patch 号）。
        """
        if _np is None:
            return []
        tris = {}
        for o in self.objects:
            t = o.dict.get("TriangleCount")
            if isinstance(t, int) and t > 0:
                tris.setdefault(t, []).append(o)
        parts = []
        for t, objs in sorted(tris.items(), reverse=True):
            fi = None
            for i, a in enumerate(self.arrays):
                if a["type"] in ("Unsigned4", "Integer4") and a["count"] == t * 3:
                    fi = i
                    break
            if fi is None:
                continue
            nf = t
            patch = None
            for i, a in enumerate(self.arrays):
                if a["type"] != "Integer4" or a["count"] != nf:
                    continue
                d = self.array_data(i)
                if d.size and int(d.min()) >= 1 and int(d.max()) <= max(1000, nf * 4):
                    patch = d
                    break
            obj = objs[0]
            parts.append({"name": obj.name or "part%d" % obj.id, "id": obj.id,
                          "triangles": t, "face_array": fi, "patch": patch})
        return parts

    def part_surface_patches(self):
        """part 表面 patch 表：per-face patch id 数组（按 part 分组）。

        返回 {"parts": [{name, id, triangles, face_array, patch: {patch_id: count}}],
              "boundary_refs": {boundary_name: [patch_ids]}}。
        """
        out = {"parts": [], "boundary_refs": {}}
        from collections import Counter
        for p in self._parts_with_mesh():
            rec = {"name": p["name"], "id": p["id"], "triangles": p["triangles"],
                   "face_array": p["face_array"], "patch": {}}
            if p["patch"] is not None:
                rec["patch"] = dict(Counter(int(x) for x in p["patch"].tolist()))
            out["parts"].append(rec)
        for o in self.objects:
            if o.class_name == "star.common.Boundary":
                ps = self.objmap.get(o.dict.get("PartSurfaces") or -1)
                ids = []
                for k in (ps.dict.get("Keys") or []) if ps is not None else []:
                    p = self.objmap.get(k)
                    if p is not None and isinstance(p.dict.get("Index"), int):
                        ids.append(p.dict["Index"])
                if ids:
                    out["boundary_refs"][o.name or "boundary%d" % o.id] = ids
        return out

    def boundary_faces(self):
        """每个 Boundary 的面索引列表（1 基）。

        规则（按 part 分组）：Boundary → PartSurfaces → PartSurface.Index →
        该 part 的每面 patch id 数组匹配。若一个边界跨多 part，逐 part 累加。
        返回 {"by_boundary": {name: {part: [face_index]}}, "total": n_faces}。
        """
        parts = self._parts_with_mesh()
        if not parts:
            return None
        br = self.part_surface_patches()["boundary_refs"]
        by_boundary = {}
        total = 0
        for p in parts:
            if p["patch"] is None:
                continue
            total += len(p["patch"])
            ids = set(int(x) for x in p["patch"].tolist())
            for bname, bpatch in br.items():
                matched = [i + 1 for i, x in enumerate(int(v) for v in p["patch"].tolist())
                           if x in bpatch]
                if matched:
                    by_boundary.setdefault(bname, {})[p["name"]] = matched
        return {"by_boundary": by_boundary, "total": total}

    def _parse_state_segments(self):
        """多 id 魔数（N>1）声明的表内分段：长度为"banner+表体"折行文本的字节区间。

        外层状态表 = 分段 0（主块：banner + 记录流）+ 分段 1..N-1（记录流的物理
        延续/内嵌子状态）。分段长度按折行后的原始字节计。
        """
        m = self.state_magic
        if not (len(m) >= 4 and m[0].startswith("CD-adapco") and m[-1] == "@"):
            return
        try:
            n = int(m[1])
        except ValueError:
            return
        vals = [int(x) for x in m[2:-1]]
        if n < 2 or len(vals) != n:
            return
        lines = self.state_text.split("\n")
        bi = next((k for k, l in enumerate(lines) if "TRANSMIT FILE" in l), None)
        if bi is None:
            return
        body = "\n".join(lines[bi:])  # banner + 表体（折行原文）
        pos = 0
        for k, v in enumerate(vals):
            chunk = body[pos:pos + v]
            self.state_segments.append({
                "index": k, "length": v, "offset": pos,
                "kind": "main" if k == 0 else "continuation",
                "preview": chunk[:60].replace("\n", "|"),
            })
            pos += v

    def _find_embedded_states(self):
        """表内嵌标记 'CD-adapco_STAR-CCM+_ID<对象id>'：其后数值为该对象内嵌
        状态的分段长度。把 id 解析到对象图（ClassName/name）。"""
        for rec in self.records:
            if rec.get("kind") == "named" and rec.get("name") == "CD-adapco_STAR-CCM+_ID":
                oid = rec.get("id")
                obj = self.objmap.get(oid) if isinstance(oid, int) else None
                self.embedded_states.append({
                    "object_id": oid,
                    "class": obj.class_name if obj else None,
                    "name": obj.name if obj else None,
                    "lengths": [v["value"] for v in rec.get("values", [])],
                    "token_index": rec.get("token_index"),
                })

    def layer_census(self):
        """按语义层统计对象（dict: layer -> count），含命名对象清单。"""
        from collections import Counter, defaultdict
        census = Counter()
        named = defaultdict(list)
        for o in self.objects:
            census[o.layer] += 1
            if o.name and o.layer in ("core", "cad-geometry", "meshing", "visualization",
                                      "post-processing", "physics", "materials",
                                      "co-simulation", "motion", "automation"):
                named[o.layer].append((o.id, o.class_name, o.name))
        return census, named

    def version_fingerprint(self):
        """版本指纹：banner 版本 / StarVersion / 状态表编码 / 头部字段组合。"""
        import re as _re
        m = _re.search(r"version (\d+)", self.state_banner or "")
        banner_ver = m.group(1) if m else None
        release = None
        for d, payload, s0, ps in self.sections:
            if isinstance(d, dict) and d.get("ClassName") == "StarVersion":
                release = d.get("ReleaseNumber")
                break
        hdr = sorted(k for k in self.header.keys() if k != "ClassName")
        return {"banner_version": banner_ver, "release": release,
                "state_mode": self.state_mode, "header_keys": hdr}

    def check_state_length(self):
        """状态表长度自校验（魔数值 vs 实际表长）。

        规则（语料验证）:
          - 外层标准魔数 N==1：value == 状态表总长 - 魔数块长 - 1
          - 多 id 魔数 N>1：sum(values) == banner+表体折行原文长
          - 嵌套块：value == 块长 - 10
        返回 {"ok": bool, "detail": str}。
        """
        m = self.state_magic
        if not m:
            return {"ok": None, "detail": "无魔数"}
        lines = self.state_text.split("\n")
        bi = next((k for k, l in enumerate(lines) if "TRANSMIT FILE" in l), None)
        if bi is None:
            return {"ok": None, "detail": "无 banner"}
        body_len = len("\n".join(lines[bi:]))
        if m[0].startswith("CD-adapco"):
            try:
                n = int(m[1])
                vals = [int(x) for x in m[2:-1]]
            except ValueError:
                return {"ok": None, "detail": "魔数解析失败"}
            if n == 1 and len(vals) == 1:
                expect = len(self.state_text) - len("\n".join(m)) - 1
                ok = vals[0] == expect
                return {"ok": ok, "detail": "外层 N=1: 魔数 %d vs 实际 %d" % (vals[0], expect)}
            if n >= 2 and len(vals) == n:
                ok = sum(vals) == body_len
                return {"ok": ok, "detail": "多 id: sum(%d 段)=%d vs 表体 %d" % (
                    n, sum(vals), body_len)}
            return {"ok": None, "detail": "魔数形态未识别"}
        # 嵌套块：3 行魔数（count, size, @）或嵌套多 id 形态——通用规则同外层：
        # 单值: value == 块总长 - 魔数块长 - 1；多值: sum(values) == banner+表体长
        if m[-1] == "@" and len(m) >= 3:
            try:
                n = int(m[0])
                vals = [int(x) for x in m[1:-1]]
            except ValueError:
                return {"ok": None, "detail": "嵌套魔数解析失败"}
            if n >= 2 and len(vals) == n:
                ok = sum(vals) == body_len
                return {"ok": ok, "detail": "嵌套多 id: sum(%d 段)=%d vs 表体 %d" % (
                    n, sum(vals), body_len)}
            expect = len(self.state_text) - len("\n".join(m)) - 1
            ok = vals[0] == expect
            return {"ok": ok, "detail": "嵌套: 魔数 %d vs 实际 %d" % (vals[0], expect)}
        return {"ok": None, "detail": "未识别"}

    def semantic_report(self):
        """语义层报告：Region↔Boundary↔Part↔网格、Continuum↔Models、Scene↔Displayer。

        基于对象图引用关系（Parent/Keys/属性），输出可读的仿真配置摘要。
        """
        om = self.objmap
        rep = {"regions": [], "continua": [], "scenes": [], "parts": []}

        def ref(v):
            if isinstance(v, int) and v in om:
                return om[v]
            return None

        def kids_of(o):
            return self.children.get(o.id, [])

        # Regions 及其边界/部件
        for o in self.objects:
            if o.class_name == "star.common.Region":
                bm = ref(o.dict.get("BoundaryManager"))
                boundaries = []
                if bm is not None:
                    for b in kids_of(bm):
                        if b.class_name == "star.common.Boundary":
                            boundaries.append({"id": b.id, "name": b.name})
                parts = []
                pv = o.dict.get("Parts")
                plist = pv if isinstance(pv, list) else [pv]
                for p in plist:
                    po = ref(p)
                    if po is None:
                        continue
                    # PartGroup 容器 → 展开其 Keys 得到实际部件
                    if "PartGroup" in (po.class_name or ""):
                        for k in po.dict.get("Keys") or []:
                            pk = ref(k)
                            if pk is not None:
                                parts.append({"id": pk.id, "name": pk.name,
                                              "class": pk.class_name,
                                              "triangles": pk.dict.get("TriangleCount")})
                    else:
                        parts.append({"id": po.id, "name": po.name,
                                      "class": po.class_name,
                                      "triangles": po.dict.get("TriangleCount")})
                rep["regions"].append({"id": o.id, "name": o.name,
                                       "boundaries": boundaries, "parts": parts})
        # Continuums 及其模型
        for o in self.objects:
            if o.class_name == "star.common.PhysicsContinuum":
                models = []
                mm = ref(o.dict.get("ModelManager"))
                if mm is not None:
                    for m2 in kids_of(mm):
                        models.append({"id": m2.id, "class": m2.class_name,
                                       "name": m2.name})
                rep["continua"].append({"id": o.id, "name": o.name,
                                        "models": models})
        # Scenes 及其显示器
        for o in self.objects:
            if o.class_name == "star.vis.Scene":
                dm = ref(o.dict.get("DisplayerManager"))
                displayers = []
                if dm is not None:
                    for k in dm.dict.get("Keys") or []:
                        d2 = ref(k)
                        if d2 is not None and (d2.class_name or "").startswith("star.vis"):
                            displayers.append({"id": d2.id, "class": d2.class_name,
                                               "name": d2.name})
                view = ref(o.dict.get("CurrentView"))
                rep["scenes"].append({"id": o.id, "name": o.name,
                                      "displayers": displayers,
                                      "view": (view.class_name, view.name) if view else None})
        # 所有命名 Part（网格规模）
        for o in self.objects:
            if isinstance(o.dict.get("TriangleCount"), int) and o.name:
                rep["parts"].append({"id": o.id, "class": o.class_name,
                                     "name": o.name,
                                     "triangles": o.dict["TriangleCount"],
                                     "vertices": o.dict.get("VertexCount")})
        return rep

    def validate_class_versions(self):
        """用文件尾部 ClassVersions 统计校验对象图（诊断性比对）。

        ClassVersions = {'ClassName': 'ClassVersions', 'Versions': {类名: 实例数}}，
        是写入方 C++ 类注册表在保存时刻的快照（含未序列化的运行时类，如
        DynamicLoader），与序列化对象图的精确类计数不完全等同——按诊断信息处理：
        - matched: 两边计数一致的类数
        - coverage: 期望计数之和 / 图内对象数
        - mismatches: 差异最大的若干类
        """
        from collections import Counter
        if not (self.objects and self.objects[-1].class_name == "ClassVersions"):
            return {"status": "no-ClassVersions"}
        expected = self.objects[-1].dict.get("Versions") or {}
        actual = Counter(o.class_name for o in self.objects[:-1])
        matched = 0
        checked = 0
        mismatches = {}
        for cn, n in expected.items():
            if not isinstance(n, int):
                continue
            checked += 1
            got = actual.get(cn, 0)
            if got == n:
                matched += 1
            else:
                mismatches[cn] = {"expected": n, "actual": got}
        total_exp = sum(v for v in expected.values() if isinstance(v, int))
        return {"status": "diagnostic", "expected_classes": checked,
                "matched": matched, "expected_total": total_exp,
                "actual_total": len(self.objects) - 1,
                "mismatches": mismatches}

    def array_data(self, index):
        """解码数组（numpy 可用时返回 ndarray，否则返回 list/bytes）。"""
        a = self.arrays[index]
        t = a["type"]
        if _np is not None and t in ("Float8", "Float", "Float4", "Unsigned4",
                                          "Integer4", "Integer8"):
            dt = {"Float8": "<f8", "Float": "<f4", "Float4": "<f4",
                  "Unsigned4": "<u4", "Integer4": "<i4", "Integer8": "<i8"}[t]
            return _np.frombuffer(a["data"], dtype=dt)
        if t == "Character1":
            return a["data"].decode("latin-1")
        return a["data"]

    # ---- 网格抽取 ----
    def mesh_metadata(self):
        """对象图里的网格规模元数据：各 part 的 TriangleCount / VertexCount。"""
        meta = {"TriangleCount": [], "VertexCount": [], "sources": []}
        for o in self.objects:
            for k in ("TriangleCount", "VertexCount"):
                v = o.dict.get(k)
                if isinstance(v, int):
                    meta[k].append(v)
                    meta["sources"].append((o.class_name, o.name, k, v))
        return meta

    def extract_mesh(self):
        """从数组表抽取网格（点/面），并用对象图元数据交叉校验。

        启发式:
          - 面索引 = Unsigned4/Integer4 数组，count == 某 part 的 TriangleCount×3
            （优先最大 part）；否则取最大的 count%3==0 的 U4/I4 数组。
          - 顶点坐标 = Float8 数组，count/3 优先等于面索引跨度 (max-min+1)
            （分块偏移编号），其次 1 基 max / 0 基 max+1；否则取 count%3==0
            且数值绝对值 >1 的最大 Float8 数组（法向量数组的值 ≤1，可区分）。
        返回 dict: vertices(N,3), faces(M,3)（0 基）, meta, flags(推断置信度)。
        """
        if _np is None:
            raise RuntimeError("网格抽取需要 numpy")
        meta = self.mesh_metadata()
        verts = faces = None
        vflags = fflags = ""
        # 面（按 part 的 TriangleCount×3 匹配，从大到小）
        tris = sorted(meta["TriangleCount"], reverse=True)
        for t in tris:
            for i, a in enumerate(self.arrays):
                if a["type"] in ("Unsigned4", "Integer4") and a["count"] == t * 3:
                    faces = self.array_data(i).reshape(-1, 3).astype("int64")
                    fflags = "part:%d" % t
                    break
            if faces is not None:
                break
        if faces is None:
            cands2 = [(i, a) for i, a in enumerate(self.arrays)
                      if a["type"] in ("Unsigned4", "Integer4") and a["count"] % 3 == 0]
            if cands2:
                i, a = max(cands2, key=lambda c: c[1]["count"])
                faces = self.array_data(i).reshape(-1, 3).astype("int64")
                fflags = "heuristic"
        # 顶点（跨度优先，再按面索引最大值确定规模）
        cands = [(i, a) for i, a in enumerate(self.arrays) if a["type"] == "Float8"]
        if faces is not None and faces.size:
            mx = int(faces.max())
            lo = int(faces.min())
            span = mx - lo + 1
            wants = []
            for want in (span, mx, mx + 1):
                if want > 0 and want not in wants:
                    wants.append(want)
            for want in wants:
                for i, a in cands:
                    if a["count"] == want * 3:
                        verts = self.array_data(i).reshape(-1, 3)
                        vflags = "face-span:%d" % want if want == span else "face-max:%d" % want
                        break
                if verts is not None:
                    break
        if verts is None:
            big = [c for c in cands if c[1]["count"] % 3 == 0 and
                   _np.abs(self.array_data(c[0])).max() > 1.0]
            if big:
                i, a = max(big, key=lambda c: c[1]["count"])
                verts = self.array_data(i).reshape(-1, 3)
                vflags = "heuristic"
        res = {"vertices": verts, "faces": faces, "meta": meta,
               "vertex_flag": vflags, "face_flag": fflags}
        if verts is not None and faces is not None:
            raw_max = int(faces.max()) if faces.size else 0
            faces, ok = compact_face_indices(faces, verts.shape[0])
            res["faces"] = faces
            res["max_index"] = raw_max
            res["n_vertices"] = int(verts.shape[0])
            res["consistent"] = bool(
                ok and faces.size and int(faces.min()) >= 0
                and int(faces.max()) < verts.shape[0])
            if not faces.size:
                res["consistent"] = True
        return res

    def export_stl(self, out_path):
        """把抽取出的网格写成 ASCII STL（三角形面片）。"""
        m = self.extract_mesh()
        if m["vertices"] is None or m["faces"] is None:
            raise RuntimeError("无法抽取网格（缺少顶点/面数组）")
        v, f = m["vertices"], m["faces"]
        idx, _ok = compact_face_indices(f, v.shape[0])
        with open(out_path, "w", encoding="ascii") as fh:
            fh.write("solid sim\n")
            for tri in idx:
                a, b, c = v[tri[0]], v[tri[1]], v[tri[2]]
                fh.write("  facet normal 0 0 0\n    outer loop\n")
                for p in (a, b, c):
                    fh.write("      vertex %.9g %.9g %.9g\n" % tuple(p))
                fh.write("    endloop\n  endfacet\n")
            fh.write("endsolid sim\n")
        return out_path

    def _storage_array(self, key):
        """把存储对象的键（dataKey/countKey/listKey...）解析为数组记录。

        支持两种形态（G3 侦察结论）：
          - 直接键：key 就是数组分区字典行的起始位置（s0）或载荷起始（ps）
            （pipeBlockage / airfoil / pipeMixingBlockage 等新版文件）；
          - 对象链：key 是 MasterArray/ScatterableArray 对象 id，其 dataKey/
            sizesKey 指向的「指针数组」（Unsigned4 n=1 / Integer8 n=1）的
            载荷值才是真实数据数组的 s0（methaneOnPt 等旧版文件）。
        解析失败返回 None。
        """
        if not isinstance(key, int):
            return None
        pos = self._storage_pos
        if not pos:
            arr_by_ps = {}
            for a in self.arrays:
                s = a.get("start")
                if isinstance(s, int):
                    arr_by_ps[s] = a
            for _d, _p, s0, ps in self.sections:
                if isinstance(_d, dict) and _d.get("ClassName") == "Array":
                    a = arr_by_ps.get(ps)
                    if a is not None:
                        pos.setdefault(s0, a)
            for a in self.arrays:
                s = a.get("start")
                if isinstance(s, int):
                    pos.setdefault(s, a)
        arr = pos.get(key)
        if arr is not None:
            return arr
        obj = self.objmap.get(key)
        if obj is not None and (obj.class_name or "").startswith(
                ("MasterArray", "ScatterableArray")):
            for f in ("dataKey", "sizesKey"):
                k = obj.dict.get(f)
                if not isinstance(k, int):
                    continue
                pa = self._storage_array(k)
                if pa is None:
                    continue
                data = pa["data"]
                try:
                    if pa["type"] == "Integer8" and len(data) >= 8:
                        val = _np.frombuffer(data[:8], dtype="<i8")[0] if _np is not None \
                            else _struct.unpack("<q", data[:8])[0]
                    elif pa["type"] == "Unsigned4" and len(data) >= 4:
                        val = _np.frombuffer(data[:4], dtype="<u4")[0] if _np is not None \
                            else _struct.unpack("<I", data[:4])[0]
                    else:
                        continue
                except Exception:
                    continue
                return self._storage_array(int(val))
        return None

    def extract_volume_mesh(self):
        """抽取体网格：存储体系驱动（DuplicateStorageManager + 键解析）+ 面→单元反演。

        识别顶点/面/单元三个主存储组：
          - Coord(Float8×3)     -> 顶点坐标
          - VertexList(CSR)     -> 每面顶点环（count/offset + 扁平索引）
          - FaceCellIndex       -> 每面左右单元对（负值/0xFFFFFFFF=边界）
        由面邻接反演每个单元的节点集合（任意多面体），单元数精确等于存储组
        SerialSize。无存储体系/缺组/连通性失败 -> ok=False + 原因，不假装成功
        （替换旧版"猜最大整型数组"启发式——它曾把顶点标志表误判为四面体表）。

        返回 {ok, kind:"poly", count, points, face_verts, face_cells,
              cell_faces, cell_loops, cells, groups, reason[, elem_types]}
        """
        if _np is None:
            return {"ok": False, "kind": None, "count": 0, "reason": "需要 numpy"}
        dups = [o for o in self.objects
                if (o.class_name or "") == "DuplicateStorageManager"]
        if not dups:
            return {"ok": False, "kind": None, "count": 0,
                    "reason": "无 DuplicateStorageManager 存储体系（纯表面网格文件？）"}
        stor_by_id = {o.id: o for o in self.objects
                      if (o.class_name or "").startswith(("SimpleStorage", "ListStorage"))}

        def stor_in(d, tag):
            i = (d.dict.get("map") or {}).get(tag)
            return stor_by_id.get(i)

        vert_d = face_d = cell_d = None
        for d in dups:
            m = d.dict.get("map") or {}
            sz = d.dict.get("SerialSize") or 0
            if "Coord" in m:
                if vert_d is None or sz > (vert_d.dict.get("SerialSize") or 0):
                    vert_d = d
            elif "FaceCellIndex" in m and "VertexList" in m:
                if not any(t in m for t in
                           ("FacePartSurfaceIndex", "ProstarBounId", "ProstarFaceId")):
                    if face_d is None or sz > (face_d.dict.get("SerialSize") or 0):
                        face_d = d
            elif any(t in m for t in
                     ("ElemType", "ProstarCellIndex", "ProstarCellType",
                      "CellGeometryPartIndex", "WallDistance", "PrismLayerCells")):
                if cell_d is None or sz > (cell_d.dict.get("SerialSize") or 0):
                    cell_d = d

        missing = "、".join(n for d, n in (
            (vert_d, "顶点"), (face_d, "面"), (cell_d, "单元")) if d is None)
        if vert_d is None or face_d is None or cell_d is None:
            return {"ok": False, "kind": None, "count": 0,
                    "reason": "缺少体网格存储组：%s" % missing}

        nvert = int(vert_d.dict.get("SerialSize") or 0)
        nface = int(face_d.dict.get("SerialSize") or 0)
        ncell = int(cell_d.dict.get("SerialSize") or 0)
        if nvert <= 0 or nface <= 0 or ncell <= 0:
            return {"ok": False, "kind": None, "count": 0,
                    "reason": "存储组实体数为空（%d/%d/%d）" % (nvert, nface, ncell)}

        def first_key(o, *fields):
            for f in fields:
                v = o.dict.get(f)
                if isinstance(v, int):
                    return v
                if isinstance(v, list) and v:
                    return v[0]
            return None

        coord_o = stor_in(vert_d, "Coord")
        vl_o = stor_in(face_d, "VertexList")
        fci_o = stor_in(face_d, "FaceCellIndex")
        et_o = stor_in(cell_d, "ElemType")
        if coord_o is None or vl_o is None or fci_o is None:
            return {"ok": False, "kind": None, "count": 0,
                    "reason": "主组缺少 Coord/VertexList/FaceCellIndex 存储"}

        arr_coord = self._storage_array(first_key(coord_o, "dataKey", "dataKeys"))
        arr_vl_c = self._storage_array(first_key(vl_o, "countKey", "offsetKeys"))
        arr_vl_l = self._storage_array(first_key(vl_o, "listKey", "listKeys"))
        arr_fci = self._storage_array(first_key(fci_o, "dataKey", "dataKeys"))
        if arr_coord is None or arr_vl_c is None or arr_vl_l is None or arr_fci is None:
            return {"ok": False, "kind": None, "count": 0,
                    "reason": "存储键解析失败（未命中数组块）"}

        try:
            pts = _np.frombuffer(arr_coord["data"], dtype="<f8")
            if pts.size >= 3 * nvert:
                pts = pts[: 3 * nvert].reshape(-1, 3)
            else:
                pts = _np.pad(pts, (0, 3 * nvert - pts.size)).reshape(-1, 3)
        except Exception as _e:
            return {"ok": False, "kind": None, "count": 0,
                    "reason": "顶点坐标解码失败：%r" % _e}

        vcounts = _np.frombuffer(arr_vl_c["data"], dtype="<u4")
        vlist = _np.frombuffer(arr_vl_l["data"], dtype="<u4")
        if vcounts.size >= nface:
            vcounts = vcounts[:nface]
        else:
            vcounts = _np.pad(vcounts, (0, nface - vcounts.size))
        if int(vcounts.sum()) != vlist.size:
            # 旧式 CSR 变体（methaneOnPt）：offset 数组是前缀和（含 0、单调递增、
            # 无尾哨兵）。转成每面顶点数再校验。
            if (vcounts.size >= 2 and int(vcounts[0]) == 0
                    and int(vcounts[-1]) < vlist.size
                    and bool((_np.diff(vcounts.astype("<i8")) > 0).all())):
                last_off = int(vcounts[-1])
                vcounts = _np.diff(vcounts.astype("<i8"))
                vcounts = _np.append(vcounts, vlist.size - last_off)
        if int(vcounts.sum()) != vlist.size:
            return {"ok": False, "kind": None, "count": 0,
                    "reason": "面顶点数总和不等于索引表长度（%d != %d）"
                             % (int(vcounts.sum()), vlist.size)}
        fci = _np.frombuffer(arr_fci["data"], dtype="<u4").astype("<i8")
        fci = fci[: 2 * nface]
        fci[fci >= 0x80000000] = -1   # 0xFFFFFFFF（无符号存储）-> 边界

        # ---- 面→单元反演 ----
        offs = _np.concatenate(([0], _np.cumsum(vcounts))).astype("<i8")
        cell_faces = [[] for _ in range(ncell)]
        for fi in range(nface):
            a, b = int(fci[2 * fi]), int(fci[2 * fi + 1])
            for c in (a, b):
                if 0 <= c < ncell:
                    cell_faces[c].append(fi)
        orphan = sum(1 for fs in cell_faces if not fs)
        if orphan:
            return {"ok": False, "kind": None, "count": 0,
                    "reason": "拓扑反演失败：%d/%d 单元无面" % (orphan, ncell)}

        if vlist.size and int(vlist.max()) >= nvert:
            return {"ok": False, "kind": None, "count": 0,
                    "reason": "面顶点下标超出顶点表（%d >= %d）"
                             % (int(vlist.max()), nvert)}
        cells, cell_loops = [], []
        for fs in cell_faces:
            nodes, loops = [], []
            for fi in fs:
                loop = vlist[offs[fi]:offs[fi + 1]].tolist()
                loops.append(loop)
                for v in loop:
                    if v not in nodes:
                        nodes.append(v)
            cells.append(nodes)
            cell_loops.append(loops)

        groups = {"verts": str(vert_d.dict.get("groupTag")),
                  "faces": str(face_d.dict.get("groupTag")),
                  "cells": str(cell_d.dict.get("groupTag"))}
        extra = {}
        if et_o is not None:
            arr_et = self._storage_array(first_key(et_o, "dataKey", "dataKeys"))
            if arr_et is not None:
                txt = arr_et["data"].decode("latin-1")
                if len(txt) >= ncell:
                    extra["elem_types"] = txt[:ncell]
        return {"ok": True, "kind": "poly", "count": ncell,
                "points": pts, "face_verts": vlist, "face_cells": fci,
                "cell_faces": cell_faces, "cell_loops": cell_loops,
                "cells": cells, "groups": groups, "reason": "", **extra}

    def export_volume_vtu(self, path, vol=None):
        """把体网格写为 VTK XML UnstructuredGrid（VTK_POLYHEDRON 任意多面体）。

        单元以"面环"定义（每个面一个顶点索引环），ParaView 可直接打开。
        抽取失败返回 None，成功返回 path。
        """
        if vol is None:
            vol = self.extract_volume_mesh()
        if not vol.get("ok"):
            return None
        pts = vol.get("points")
        loops = vol.get("cell_loops")
        if pts is None or not loops:
            return None
        npt = int(pts.shape[0])
        ncell = int(vol.get("count") or len(loops))
        conn, offs = [], []
        acc = 0
        for cell in loops:
            conn.append(len(cell))          # 面数
            for loop in cell:
                conn.append(len(loop))      # 该面顶点数
                conn.extend(int(v) for v in loop)
            acc += 1 + sum(1 + len(l) for l in cell)
            offs.append(acc)
        L = []
        L.append('<?xml version="1.0"?>')
        L.append('<VTKFile type="UnstructuredGrid" version="1.0" '
                 'byte_order="LittleEndian" header_type="UInt64">')
        L.append('  <UnstructuredGrid>')
        L.append('    <Piece NumberOfPoints="%d" NumberOfCells="%d">' % (npt, ncell))
        L.append('      <Points>')
        L.append('        <DataArray type="Float64" Name="Points" '
                 'NumberOfComponents="3" format="ascii">')
        for x, y, z in pts:
            L.append("          %.10g %.10g %.10g" % (x, y, z))
        L.append('        </DataArray>')
        L.append('      </Points>')
        L.append('      <Cells>')
        L.append('        <DataArray type="Int64" Name="connectivity" format="ascii">')
        L.append("          " + " ".join(str(v) for v in conn))
        L.append('        </DataArray>')
        L.append('        <DataArray type="Int64" Name="offsets" format="ascii">')
        L.append("          " + " ".join(str(v) for v in offs))
        L.append('        </DataArray>')
        L.append('        <DataArray type="UInt8" Name="types" format="ascii">')
        L.append("          " + " ".join(["41"] * ncell))
        L.append('        </DataArray>')
        L.append('      </Cells>')
        L.append('    </Piece>')
        L.append('  </UnstructuredGrid>')
        L.append('</VTKFile>')
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(L) + "\n")
        return path

    # ---- G4：边界 ↔ 面片映射 ----
    def _storage_payload(self, so):
        """存储对象载荷：SimpleStorage → ndarray；ListStorage → (counts/offsets, values)。"""
        d = (so.dict or {}) if so is not None else {}
        if (so.class_name or "").startswith("SimpleStorage"):
            k = d.get("dataKey", d.get("dataKeys"))
            if isinstance(k, list):
                k = k[0] if k else None
            if not isinstance(k, int):
                return None
            a = self._storage_array(k)
            if a is None:
                return None
            v = self.array_data(a["index"])
            return v if isinstance(v, _np.ndarray) else None
        lk = d.get("listKey", d.get("listKeys"))
        if isinstance(lk, list):
            lk = lk[0] if lk else None
        if not isinstance(lk, int):
            return None
        a = self._storage_array(lk)
        if a is None:
            return None
        vals = self.array_data(a["index"])
        if not isinstance(vals, _np.ndarray):
            return None
        for ck in ("offsetKey", "countKey", "offsetKeys", "countKeys"):
            c = d.get(ck)
            if isinstance(c, list):
                c = c[0] if c else None
            if isinstance(c, int):
                ca = self._storage_array(c)
                if ca is not None:
                    cv = self.array_data(ca["index"])
                    if isinstance(cv, _np.ndarray):
                        return (cv, vals)
        return (None, vals)

    def _dup_rings(self, m, stor_by_id):
        """DUP 组 map → (face_cell 对, [每面顶点环])；解析失败返回 (None, None)。"""
        fci = self._storage_payload(stor_by_id.get((m or {}).get("FaceCellIndex")))
        if not isinstance(fci, _np.ndarray):
            return None, None
        pairs = fci.astype(_np.int64).reshape(-1, 2)
        vl = self._storage_payload(stor_by_id.get((m or {}).get("VertexList")))
        if not isinstance(vl, tuple):
            return pairs, None
        cnts, vlist = vl
        if cnts is None:
            cnts = _np.full(vlist.size // 4, 4, dtype=_np.int64)
        else:
            cnts = _np.asarray(cnts).astype(_np.int64)
            if int(cnts.sum()) != vlist.size and cnts.size >= 2 \
                    and int(cnts[0]) == 0 and bool((_np.diff(cnts) > 0).all()):
                last = int(cnts[-1])
                cnts = _np.diff(cnts)
                cnts = _np.append(cnts, vlist.size - last)
        if int(cnts.sum()) != vlist.size:
            return pairs, None
        offs = _np.concatenate(([0], _np.cumsum(cnts)))
        return pairs, [vlist[offs[i]:offs[i + 1]].tolist()
                       for i in range(cnts.size)]

    def extract_boundary_faces(self, vol=None):
        """G4：FvBoundary → Boundary 对象链抽取体网格边界面片（按边界分组）。

        语义（G4 侦查结论）：
          - FvBoundary.Boundary → star.common.Boundary（名字/Region/Index）
          - FvBoundary.faces → BDY DuplicateStorageManager；
            FaceCount == SerialSize == FaceCellIndex 面数 == VertexList 环数
          - BDY FaceCellIndex = (owner_cell, 占位)：owner-only 挂载
            （pipeBlockage 14882/14882 单元边闭包 0 坏边的决定性验证）
          - FacePartSurfaceIndex（可选常量）== Boundary.PartSurfaces 组 Keys 所指
            PartSurface.Index（airfoil 全局 47..55；pipeBlockage 1:1 场景与
            Boundary.Index 巧合相等）；methaneOnPt 型无此通道（ProstarBounId
            代替），靠 FvBoundary 链兜底
        返回 {"ok", "total_faces", "boundaries", "reason"}；体网格抽取失败的
        文件（直升机/纯表面网格）诚实拒绝 ok=False。
        """
        if vol is None:
            vol = self.extract_volume_mesh()
        if not vol.get("ok"):
            return {"ok": False, "total_faces": 0, "boundaries": [],
                    "reason": vol.get("reason") or "体网格未抽取"}
        ncell = int(vol.get("count") or 0)
        nvert = int(vol["points"].shape[0])
        stor_by_id = {o.id: o for o in self.objects
                      if (o.class_name or "").startswith(
                          ("SimpleStorage", "ListStorage"))}
        dups = {o.id: o for o in self.objects
                if (o.class_name or "") == "DuplicateStorageManager"}
        out = []
        for fb in self.objects:
            if not (fb.class_name or "").endswith(".FvBoundary"):
                continue
            dup = dups.get(fb.dict.get("faces"))
            bobj = self.objmap.get(fb.dict.get("Boundary"))
            if dup is None or bobj is None:
                continue
            m = dup.dict.get("map") or {}
            pairs, rings = self._dup_rings(m, stor_by_id)
            sz = int(dup.dict.get("SerialSize") or 0)
            if pairs is None or rings is None or len(rings) != pairs.shape[0] \
                    or pairs.shape[0] != sz \
                    or sz != int(fb.dict.get("FaceCount") or -1):
                continue
            owners = pairs[:, 0]
            if owners.size and (int(owners.min()) < 0
                                or int(owners.max()) >= ncell):
                continue
            vmax = max((max(r) for r in rings if r), default=-1)
            if vmax >= nvert:
                continue
            pay = {t: self._storage_payload(stor_by_id.get(sid))
                   for t, sid in m.items()}

            def const_of(tag):
                a = pay.get(tag)
                if isinstance(a, _np.ndarray) and a.size:
                    u = _np.unique(a)
                    if u.size == 1:
                        return int(u[0])
                return None

            robj = self.objmap.get(bobj.dict.get("Region"))
            ps_group = self.objmap.get(bobj.dict.get("PartSurfaces"))
            ps_infos = []
            for k in ((ps_group.dict or {}).get("Keys") or []) if ps_group else []:
                o = self.objmap.get(k)
                if o is not None and (o.class_name or "").endswith(".PartSurface"):
                    ps_infos.append({"id": o.id, "name": o.name,
                                     "index": o.dict.get("Index")})
            out.append({"id": bobj.id,
                        "name": bobj.name or "boundary%d" % bobj.id,
                        "region_id": robj.id if robj is not None else None,
                        "region_name": (robj.name or "") if robj is not None else "",
                        "index": bobj.dict.get("Index"),
                        "face_count": sz,
                        "owner_cells": owners.tolist(),
                        "rings": rings,
                        "part_surface_index": const_of("FacePartSurfaceIndex"),
                        "part_surfaces": ps_infos,
                        "prostar_boun_id": const_of("ProstarBounId")})
        if not out:
            return {"ok": True, "total_faces": 0, "boundaries": [],
                    "reason": "无 FvBoundary 边界链"}
        out.sort(key=lambda b: (str(b["region_name"]),
                                str(b["index"]) if b["index"] is not None else ""))
        return {"ok": True, "total_faces": sum(b["face_count"] for b in out),
                "boundaries": out, "reason": ""}

    def export_boundary_csv(self, path, bf=None):
        """G4 边界映射 → CSV（每边界一行：面数/区域/占位通道常量）。"""
        if bf is None:
            bf = self.extract_boundary_faces()
        if not bf.get("ok"):
            return None
        with open(path, "w", encoding="utf-8-sig", newline="") as fh:
            fh.write("boundary_id,boundary_name,region_id,region_name,"
                     "index,face_count,part_surface_index,part_surface,"
                     "prostar_boun_id\n")
            for b in bf["boundaries"]:
                ps = b.get("part_surfaces") or []
                fh.write("%d,%s,%s,%s,%s,%d,%s,%s,%s\n" % (
                    b["id"], b["name"], b["region_id"], b["region_name"],
                    b["index"], b["face_count"], b["part_surface_index"],
                    ";".join(p["name"] for p in ps) if ps else "",
                    b["prostar_boun_id"]))
        return path

    def extract_solution_fields(self):
        """G5：抽取 .sim 内嵌解场（单元中心标量/矢量）。

        语义链（G5 侦查结论）：
          - star.post.SolutionRepresentation 携带 FunctionNames 解字段清单，
            经 Objects → star.common.TypedObjectManager →
            star.common.FvRegionManager → star.common.FvRegion 到达数据；
          - FvRegion.cells → DuplicateStorageManager（SerialSize == CellCount），
            map = {字段tag: SimpleStorage}（G3 存储体系直接复用）：
            标量 SimpleStorage<T> 数组 n == CellCount；矢量
            SimpleStorage<Vector<3,T>> n == 3*CellCount（每单元三分量）；
          - 解场 FvRegion 的 mesh 组（Coord/VertexList/FaceCellIndex）与
            extract_volume_mesh() 所用数组逐字节相同 → cell 序严格对齐，
            抽取结果可直接做单元标量着色；
          - 无解场文件（教程 *_start / 纯表面网格 / 未求解）诚实拒绝
            ok=False + reason。
        返回 {"ok", "n_fields", "cell_count", "region_name", "fields",
              "data", "reason"}；fields = [{name, components, count,
              min, max, type}]，data = {tag: ndarray}。
        """
        if _np is None:
            return {"ok": False, "n_fields": 0, "cell_count": 0,
                    "fields": [], "data": {}, "reason": "需要 numpy"}
        srs = [o for o in self.objects
               if (o.class_name or "") == "star.post.SolutionRepresentation"]
        if not srs:
            return {"ok": False, "n_fields": 0, "cell_count": 0,
                    "fields": [], "data": {}, "reason": "无解场表示（SolutionRepresentation）"}
        stor_by_id = {o.id: o for o in self.objects
                      if (o.class_name or "").startswith(
                          ("SimpleStorage", "ListStorage"))}
        dups = {o.id: o for o in self.objects
                if (o.class_name or "") == "DuplicateStorageManager"}
        out_data, out_fields = {}, []
        ncell = 0
        region = None
        for sr in srs:
            tmo = self.objmap.get(sr.dict.get("Objects"))
            if tmo is None:
                continue
            for mid in tmo.dict.get("Keys") or []:
                mgr = self.objmap.get(mid)
                if mgr is None or (mgr.class_name or "") != \
                        "star.common.FvRegionManager":
                    continue
                for rid in mgr.dict.get("Keys") or []:
                    fr = self.objmap.get(rid)
                    if fr is None or (fr.class_name or "") != \
                            "star.common.FvRegion":
                        continue
                    dup = dups.get(fr.dict.get("cells"))
                    if dup is None:
                        continue
                    ncell = int(dup.dict.get("SerialSize") or 0)
                    robj = self.objmap.get(fr.dict.get("Region"))
                    if robj is not None:
                        region = robj.name or robj.class_name
                    for tag, sid in (dup.dict.get("map") or {}).items():
                        if tag in out_data:
                            continue
                        so = stor_by_id.get(sid)
                        if so is None:
                            continue
                        cn = so.class_name or ""
                        if not cn.startswith("SimpleStorage"):
                            continue
                        v = self._storage_payload(so)
                        if not isinstance(v, _np.ndarray) or v.size == 0:
                            continue
                        comps = 3 if re.search(r"Vector<\d+", cn) else 1
                        if comps == 3:
                            if v.size != ncell * 3:
                                continue
                            v = v.reshape(ncell, 3)
                        elif v.size != ncell:
                            continue
                        out_data[tag] = v
                        out_fields.append({
                            "name": tag, "components": comps,
                            "count": int(v.shape[0]),
                            "min": float(v.min()), "max": float(v.max()),
                            "type": cn})
                    if out_fields:
                        break
                if out_fields:
                    break
            if out_fields:
                break
        if not out_fields:
            return {"ok": False, "n_fields": 0, "cell_count": ncell,
                    "fields": [], "data": {}, "reason": "解场 FvRegion 无字段存储"}
        return {"ok": True, "n_fields": len(out_fields), "cell_count": ncell,
                "region_name": region, "fields": out_fields,
                "data": out_data, "reason": ""}

    def export_solution_csv(self, path, sf=None, field=None):
        """G5 解场 → CSV（逐单元一行）。field 缺省取第一个标量字段。"""
        if sf is None:
            sf = self.extract_solution_fields()
        if not sf.get("ok"):
            return None
        if field is None:
            cand = [f for f in sf["fields"] if f["components"] == 1]
            field = cand[0]["name"] if cand else sf["fields"][0]["name"]
        v = sf["data"].get(field)
        if v is None or not getattr(v, "size", 0):
            return None
        comps = v.shape[1] if v.ndim == 2 else 1
        with open(path, "w", encoding="utf-8-sig", newline="") as fh:
            fh.write("cell,%s\n" % ",".join(
                "%s_%d" % (field, i) for i in range(comps)))
            if v.ndim == 2:
                for i in range(v.shape[0]):
                    fh.write("%d,%s\n" % (i, ",".join(
                        "%.6g" % x for x in v[i])))
            else:
                for i in range(v.size):
                    fh.write("%d,%.6g\n" % (i, v[i]))
        return path

    # ---- G6 绘图/监视器曲线 ----
    def _fix_name(self, s):
        """状态表按 latin-1 解码，UTF-8 中文名呈 mojibake；round-trip 还原。"""
        if not isinstance(s, str):
            return s
        try:
            b = s.encode("latin-1")
        except UnicodeEncodeError:
            return s
        try:
            return b.decode("utf-8")
        except UnicodeDecodeError:
            return s

    def _monitor_array(self, oid):
        """监视器数据引用 → Float8 数组（G6 语义链，兼容两代子格式）：
        XAxisData → MasterArray<Float8>(dataKey)；
        MultiYAxisData → PlotableMonitor.YAxisValues：
          新版 {values:[MasterArray id]}；旧版 {map:{YAxisData: MasterArray id}}。
        空数据（dataKey=0/size=0）返回 None。
        """
        o = self.objmap.get(oid)
        if o is None:
            return None
        vals = o.dict.get("values")
        if isinstance(vals, list) and vals:
            o = self.objmap.get(vals[0])
            if o is None:
                return None
        mp = o.dict.get("map")
        if isinstance(mp, dict) and isinstance(mp.get("YAxisData"), int):
            o = self.objmap.get(mp["YAxisData"])
            if o is None:
                return None
        a = self._storage_array(o.dict.get("dataKey"))
        if a is None or not a.get("count"):
            return None
        try:
            v = _np.frombuffer(a["data"], dtype="<f8")
        except Exception:
            return None
        return v if v.size else None

    def extract_monitor_curves(self):
        """G6：抽取监视器曲线数据（采样索引 + 值序列）。
        语义链（G6 侦查结论）：
          - MonitorManager.Keys → 监视器（ResidualMonitor/IterationMonitor/
            PhysicalTimeMonitor/ReportMonitor/SolutionView*Monitor）
          - 监视器.XAxisData → MasterArray<Float8>（dataKey→数组，记录采样迭代号）
          - 监视器.MultiYAxisData → PlotableMonitor.YAxisValues
            {values:[MasterArray id]} → 值序列
        物理语义：残差/迭代每步记录（n=总迭代数），物理时间/报告监视器按
        StarUpdate 间隔采样（n=总迭代/间隔，间隔=15 时 X 恒为 15,30,...）。
        无监视器管理器或全部监视器无数据（未求解 Xsize=0）诚实拒绝 ok=False。
        返回 {"ok", "monitors", "data", "reason"}；
        monitors=[{name,class,n,y_min,y_max,last,cur_value,index_first,index_last}]，
        data={name:{"index":arr|None,"y":arr}}
        """
        if _np is None:
            return {"ok": False, "monitors": [], "data": {}, "reason": "需要 numpy"}
        mm = next((o for o in self.objects
                   if (o.class_name or "") == "star.base.report.MonitorManager"),
                  None)
        if mm is None:
            return {"ok": False, "monitors": [], "data": {},
                    "reason": "无监视器管理器（MonitorManager）"}
        out_m, out_d = [], {}
        for mid in mm.dict.get("Keys") or []:
            m = self.objmap.get(mid)
            if m is None:
                continue
            y = self._monitor_array(m.dict.get("MultiYAxisData"))
            if y is None:
                continue
            x = self._monitor_array(m.dict.get("XAxisData"))
            name = self._fix_name(m.name) or ("Monitor%d" % m.id)
            aligned = x is not None and x.size == y.size
            entry = {"id": m.id, "name": name, "class": m.class_name,
                     "n": int(y.size),
                     "y_min": float(y.min()), "y_max": float(y.max()),
                     "last": float(y[-1]), "cur_value": m.dict.get("CurrentValue"),
                     "index_first": float(x[0]) if aligned else None,
                     "index_last": float(x[-1]) if aligned else None}
            out_m.append(entry)
            out_d[name] = {"index": x if aligned else None, "y": y}
        if not out_m:
            return {"ok": False, "monitors": [], "data": {},
                    "reason": "监视器无数据（未求解或未记录）"}
        return {"ok": True, "monitors": out_m, "data": out_d, "reason": ""}

    def extract_plots(self, mc=None):
        """G6：绘图 → 曲线序列关联（G2 标注：标题/轴标题/单位/图例）。
        语义链（G6 侦查结论）：
          PlotManager.Keys → Plot（Title；XAxis→Cartesian2DAxis→AxisTitle.Text；
          XUnits→Units 符号）→ DataSetManager.Keys →
            MonitorDataSet{SeriesName=图例, XAxisMonitor, YAxisMonitor}（监视器曲线）
            DerivedDataSet{InputData→TableDerivedData{XValuesName,YValuesName}
                           →FileTable{FileName}}（导入表数据，仅标注不取数）
        返回 {"ok", "plots", "reason"}；
        plots=[{title,class,x_title,x_units,series:[{kind,name,x_monitor,
        y_monitor,x_id,y_id,x_scale,x_offset,y_scale,y_offset,...}]}]
        """
        pm = next((o for o in self.objects
                   if (o.class_name or "") == "star.common.PlotManager"), None)
        if pm is None:
            return {"ok": False, "plots": [], "reason": "无绘图管理器（PlotManager）"}

        def axis_title(plt):
            ax = self.objmap.get(plt.dict.get("XAxis"))
            if ax is None:
                return None
            t = self.objmap.get(ax.dict.get("AxisTitle"))
            return self._fix_name(t.dict.get("Text")) if t else None

        def units_sym(oid):
            u = self.objmap.get(oid)
            return self._fix_name(u.name) if u is not None else None

        plots = []
        for pid in pm.dict.get("Keys") or []:
            p = self.objmap.get(pid)
            if p is None:
                continue
            series = []
            dsm = self.objmap.get(p.dict.get("DataSetManager"))
            for did in (dsm.dict.get("Keys") or []) if dsm else []:
                d = self.objmap.get(did)
                if d is None:
                    continue
                cn = d.class_name or ""
                s = {"kind": "monitor" if cn.endswith("MonitorDataSet")
                     else ("tabular" if cn.endswith("DerivedDataSet") else "other"),
                     "name": self._fix_name(d.dict.get("SeriesName") or d.name),
                     "x_monitor": None, "y_monitor": None,
                     "x_id": d.dict.get("XAxisMonitor"),
                     "y_id": d.dict.get("YAxisMonitor"),
                     "x_scale": d.dict.get("XScale", 1.0),
                     "x_offset": d.dict.get("XOffset", 0.0),
                     "y_scale": d.dict.get("YScale", 1.0),
                     "y_offset": d.dict.get("YOffset", 0.0)}
                xo = self.objmap.get(s["x_id"])
                yo = self.objmap.get(s["y_id"])
                if xo is not None:
                    s["x_monitor"] = self._fix_name(xo.name)
                if yo is not None:
                    s["y_monitor"] = self._fix_name(yo.name)
                if s["kind"] == "tabular":
                    td = self.objmap.get(d.dict.get("InputData"))
                    if td is not None and (td.class_name or "").endswith(
                            "TableDerivedData"):
                        s["x_column"] = self._fix_name(td.dict.get("XValuesName"))
                        s["y_column"] = self._fix_name(td.dict.get("YValuesName"))
                        ft = self.objmap.get(td.dict.get("InputData"))
                        if ft is not None and (ft.class_name or "").endswith(
                                "FileTable"):
                            s["table_file"] = self._fix_name(
                                ft.dict.get("FileName"))
                series.append(s)
            plots.append({"title": self._fix_name(p.dict.get("Title") or p.name),
                          "class": p.class_name, "x_title": axis_title(p),
                          "x_units": units_sym(p.dict.get("XUnits")),
                          "series": series})
        if not plots:
            return {"ok": False, "plots": [], "reason": "无绘图对象"}
        return {"ok": True, "plots": plots, "reason": ""}

    def export_plot_csv(self, path, mc=None, plot=None):
        """G6：按绘图导出对齐 XY 曲线 CSV（复现官方绘图导出：X 一列+每序列一列）。
        X 取 XAxisMonitor 的值序列（Iteration→迭代号；PhysicalTime→物理时间），
        Y 取 YAxisMonitor 的值序列，应用 XOffset/XScale/YOffset/YScale 变换。
        序列长度不一致或缺数据时该序列诚实跳过；无可导出序列返回 None。
        """
        if mc is None:
            mc = self.extract_monitor_curves()
        if not mc.get("ok"):
            return None
        if plot is None:
            pl = self.extract_plots(mc)
            plot = next((q for q in pl.get("plots", [])
                         if any(s["kind"] == "monitor" for s in q["series"])),
                        None)
            if plot is None:
                return None
        self._last_plot_export_skipped = []
        m_by_id = {}
        mm = next((o for o in self.objects
                   if (o.class_name or "") == "star.base.report.MonitorManager"),
                  None)
        if mm is not None:
            for mid in mm.dict.get("Keys") or []:
                mo = self.objmap.get(mid)
                if mo is not None:
                    m_by_id[mid] = self._fix_name(mo.name)
        data = mc["data"]
        xs, x_header = None, None
        cols, skipped = [], []
        for s in plot["series"]:
            if s["kind"] != "monitor":
                skipped.append("%s[tabular]" % s["name"])
                continue
            yn = m_by_id.get(s["y_id"])
            yd = data.get(yn) if yn else None
            if yd is None:
                skipped.append("%s[无Y数据]" % s["name"])
                continue
            y = _np.asarray(yd["y"], dtype=_np.float64) * \
                float(s["y_scale"] or 1.0) + float(s["y_offset"] or 0.0)
            if xs is None:
                xn = m_by_id.get(s["x_id"])
                xd = data.get(xn) if xn else None
                if xd is None or xd["y"] is None:
                    skipped.append("%s[无X数据]" % s["name"])
                    continue
                xs = _np.asarray(xd["y"], dtype=_np.float64) * \
                    float(s["x_scale"] or 1.0) + float(s["x_offset"] or 0.0)
                x_header = s.get("x_monitor") or "X"
            if xs is None or xs.size != y.size:
                skipped.append("%s[长度不一致 %d!=%d]" % (
                    s["name"], y.size, xs.size if xs is not None else -1))
                continue
            cols.append((s["name"], y))
        if xs is None or not cols:
            return None
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(",".join([x_header] + [c[0] for c in cols]) + "\n")
            for i in range(xs.size):
                fh.write("%.6g,%s\n" % (xs[i],
                                        ",".join("%.6g" % c[1][i] for c in cols)))
        self._last_plot_export_skipped = skipped
        return path

    # ---- G7：物理模型/材料/运动参数 ----
    _G7_SKIP_ATTRS = frozenset((
        "ClassName", "ObjectId", "Parent", "Seniority", "PhysicsContinuum",
        "AllModels", "Models", "Domain", "FullDomain", "NameManager",
        "ObjectsManager", "Manager", "Simulation", "stream", "continuum",
        "InactiveObjects", "Keys", "LinkedMaterials", "MaterialPropertyManager",
        "DataBaseMaterial", "GlobalData", "HolderID", "RelativeReferenceFrameManager",
        "CoordinateSystem",
        # 物理量对象内部的量纲/表达式元数据（G7 参数视角视为噪声）
        "DimensionsVector", "FunctionExpression", "AssemblyCode", "Definition",
        "IsDeltaValue", "IncludeLowerBound", "IncludeUpperBound",
        "LowerBound", "UpperBound", "PresentationName", "Active"))
    _G7_MAT_TAILS = ("Gas", "Liquid", "Solid", "GasComponent",
                     "LiquidComponent")

    def _g7_quantity(self, t):
        """物理量对象 → {value, units, oid, key, kind}。

        Scalar 取 Value，Vector 取 Vector；P1 写侧：oid/key 指向承载值的
        对象与 dict 键（编辑描述符，SetPropertyCommand 的目标）。
        """
        u = self.objmap.get(t.dict.get("Units"))
        d = {"units": (u.dict.get("PresentationName")
                       or u.dict.get("Description") or "") if u else ""}
        tail = (t.class_name or "").rsplit(".", 1)[-1]
        vec = tail == "VectorPhysicalQuantity"
        d["value"] = t.dict.get("Vector" if vec else "Value")
        d["oid"] = t.id
        d["key"] = "Vector" if vec else "Value"
        d["kind"] = "quantity"
        return d

    def _g7_option(self, t):
        """选项对象 → {selected, options, oid, key, kind}。

        Selected 为枚举序号不解引用；P1 写侧：oid/key 定位 Selected。
        """
        d = {}
        if "Selected" in t.dict:
            d["selected"] = t.dict.get("Selected")
        if "AvailableOptionsVector" in t.dict:
            d["options"] = t.dict.get("AvailableOptionsVector")
        d["oid"] = t.id
        d["key"] = "Selected"
        d["kind"] = "option"
        return d

    def _g7_value(self, v, depth):
        """单属性值解析（G7）。

        整数属性按引用解引用，按目标类尾分流：
          *PhysicalQuantity → {value, units}；*Option → {selected, options}；
          *Parameters/Limits → 递归字典；*SubModel/Model → 递归字典（限深）；
          其余（管理器/求解器/监视器/状态）丢弃。非整数标量/浮点/bool 原样保留。
        注意：Selected/AvailableOptionsVector 等枚举序号不走本入口（选项整体解析），
        避免与对象 id 数值碰撞；bool 先于 int 判断（bool 是 int 子类）；
        列表一律原样保留（整数列表可能是 DimensionsVector 这类向量而非引用）。
        """
        if isinstance(v, bool):
            return v
        if isinstance(v, list):
            return list(v)
        if not isinstance(v, int):
            return v
        t = self.objmap.get(v)
        if t is None or depth <= 0:
            return None
        tail = (t.class_name or "").rsplit(".", 1)[-1]
        if tail in ("ScalarPhysicalQuantity", "VectorPhysicalQuantity"):
            return self._g7_quantity(t)
        if tail.endswith("Option"):
            return self._g7_option(t)
        if tail.endswith(("Parameters", "Parameter", "Limits")):
            return self._g7_dict(t, depth - 1)
        if tail.endswith(("SubModel", "Model")):
            return self._g7_dict(t, depth - 1)
        return None

    def _g7_dict(self, o, depth, top=False):
        """对象 → 过滤簿记键后的参数字典（G7 模型/参数组通用）。

        P1 写侧：嵌套参数组层附带 _oid（编辑锚点：原始标量叶子回写到
        该对象）；顶层参数根（top=True）由模型行直接持有，不重复附加。
        """
        d = {}
        for k, v in o.dict.items():
            if k in self._G7_SKIP_ATTRS:
                continue
            val = self._g7_value(v, depth)
            if val is not None:
                d[k] = val
        if not top:
            d["_oid"] = o.id
        return d

    def extract_physics(self):
        """G7：物理模型/材料/运动参数解码。

        语义链（G7 侦查结论，Javadoc 属性名 ↔ 语料对照）：
          - star.common.PhysicsContinuum.ModelManager → 模型序列（Seniority 序）。
            模型参数按类尾分流解引用：标量/矢量物理量（Value/Vector+Units）、
            选项（Selected 序号+AvailableOptionsVector 值域）、参数组
            （*Parameters/*Limits 递归）、子模型（flowSubModel 等一层跟进）；
          - star.material.Gas/Liquid/... .MaterialPropertyManager → *Property 子对象，
            ActiveMethod → ConstantMaterialPropertyMethod(DbXmlTag) → Quantity
            取常量值+单位；UnsupportedMethod 等非常量方法只记方法与标签；
          - star.motion.MotionSpecification → Region+Continuum+Motion+
            ReferenceFrame；UserRotatingReferenceFrame 携带 RotationRate/
            AxisVector/OriginVector 物理量（MRF/旋转域）。
        返回 {"ok", "continua", "materials", "motion", "reason"}。
        """
        om = self.objmap
        continua, materials, motion = [], [], []
        for o in self.objects:
            cn = o.class_name or ""
            if cn == "star.common.PhysicsContinuum":
                models = []
                mm = om.get(o.dict.get("ModelManager"))
                if mm is not None:
                    for m2 in self.children.get(mm.id, []):
                        if (m2.class_name or "").rsplit(".", 1)[-1] == \
                                "NameManager":
                            continue
                        models.append({
                            "id": m2.id, "class": m2.class_name,
                            "name": self._fix_name(m2.name) or "",
                            "params": self._g7_dict(m2, 2, True)})
                continua.append({
                    "id": o.id, "name": self._fix_name(o.name) or "",
                    "is_active": o.dict.get("IsActive"),
                    "is_motion_active": o.dict.get("IsMotionActive"),
                    "models": models})
            elif cn.startswith("star.material.") and \
                    cn.rsplit(".", 1)[-1] in self._G7_MAT_TAILS:
                props = []
                pm = om.get(o.dict.get("MaterialPropertyManager"))
                if pm is not None:
                    for p in self.children.get(pm.id, []):
                        ptail = (p.class_name or "").rsplit(".", 1)[-1]
                        if not ptail.endswith("Property"):
                            continue
                        entry = {"class": p.class_name, "name": ptail}
                        am = om.get(p.dict.get("ActiveMethod"))
                        if am is not None:
                            entry["method"] = (am.class_name or "").rsplit(".", 1)[-1]
                            entry["method_tag"] = am.dict.get("DbXmlTag") or ""
                            q = om.get(am.dict.get("Quantity"))
                            if q is not None and (q.class_name or "").endswith(
                                    "PhysicalQuantity"):
                                entry.update(self._g7_quantity(q))
                        props.append(entry)
                materials.append({
                    "id": o.id, "class": cn,
                    "name": self._fix_name(o.name)
                    or o.dict.get("PresentationName") or "",
                    "properties": props})
            elif cn == "star.motion.MotionSpecification":
                reg = om.get(o.dict.get("Region"))
                cont = om.get(o.dict.get("Continuum"))
                mo = om.get(o.dict.get("Motion"))
                rf = om.get(o.dict.get("ReferenceFrame"))
                entry = {
                    "id": o.id,
                    "region": self._fix_name(reg.name) if reg else "",
                    "continuum": self._fix_name(cont.name) if cont else "",
                    "motion_class": mo.class_name if mo else "",
                    "motion_name": self._fix_name(mo.name) if mo else "",
                    "ref_frame_class": rf.class_name if rf else "",
                    "ref_frame_name": self._fix_name(rf.name) if rf else ""}
                if rf is not None:
                    for k in ("RotationRate", "AxisVector", "OriginVector"):
                        q = om.get(rf.dict.get(k))
                        if q is not None and (q.class_name or "").endswith(
                                "PhysicalQuantity"):
                            entry[k] = self._g7_quantity(q)
                motion.append(entry)
        if not continua:
            return {"ok": False, "continua": [], "materials": [],
                    "motion": [], "reason": "无 PhysicsContinuum（纯几何/网格文件）"}
        return {"ok": True, "continua": continua, "materials": materials,
                "motion": motion, "reason": ""}

    def _g8_parts(self, d2):
        """Displayer → InputParts/Collector PartGroup → Keys 解引用（Region/Boundary）。"""
        pg = self.objmap.get(d2.dict.get("InputParts")) \
            or self.objmap.get(d2.dict.get("Collector"))
        parts = []
        if pg is None:
            return parts
        for k in pg.dict.get("Keys") or []:
            t = self.objmap.get(k)
            if t is not None:
                parts.append({"name": self._fix_name(t.name) or "",
                              "class": (t.class_name or "").rsplit(".", 1)[-1]})
        return parts

    def _g8_displayer(self, d2):
        """单个 Displayer → 通用属性 + ScalarDisplayer 的场/图例/颜色映射。"""
        tail = (d2.class_name or "").rsplit(".", 1)[-1]
        om = self.objmap
        col = d2.dict.get("DisplayerColor")
        entry = {
            "id": d2.id, "class": tail,
            "name": self._fix_name(d2.name)
                    or self._fix_name(d2.dict.get("PresentationName") or ""),
            "color": col if isinstance(col, list) else None,
            "opacity": d2.dict.get("Opacity"),
            "parts": self._g8_parts(d2)}
        rep = om.get(d2.dict.get("Representation"))
        if rep is not None:
            entry["representation"] = (rep.class_name or "").rsplit(".", 1)[-1]
        if tail != "ScalarDisplayer":
            return entry
        q = om.get(d2.dict.get("ScalarDisplayQuantity"))
        if q is not None:
            ff = om.get(q.dict.get("FieldFunction"))
            un = om.get(q.dict.get("Units"))
            field = {
                "name": self._fix_name(ff.name) if ff is not None else "",
                "units": self._fix_name(un.name) if un is not None else "",
                "auto_range": q.dict.get("AutoRange")}
            gr = q.dict.get("GlobalRange")
            if isinstance(gr, list) and len(gr) == 2:
                field["range"] = gr
            lo = om.get(q.dict.get("MinimumValue"))
            hi = om.get(q.dict.get("MaximumValue"))
            if lo is not None and "Value" in lo.dict:
                field["min"] = lo.dict.get("Value")
            if hi is not None and "Value" in hi.dict:
                field["max"] = hi.dict.get("Value")
            entry["field"] = field
        lg = om.get(d2.dict.get("Legend"))
        if lg is not None:
            lut = om.get(lg.dict.get("LookupTable"))
            entry["legend"] = {
                "lut": self._fix_name(lut.name) if lut is not None else "",
                "lut_class": (lut.class_name or "").rsplit(".", 1)[-1]
                             if lut is not None else "",
                "labels": lg.dict.get("NumberOfLabels"),
                "format": lg.dict.get("LabelFormat"),
                "position": lg.dict.get("PositionCoordinate"),
                "height": lg.dict.get("Height"),
                "visible": lg.dict.get("Visible")}
            cm = lut.dict.get("ColorMap") if lut is not None else None
            if isinstance(cm, dict):
                cv = cm.get("ColorValues")
                bps = []
                if isinstance(cv, list) and len(cv) >= 8 and len(cv) % 4 == 0:
                    for i in range(0, len(cv), 4):
                        bps.append({"pos": cv[i], "rgb": cv[i + 1:i + 4]})
                entry["colormap"] = {
                    "name": self._fix_name(lut.name) or "",
                    "values": cv,
                    "breakpoints": bps,
                    "alphas": cm.get("AlphaValues"),
                    "colorspace": cm.get("ColorSpace")}
        return entry

    def _g8_annotations(self):
        """全局注记定义（AnnotationManager.Keys → star.vis.*Annotation 摘要）。"""
        am = next((o for o in self.objects
                   if (o.class_name or "") == "star.vis.AnnotationManager"),
                  None)
        defs = {}
        if am is None:
            return defs
        for k in am.dict.get("Keys") or []:
            a = self.objmap.get(k)
            if a is None or not (a.class_name or "").endswith("Annotation"):
                continue
            defs[a.id] = {
                "class": (a.class_name or "").rsplit(".", 1)[-1],
                "name": self._fix_name(a.name)
                        or self._fix_name(a.dict.get("PresentationName") or ""),
                "text": a.dict.get("Text") or a.dict.get("DisplayText") or "",
                "color": a.dict.get("Color"),
                "opacity": a.dict.get("Opacity"),
                "font": a.dict.get("FontString")}
        return defs

    def extract_scene_display(self):
        """G8：场景显示参数解码（Scene → Displayer/颜色映射/图例/灯光）。

        语义链（G8 侦查结论，语料 vortexShed_tutor.sim 对照）：
          - star.vis.Scene：DisplayerManager.Keys → PartDisplayer/ScalarDisplayer；
            BackgroundColorMode + SolidBackgroundColor（实背景）、LightManager
            子树（Light 的 Azimuth/Elevation/Intensity/Color/Enabled）、
            CurrentView/AnnotationPropManager（注记挂点）；
          - PartDisplayer：DisplayerColor 直接 RGB 三元组、Opacity、
            Representation → FvRepresentation、InputParts PartGroup.Keys →
            Region/Boundary（部件解引用，与 semantic_report 同构）；
          - ScalarDisplayer：ScalarDisplayQuantity 键 → FieldFunction（名称）、
            Units（'/s'）、GlobalRange [min,max] 直接列表 + Minimum/MaximumValue
            物理量；Legend 键 → LookupTable（PredefinedLookupTable 名）+
            LabelFormat/PositionCoordinate/NumberOfLabels；LookupTable.ColorMap
            携带 ColorValues（4n 组「位置,R,G,B」断点，位置单调 0→1 非均匀）
            + AlphaValues；
          - 注记链：Scene.AnnotationPropManager → prop.Keys → *AnnotationProp
            （Annotation 解引用 + Visible/Position/Height/Width/Location），
            AnnotationGroup（SimpleViewAnnotationGroup）.Keys → 场景默认显示的
            注记对象；全局定义 AnnotationManager.Keys → star.vis.*Annotation
            （Text/Color/Opacity/FontString）。
        返回 {"ok", "scenes": [...], "reason"}；无 Scene 对象时诚实拒绝。
        """
        om = self.objmap
        scenes = []
        for o in self.objects:
            if (o.class_name or "") != "star.vis.Scene":
                continue
            lights = []
            for c in self.children.get(o.id, []):
                if (c.class_name or "").rsplit(".", 1)[-1] != "LightManager":
                    continue
                for lt in self.children.get(c.id, []):
                    if (lt.class_name or "").rsplit(".", 1)[-1] != "Light":
                        continue
                    lights.append({
                        "name": self._fix_name(lt.name)
                                or self._fix_name(
                                    lt.dict.get("PresentationName") or ""),
                        "azimuth": lt.dict.get("Azimuth"),
                        "elevation": lt.dict.get("Elevation"),
                        "intensity": lt.dict.get("Intensity"),
                        "color": lt.dict.get("Color"),
                        "enabled": lt.dict.get("Enabled")})
            bg = {"mode": o.dict.get("BackgroundColorMode")}
            sb = om.get(o.dict.get("SolidBackgroundColor"))
            if sb is not None and "Color" in sb.dict:
                bg["solid"] = sb.dict.get("Color")
            displayers = []
            dm = om.get(o.dict.get("DisplayerManager"))
            if dm is not None:
                for k in dm.dict.get("Keys") or []:
                    d2 = om.get(k)
                    if d2 is not None and (d2.class_name or "").startswith(
                            "star.vis.") and "Displayer" in (d2.class_name or ""):
                        displayers.append(self._g8_displayer(d2))
            anns = {"props": [], "shown": []}
            apm = om.get(o.dict.get("AnnotationPropManager"))
            if apm is not None:
                for k in apm.dict.get("Keys") or []:
                    pr = om.get(k)
                    if pr is None or "AnnotationProp" not in (
                            pr.class_name or ""):
                        continue
                    tgt = om.get(pr.dict.get("Annotation"))
                    anns["props"].append({
                        "class": (pr.class_name or "").rsplit(".", 1)[-1],
                        "annotation": self._fix_name(tgt.name)
                                      if tgt is not None else "",
                        "visible": pr.dict.get("Visible"),
                        "position": pr.dict.get("Position"),
                        "height": pr.dict.get("Height"),
                        "width": pr.dict.get("Width"),
                        "location": pr.dict.get("Location")})
                grp = om.get(apm.dict.get("AnnotationGroup"))
                if grp is not None:
                    for k in grp.dict.get("Keys") or []:
                        a = om.get(k)
                        if a is not None:
                            anns["shown"].append(self._fix_name(a.name) or "")
            scenes.append({
                "id": o.id,
                "name": self._fix_name(o.name)
                        or self._fix_name(
                            o.dict.get("PresentationName") or ""),
                "background": bg, "lights": lights, "displayers": displayers,
                "annotations": anns})
        if not scenes:
            return {"ok": False, "scenes": [], "annotations": {},
                    "reason": "无 Scene 对象（纯几何/网格文件）"}
        return {"ok": True, "scenes": scenes,
                "annotations": self._g8_annotations(), "reason": ""}

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
        L.append("状态表: %d 字符 / %d token / %d 记录 (%s 编码)" % (
            len(self.state_text or ""), len(self.tokens), len(self.records), self.state_mode))
        if self.state_magic:
            L.append("魔数块: %r" % self.state_magic)
        if self.state_banner:
            L.append("Banner: %s" % self.state_banner.strip())
        L.append("对象图: %d 个对象 (id 2..%d)" % (len(self.objects),
                                                  len(self.objects) + 1 if self.objects else 0))
        if self.nested_transmits:
            L.append("嵌套 TRANSMIT 子块: %d (记录数 %s)" % (len(self.nested_transmits),
                     ", ".join(str(nt["count"]) for nt in self.nested_transmits[:8])))
        if self.state_segments:
            L.append("状态表分段（多 id 魔数）: %d 段 (%s...)" % (
                len(self.state_segments),
                ", ".join(str(s["length"]) for s in self.state_segments[:6])))
        if self.embedded_states:
            L.append("表内嵌状态标记: %d 处 -> %s" % (len(self.embedded_states),
                     "; ".join("%s:%s" % (es["object_id"], es["class"] or "?")
                               for es in self.embedded_states[:5])))
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
# 5.5 G1 状态表结构化语义树（T/V/S/Z/J/L 块 + ?指针 + 标记流）
# ----------------------------------------------------------------------------
# 依据 21 文件语料（16 个 ASCII 状态表）的 token 频率与上下文模式审计：
#   - ?N 三分法：N 命中 objmap → 对象图引用；N<4 → 结构类型标记；
#     其余 → 状态表内部实体 id（如 face/edge 流的递增 id，1097/1098...）
#   - 29 元素：`29, A, B, 0, C, D, E, f0, f1, f2, 18, ...`，f 三元组在
#     adjointWing 上 27/39 命中 Float8 顶点表坐标（几何元素，B-Rep 候选）
#   - 135 元素：前面必跟 3 个数字（顶点索引，7/8 验证）→ face 族标记
#   - 255：T 块分段分隔 + 元素边界
# 置信度：confirmed = 多文件交叉验证；likely = 单文件强模式；unknown = 保留结构。

STATE_MARKERS = {
    81: ("container-open", "likely"),      # `81, 1, <id>` 块头模式；81/82 不严格配对
    82: ("container-close", "likely"),
    83: ("container-mark", "likely"),
    29: ("geometry-element", "confirmed"),  # A,B,0,C,D,E + 3 浮点(顶点表相关)
    134: ("face-family", "likely"),
    135: ("face-family", "confirmed"),      # 前跟 v0,v1,v2 顶点索引
    204: ("element-mark", "likely"),
    255: ("separator", "confirmed"),        # T 块分段 + 元素边界
    17: ("field-mark", "likely"),
    18: ("field-mark", "likely"),
    11: ("field-mark", "likely"),
    16: ("field-mark", "likely"),
    30: ("field-mark", "likely"),
    -15: ("negative-mark", "likely"),
    -17: ("negative-mark", "likely"),
    -18: ("negative-mark", "confirmed"),    # ?1 face 流固定偏移出现
}


def _sc_float(tok):
    """STAR-CORE 数值表示法 → float（.xxx 无前导 0 / 定点整数+指数 600494083547542e-31）。"""
    try:
        return float(tok)
    except (TypeError, ValueError):
        return None


def _decode_pointer(rec, objmap):
    """?N 指针三分法：对象引用 / 结构类型标记 / 内部实体 id。"""
    ref = rec.get("ref")
    if not isinstance(ref, int):
        return {"role": "raw", "ref": ref, "conf": "unknown"}
    if ref in objmap:
        o = objmap[ref]
        return {"role": "object-ref", "ref": ref, "conf": "confirmed",
                "target": {"class": o.class_name, "name": o.name,
                           "layer": o.layer}}
    if ref < 4:
        return {"role": "type-tag", "ref": ref, "conf": "confirmed",
                "meaning": {0: "type-0", 1: "face-element", 2: "type-2",
                            3: "type-3"}.get(ref, "?")}
    return {"role": "entity-ref", "ref": ref, "conf": "likely",
            "meaning": "状态表内部实体 id（face/edge 流）"}


def _decode_elements(values, markers=None):
    """把记录 values 切成类型化元素序列（标记/int/float/+-元素）。"""
    markers = markers or STATE_MARKERS
    out = []
    for v in values:
        val = v.get("value")
        if v.get("kind") == "element":
            out.append({"t": "element", "v": val, "conf": "confirmed"})
            continue
        if isinstance(val, int) and not isinstance(val, bool) and val in markers:
            meaning, conf = markers[val]
            out.append({"t": "marker", "v": val, "meaning": meaning, "conf": conf})
        elif isinstance(val, int) and not isinstance(val, bool):
            out.append({"t": "int", "v": val, "conf": "confirmed"})
        elif isinstance(val, float):
            out.append({"t": "float", "v": val, "conf": "confirmed"})
        else:
            out.append({"t": "other", "v": val, "conf": "unknown"})
    return out


def decode_state_tree(sim, max_records=0, vertex_check=True):
    """状态表 → 结构化语义树（G1）。

    每个记录产出：
      head    记录头（kind/name/id/flags/version/fmt/value）
      ref     pointer 记录的三分法引用解析
      elements 类型化元素流（标记字典标注 + 置信度）
      segments T 块按 255 分段
      geometry_check 29 元素浮点三元组命中 Float8 顶点表的数量（可选）
    未知字段保留原始结构并标 conf=unknown —— 任意记录可解码，无丢失。
    """
    vset = None
    if vertex_check and _np is not None:
        merged = set()
        for a in sim.arrays:
            # 多网格文件含多张顶点表：合并全部 Float8 候选再判定命中
            if (a.get("type") == "Float8" and a.get("count")
                    and a["count"] % 3 == 0 and 0 < a["count"] <= 300000):
                d = sim.array_data(a["index"])
                if d is not None and hasattr(d, "dtype"):
                    verts = d.reshape(-1, 3)
                    merged.update(map(tuple, _np.round(verts, 9)))
        if merged:
            vset = merged
    tree = []
    for rec in sim.records[:max_records or None]:
        node = {"head": {k: rec.get(k) for k in
                         ("kind", "name", "id", "flags", "version", "fmt", "value")}}
        if rec.get("kind") == "pointer":
            node["ref"] = _decode_pointer(rec, sim.objmap)
        if rec.get("kind") == "banner":
            node["message"] = rec.get("message")
            tree.append(node)
            continue
        if rec.get("raw") is not None:
            node["raw_hex"] = rec["raw"][:96]
            if rec.get("values_decoded"):
                node["elements"] = _decode_elements(
                    [{"kind": "d", "value": v} for v in rec["values_decoded"]])
            tree.append(node)
            continue
        elements = _decode_elements(rec.get("values", []))
        node["elements"] = elements
        node["n_elements"] = len(elements)
        # T 块：按 255 分段
        if rec.get("fmt") == "T" and rec.get("kind") == "anonymous":
            segs, cur = [], []
            for e in elements:
                cur.append(e)
                if e.get("t") == "marker" and e.get("v") == 255:
                    segs.append(cur)
                    cur = []
            if cur:
                segs.append(cur)
            node["segments"] = [{"n": len(s),
                                 "head": [e.get("v") for e in s[:6]]} for s in segs]
        # 29 元素几何验证：其后第 7-9 个值为浮点三元组
        if vset is not None:
            els = elements
            hits = tot = 0
            for i, e in enumerate(els):
                if e.get("t") == "marker" and e.get("v") == 29 and i + 9 < len(els):
                    tri = [els[i + j].get("v") for j in (7, 8, 9)]
                    if all(isinstance(x, float) for x in tri):
                        tot += 1
                        if tuple(round(x, 9) for x in tri) in vset:
                            hits += 1
            if tot:
                node["geometry_check"] = {"vertex_hits": hits, "triples": tot}
        tree.append(node)
    return tree


def state_grammar_report(sim):
    """G1 文法统计：记录/fmt 分布、标记频率、指针解析率、几何验证率。"""
    from collections import Counter
    fmts = Counter()
    roles = Counter()
    marker_freq = Counter()
    markers = set(STATE_MARKERS)
    geo = Counter()
    for rec in sim.records:
        if rec.get("fmt"):
            fmts[rec["fmt"]] += 1
        if rec.get("kind") == "pointer":
            ref = rec.get("ref")
            if isinstance(ref, int):
                roles["object-ref" if ref in sim.objmap else
                       ("type-tag" if ref < 4 else "entity-ref")] += 1
            else:
                roles["raw-pointer"] += 1
        for v in rec.get("values", []):
            val = v.get("value")
            if isinstance(val, int) and not isinstance(val, bool) and val in markers:
                marker_freq[val] += 1
    tree = decode_state_tree(sim, vertex_check=True)
    for node in tree:
        gc = node.get("geometry_check")
        if gc:
            geo["triples"] += gc["triples"]
            geo["hits"] += gc["vertex_hits"]
    n_ptr = sum(roles.values())
    return {
        "state_mode": sim.state_mode,
        "n_records": len(sim.records),
        "fmt_distribution": dict(sorted(fmts.items(), key=lambda kv: -kv[1])),
        "pointer_roles": dict(roles),
        "pointer_resolved_pct": round(100.0 * roles.get("object-ref", 0) / n_ptr, 1) if n_ptr else None,
        "marker_frequency": {str(k): v for k, v in
                             sorted(marker_freq.items(), key=lambda kv: -kv[1])[:15]},
        "geometry_vertex_check": dict(geo) if geo else None,
    }


# ----------------------------------------------------------------------------
# 5.6 G2 数组块语义标注（A<n> 引用 × 网格规模自洽性交叉验证）
# ----------------------------------------------------------------------------
# 三类证据源：
#   a) 结构：Character1 = 主状态表 / 嵌套 TRANSMIT 子块（__init__ 已切分）
#   b) 引用：ascii 状态表中 fmt 含 "A" 的记录，值 = 文件级数组块下标
#      （binary 变体的单字母 "A" 是字符串类型码，不作引用解释）
#   c) 规模自洽性：count == part.TriangleCount*3 → 面索引表；
#      nverts 匹配面索引跨度 → 顶点坐标表；|max|<=1 → 法向；
#      Integer8 count==1 递增 → 字节偏移表；其余按内容给候选拍注。
ARRAY_ROLES_CN = {
    "state-table": "主状态表",
    "nested-transmit": "嵌套 TRANSMIT 子状态表",
    "state-referenced": "状态表 A<n> 引用数组",
    "face-indices": "面索引表",
    "vertex-coords": "顶点坐标表",
    "normals": "法向量表",
    "cell-indices-hex": "体单元表候选(hex)",
    "cell-indices-tet": "体单元表候选(tet)",
    "scalar-table-f64": "Float8 标量表",
    "scalar-table-f32": "Float4 标量表",
    "byte-offsets": "字节偏移表(Integer8)",
    "param-vector": "参数向量",
    "index-permutation": "排序/恒等映射表",
    "type-id-table": "类型/材质 ID 表（边界映射线索）",
    "char-blob": "字符数据块",
    "unclassified": "未分类",
}

# 角色优先级（小者先）：引用证据 > 结构匹配 > 内容猜测
_ROLE_ORDER = {r: k for k, r in enumerate([
    "state-table", "nested-transmit", "state-referenced", "face-indices",
    "vertex-coords", "normals", "byte-offsets", "scalar-table-f32",
    "scalar-table-f64", "cell-indices-hex", "cell-indices-tet",
    "index-permutation", "type-id-table", "param-vector",
    "char-blob", "unclassified"])}


def annotate_arrays(sim):
    """给每个数组块标 role/names/refs/evidence。返回与 sim.arrays 等长的 list。"""
    n = len(sim.arrays)
    anns = [{"index": i, "type": a["type"], "count": a["count"],
             "role": None, "role_cn": None, "names": [],
             "refs": [], "evidence": []} for i, a in enumerate(sim.arrays)]

    def tag(i, role, ev=None):
        a = anns[i]
        if ev:
            a["evidence"].append(ev)
        if a["role"] is None or _ROLE_ORDER[role] < _ROLE_ORDER[a["role"]]:
            a["role"] = role
            a["role_cn"] = ARRAY_ROLES_CN.get(role, role)

    # a) Character1：主状态表 / 嵌套子块 / 其余字符块
    state_seen = False
    for i, a in enumerate(sim.arrays):
        if a["type"] == "Character1":
            if not state_seen:
                state_seen = True
                tag(i, "state-table", "首个 Character1（parse_state_table 输入）")
            else:
                tag(i, "char-blob")
    for nt in sim.nested_transmits:
        ai = nt.get("array_index")
        if isinstance(ai, int) and 0 <= ai < n:
            nrec = len(nt.get("records") or [])
            tag(ai, "nested-transmit", "嵌套 TRANSMIT 子表（记录数 %d）" % nrec)

    # b) ascii A<n> 引用
    if sim.state_mode == "ascii":
        for rec in sim.records:
            fmt = rec.get("fmt") or ""
            v = rec.get("value")
            if "A" not in fmt or not isinstance(v, int) or isinstance(v, bool):
                continue
            if not (0 <= v < n):
                continue
            name = rec.get("name")
            oid = rec.get("id")
            o = sim.objmap.get(oid) if oid is not None else None
            anns[v]["refs"].append({
                "record": name or "<anon>", "id": oid, "fmt": fmt,
                "owner_class": o.class_name if o else None})
            if name and name not in anns[v]["names"]:
                anns[v]["names"].append(name)
            tag(v, "state-referenced", "A%d <- %s(id=%s%s)" % (
                v, name or "<anon>", oid,
                ", %s" % o.class_name if o else ""))

    # c) 规模自洽性（需要 numpy 解码内容）
    if _np is not None:
        meta = sim.mesh_metadata()
        tri_by_count = {}
        for t in sorted(set(x for x in meta["TriangleCount"]
                            if isinstance(x, int) and x > 0), reverse=True):
            tri_by_count[t * 3] = t
        face_tabs = []
        for i, a in enumerate(sim.arrays):
            c = a["count"]
            if (a["type"] in ("Unsigned4", "Integer4") and isinstance(c, int)
                    and c > 0 and c in tri_by_count):
                face_tabs.append((i, tri_by_count[c]))
                tag(i, "face-indices", "count==TriangleCount(%d)*3" % tri_by_count[c])
        spans = []
        for i, tc in face_tabs:
            d = sim.array_data(i)
            if hasattr(d, "dtype") and d.size:
                f = d.ravel().astype("int64")
                spans.append((i, int(f.min()), int(f.max())))
        # 顶点坐标：nverts 与某面表跨度/最大索引匹配
        vhit = 0
        for i, a in enumerate(sim.arrays):
            if a["type"] != "Float8":
                continue
            c = a["count"] or 0
            if c % 3 or c == 0:
                continue
            nv = c // 3
            for fi, lo, hi in spans:
                if nv in (hi - lo + 1, hi + 1, hi):
                    tag(i, "vertex-coords", "nverts=%d 匹配面表[%d](min=%d,max=%d)"
                        % (nv, fi, lo, hi))
                    vhit += 1
                    break
        # 其余 Float8：法向 / 顶点候选 / 参数向量 / 标量
        for i, a in enumerate(sim.arrays):
            if a["type"] != "Float8" or anns[i]["role"] is not None:
                continue
            c = a["count"] or 0
            d = sim.array_data(i)
            flat = getattr(d, "ravel", lambda: [])()
            if not len(flat):
                tag(i, "unclassified", "空表")
                continue
            mx = float(_np.abs(_np.asarray(flat)).max())
            if c % 3 == 0 and mx <= 1.0 + 1e-9:
                tag(i, "normals", "|max|=%.3g<=1 且 count%%3==0（n=%d）" % (mx, c // 3))
            elif c % 3 == 0 and c >= 24:
                tag(i, "vertex-coords", "count%%3==0 无跨度证据（|max|=%.3g）" % mx)
            elif c <= 16:
                tag(i, "param-vector", "小表 count=%d" % c)
            else:
                tag(i, "scalar-table-f64", "|max|=%.3g count=%d" % (mx, c))
        # Integer8：字节偏移
        for i, a in enumerate(sim.arrays):
            if a["type"] != "Integer8":
                continue
            d = sim.array_data(i)
            if hasattr(d, "dtype"):
                flat = d.ravel()
                mono = bool(flat.size > 1 and _np.all(_np.diff(flat) > 0)) \
                    or bool(flat.size == 1)
                tag(i, "byte-offsets", "Integer8 count=%s 递增=%s sample=%s"
                    % (a["count"], mono, flat[:2].tolist()))
            else:
                tag(i, "unclassified", "Integer8 无法解码")
        # Float4：标量表
        for i, a in enumerate(sim.arrays):
            if a["type"] != "Float4":
                continue
            d = sim.array_data(i)
            if hasattr(d, "dtype"):
                flat = d.ravel()
                mx = float(_np.abs(flat).max()) if flat.size else 0.0
                tag(i, "scalar-table-f32", "|max|=%.3g count=%s" % (mx, a["count"]))
        # 剩余 U4/I4：体单元表候选
        claimed = {i for i, tc in face_tabs}
        for i, a in enumerate(sim.arrays):
            if (a["type"] not in ("Unsigned4", "Integer4")
                    or i in claimed or anns[i]["role"] is not None):
                continue
            c = a["count"] or 0
            d = sim.array_data(i)
            flat = d.ravel().astype("int64") if hasattr(d, "dtype") else []
            mx = int(flat.max()) if len(flat) else 0
            mn = int(flat.min()) if len(flat) else 0
            if len(flat) and mn > 0 and _np.array_equal(
                    _np.sort(flat), _np.arange(1, c + 1)):
                tag(i, "index-permutation", "值为 1..%d 的排列" % c)
            elif len(flat) and mn >= -10 ** 6 and mx < 10 ** 6 \
                    and len(_np.unique(flat)) <= 64:
                u = _np.unique(flat)
                tag(i, "type-id-table", "%d 个相异值 %s（逐元素类别）"
                    % (len(u), u[:8].tolist()))
            elif c >= 32 and c % 8 == 0:
                tag(i, "cell-indices-hex", "count%%8==0 ncell=%d max=%d（弱证据）"
                    % (c // 8, mx))
            elif c >= 32 and c % 4 == 0:
                tag(i, "cell-indices-tet", "count%%4==0 ncell=%d max=%d（弱证据）"
                    % (c // 4, mx))
            else:
                tag(i, "unclassified", "count=%d max=%d" % (c, mx))
    for i in range(n):
        tag(i, "unclassified")
    return anns


def array_annotation_report(sim):
    """G2 汇总：角色分布 / 覆盖率 / 面表↔顶点表自洽计数。"""
    anns = annotate_arrays(sim)
    from collections import Counter
    roles = Counter(a["role"] for a in anns)
    labeled = sum(1 for a in anns if a["role"] != "unclassified")
    n_face = sum(1 for a in anns if a["role"] == "face-indices")
    n_vhit = sum(1 for a in anns if a["role"] == "vertex-coords"
                 and any("匹配面表" in e for e in a["evidence"]))
    n_ref = sum(len(a["refs"]) for a in anns)
    return {
        "n_arrays": len(anns),
        "labeled_pct": round(100.0 * labeled / len(anns), 1) if anns else 100.0,
        "roles": dict(sorted(roles.items(), key=lambda kv: kv[0])),
        "a_refs_resolved": n_ref,
        "face_tables": n_face,
        "vertex_span_matched": n_vhit,
        "annotations": anns,
    }


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


def g7_format_value(v):
    """G7 参数值的单行语义格式化（CLI --physics 与 GUI 属性面板共用）。

    识别 extract_physics() 的三类节点：物理量 {"value","units"}、
    选项 {"selected","options"}、参数组（dict 递归）；其余按类型直排。
    """
    if isinstance(v, dict):
        if "value" in v:
            val, u = v["value"], v.get("units", "")
            if isinstance(val, list):
                return "(%s)%s" % (
                    ", ".join("%.6g" % x for x in val),
                    (" %s" % u) if u else "")
            s = "%.6g" % val if isinstance(val, float) else str(val)
            return s + ((" %s" % u) if u else "")
        if "selected" in v:
            return "选项[%s]%s" % (v["selected"],
                                  (" 值域%s" % (v["options"],))
                                  if v.get("options") else "")
        return "{%s}" % " ".join("%s=%s" % (k, g7_format_value(x))
                                 for k, x in v.items()
                                 if not k.startswith("_"))
    if isinstance(v, list):
        return "[%s]" % ", ".join(g7_format_value(x) for x in v)
    if isinstance(v, float):
        return "%.6g" % v
    return str(v)


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
    ap.add_argument("--layers", action="store_true", help="按语义层统计对象（几何/网格/物理/场景...）")
    ap.add_argument("--aliases", action="store_true", help="--objects 时同时显示类名别名解析结果")
    ap.add_argument("--validate", action="store_true", help="校验 ClassVersions 尾部统计与对象图一致性")
    ap.add_argument("--mesh", action="store_true", help="抽取网格并输出统计（顶点/面）")
    ap.add_argument("--mesh-export", metavar="STL", help="把抽取出的网格写为 ASCII STL")
    ap.add_argument("--volume-mesh", action="store_true",
                    help="抽取体网格（存储体系驱动）并输出统计")
    ap.add_argument("--volume-export", metavar="VTU",
                    help="把体网格写为 VTK XML UnstructuredGrid（VTK_POLYHEDRON）")
    ap.add_argument("--volume-boundaries", action="store_true",
                    help="G4：FvBoundary → Boundary 链抽取体网格边界面片（按边界统计）")
    ap.add_argument("--boundary-csv", metavar="CSV",
                    help="把 G4 体网格边界映射写为 CSV（每边界一行）")
    ap.add_argument("--solution-fields", action="store_true",
                    help="G5：抽取 .sim 内嵌解场（SolutionRepresentation → FvRegion cells 组）")
    ap.add_argument("--solution-csv", metavar="CSV",
                    help="把 G5 解场标量字段写为 CSV（逐单元一行）")
    ap.add_argument("--curves", action="store_true",
                    help="G6：抽取监视器曲线与绘图关联（MonitorManager + PlotManager）")
    ap.add_argument("--curves-csv", metavar="PREFIX",
                    help="按绘图导出对齐 XY 曲线 CSV（PREFIX_绘图名.csv）")
    ap.add_argument("--fingerprint", action="store_true", help="输出版本指纹（banner/release/编码/头部）")
    ap.add_argument("--check-length", action="store_true", help="状态表长度自校验")
    ap.add_argument("--report", action="store_true", help="语义层报告（Region/Continuum/Scene/Part）")
    ap.add_argument("--physics", action="store_true",
                    help="物理模型/材料/运动参数报告（G7）")
    ap.add_argument("--scenes", action="store_true",
                    help="场景显示参数报告（G8：Displayer/颜色映射/图例/灯光）")
    ap.add_argument("--boundaries", action="store_true", help="边界→面片映射统计（by_part 面索引/覆盖）")
    ap.add_argument("--binary-decode", type=int, default=0, help="二进制状态表：解码前 N 个带 raw 记录（int/double 段显示）")
    ap.add_argument("--binary-verify", action="store_true", help="二进制状态表：完整文法无损往返校验（G9）")
    ap.add_argument("--state-edit", metavar="EDIT", action="append", default=None,
                    help="状态表安全编辑（W2）：'记录序号:字段=值'（如 5:id=222 / 6:flags=2 / "
                         "12:version=3 / 4:name=x 等长）。仅等宽编辑已验证字段，不改他处字节")
    ap.add_argument("--edit-out", metavar="PATH", default=None,
                    help="--state-edit 的写出目标（默认同目录 *_edit.sim）")
    ap.add_argument("--array-op", metavar="SPEC", action="append", default=None,
                    help="数组块变长替换/删除（W1）：'replace:IDX=N' 把第 IDX 块重设为 N 元素 "
                         "（载荷以 0 填充）/'delete:IDX' 删除该块；自动重定位后续偏移并重算 StatePosition")
    ap.add_argument("--export", metavar="DIR", help="导出到目录（数组 .npy/.csv + JSON）")
    ap.add_argument("--max-records", type=int, default=0, help="--state 最多输出的记录数（0=全部）")
    ap.add_argument("--state-tree", action="store_true", help="输出状态表结构化语义树（G1）")
    ap.add_argument("--grammar", action="store_true", help="状态表文法统计报告（G1）")
    args = ap.parse_args(argv)

    sim = SimFile(args.file)

    if args.summary or not any([args.sections, args.arrays, args.state,
                                args.objects, args.tree, args.layers,
                                args.validate, args.mesh, args.mesh_export,
                                args.volume_mesh, args.volume_export,
                                args.volume_boundaries, args.boundary_csv,
                                args.solution_fields, args.solution_csv,
                                args.curves, args.curves_csv,
                                args.physics, args.scenes,
                                args.boundaries,
                                args.fingerprint, args.check_length,
                                args.report, args.export, args.state_tree,
                                args.grammar]):
        print(sim.summary())

    if args.fingerprint:
        fp = sim.version_fingerprint()
        print("\n== 版本指纹 ==")
        print("  banner 版本: %s" % fp["banner_version"])
        print("  StarVersion Release: %s" % fp["release"])
        print("  状态表编码: %s" % fp["state_mode"])
        print("  头部字段: %s" % ",".join(fp["header_keys"]))
        if sim.container_entry:
            print("  ZIP 容器条目: %s" % sim.container_entry)

    if args.check_length:
        chk = sim.check_state_length()
        print("\n== 状态表长度自校验 ==")
        print("  %s: %s" % ("通过" if chk["ok"] else ("失败" if chk["ok"] is False else "跳过"), chk["detail"]))

    if args.sections:
        print("\n== 分区 ==")
        for i, (d, payload, s0, ps) in enumerate(sim.sections):
            cls = d.get("ClassName") if isinstance(d, dict) else "?"
            extra = ""
            if payload is not None:
                extra = "  payload %d 字节" % len(payload)
            print("%4d @%-8d %-40s%s" % (i, s0, cls, extra))

    if args.arrays:
        rep = array_annotation_report(sim)
        ann_by_idx = {a["index"]: a for a in rep["annotations"]}
        print("\n== 数组表（G2 语义标注 | 覆盖率 %.1f%% | A引用解析 %d 条"
              " | 面表 %d 张 / 顶点跨度配对 %d 张）==" % (
                  rep["labeled_pct"], rep["a_refs_resolved"],
                  rep["face_tables"], rep["vertex_span_matched"]))
        for a in sim.arrays:
            ann = ann_by_idx.get(a["index"]) or {}
            data = sim.array_data(a["index"])
            if _np is not None and hasattr(data, "dtype"):
                preview = ", ".join("%r" % v for v in data[:6].tolist())
            elif isinstance(data, str):
                preview = data[:40]
            else:
                preview = repr(data[:24])
            sem = ann.get("role") or "-"
            names = ",".join(ann.get("names") or [])
            ev = "; ".join((ann.get("evidence") or [])[:2])
            print("%3d  %-10s n=%-6s @%-8d %-18s %-24s [%s ...]" % (
                a["index"], a["type"], a["count"], a["start"], sem,
                ("name=%s" % names) if names else "", preview))
            if ev:
                print("     -> %s" % ev)

    if args.state:
        print("\n== STAR-CORE 状态表（%d 记录）==" % len(sim.records))
        for k, rec in enumerate(sim.records):
            if args.max_records and k >= args.max_records:
                print("... 共 %d 条，其余略" % len(sim.records))
                break
            print("%5d  %s" % (k, _fmt_rec(rec, sim.objmap)))

    if args.state_tree:
        print("\n== 状态表结构化语义树（G1）==")
        tree = decode_state_tree(sim, max_records=args.max_records)
        for k, node in enumerate(tree):
            head = node["head"]
            parts = ["#%d" % k, head.get("kind") or "?"]
            if head.get("fmt") is not None:
                parts.append("fmt=%s" % head["fmt"])
            if head.get("name") is not None:
                parts.append("name=%s" % head["name"])
            if head.get("value") is not None:
                parts.append("val=%s" % head["value"])
            if node.get("ref"):
                r = node["ref"]
                parts.append("ref=%s(%s)" % (r.get("role"), r.get("ref")))
                if r.get("target"):
                    parts.append("-> %s%s" % (r["target"]["class"],
                                 (" name=%r" % r["target"]["name"]) if r["target"]["name"] else ""))
            if node.get("message"):
                parts.append("msg=%r" % node["message"])
            if node.get("n_elements") is not None:
                parts.append("n_el=%d" % node["n_elements"])
            if node.get("segments"):
                parts.append("segments=%d" % len(node["segments"]))
            if node.get("geometry_check"):
                gc = node["geometry_check"]
                parts.append("geo=%d/%d" % (gc["vertex_hits"], gc["triples"]))
            print("  " + "  ".join(str(x) for x in parts))
            if node.get("raw_hex"):
                print("      raw=%s..." % node["raw_hex"][:40])
            if args.max_records and k >= args.max_records - 1 and k + 1 < len(tree):
                print("... 共 %d 条，其余略" % len(tree))
                break

    if args.grammar:
        print("\n== 状态表文法统计（G1）==")
        rep = state_grammar_report(sim)
        print("  编码: %s   记录数: %d" % (rep["state_mode"], rep["n_records"]))
        print("  fmt 分布: %s" % ", ".join(
            "%s:%d" % (f, c) for f, c in rep["fmt_distribution"].items()))
        if rep["pointer_roles"]:
            print("  指针角色: %s" % ", ".join(
                "%s:%d" % (r, c) for r, c in rep["pointer_roles"].items()))
            if rep["pointer_resolved_pct"] is not None:
                print("  对象引用解析率: %.1f%%" % rep["pointer_resolved_pct"])
        if rep["marker_frequency"]:
            print("  高频标记: %s" % ", ".join(
                "?%s×%d" % (m, c) for m, c in rep["marker_frequency"].items()))
        if rep["geometry_vertex_check"]:
            g = rep["geometry_vertex_check"]
            h, t = g.get("hits", 0), g.get("triples", 0)
            print("  几何验证: %d/%d 顶点三元组命中 Float8 顶点表（%.1f%%）" % (
                h, t, 100.0 * h / t if t else 0))
    if args.objects:
        print("\n== 对象图（%d）==" % len(sim.objects))
        for o in sim.objects:
            extra = ""
            if args.aliases:
                rn = o.resolved_class
                if rn != o.class_name:
                    extra = "  (alias: %s)" % rn
            print("%5d  %-60s %s%s" % (o.id, o.class_name,
                                       ("name=%r" % o.name) if o.name else "", extra))

    if args.layers:
        census, named = sim.layer_census()
        print("\n== 语义层统计 ==")
        for layer, n in census.most_common():
            print("  %-18s %-12s %5d 个对象" % (layer, LAYER_CN.get(layer, ""), n))
        print("\n== 各层命名对象（前 12 个/层）==")
        for layer, items in sorted(named.items()):
            print("  [%s %s]" % (layer, LAYER_CN.get(layer, "")))
            for oid, cn, nm in items[:12]:
                print("     %5d %-48s %r" % (oid, cn, nm))
            if len(items) > 12:
                print("     ... 其余 %d 个" % (len(items) - 12))

    if args.validate:
        v = sim.validate_class_versions()
        print("\n== ClassVersions 校验 ==")
        if v["status"] == "no-ClassVersions":
            print("  本文件无 ClassVersions 尾部统计")
        else:
            print("  类注册表快照：%d 类，%d 类计数一致；注册表计数和 %d / 图内对象 %d" % (
                v["expected_classes"], v["matched"], v["expected_total"], v["actual_total"]))
            print("  （注：注册表含未序列化的运行时类，计数语义与图内对象数不同，按诊断信息使用）")
            for cn, m in list(v["mismatches"].items())[:10]:
                print("    %-50s 注册表 %s / 图内 %s" % (cn, m["expected"], m["actual"]))

    if args.tree:
        print("\n== 对象树 ==")
        for r in sim.roots:
            if sim.children.get(r.id):
                print_tree([r], sim.children)

    if args.report:
        rep = sim.semantic_report()
        print("\n== 语义层报告 ==")
        for r in rep["regions"]:
            print("  Region %r (id %d):" % (r["name"], r["id"]))
            for p in r["parts"]:
                print("    Part %r (%s, %s 三角)" % (
                    p["name"], p["class"], p["triangles"]))
            for b in r["boundaries"][:8]:
                print("    Boundary %r (id %d)" % (b["name"], b["id"]))
            if len(r["boundaries"]) > 8:
                print("    ... 共 %d 个边界" % len(r["boundaries"]))
        for c in rep["continua"]:
            print("  Continuum %r (id %d): %d 个模型" % (c["name"], c["id"], len(c["models"])))
            for m2 in c["models"][:10]:
                print("    %s %r" % (m2["class"], m2["name"]))
            if len(c["models"]) > 10:
                print("    ... 其余 %d 个" % (len(c["models"]) - 10))
        for s in rep["scenes"]:
            print("  Scene %r (id %d): %d 个显示器%s" % (
                s["name"], s["id"], len(s["displayers"]),
                ("，视图 %s" % (s["view"],) if s["view"] else "")))
            for d2 in s["displayers"][:6]:
                print("    %s %r" % (d2["class"], d2["name"]))
        print("  Part 清单（有网格规模）:")
        for p in rep["parts"]:
            print("    %r %s: %s 三角%s" % (
                p["name"], p["class"], p["triangles"],
                (" %s 顶点" % p["vertices"]) if p["vertices"] else ""))

    if args.physics:
        g7fmt = g7_format_value
        ph = sim.extract_physics()
        print("\n== 物理模型/材料/运动参数（G7） ==")
        if not ph.get("ok"):
            print("  诚实拒绝：%s" % ph.get("reason"))
        for c in ph["continua"]:
            print("  Continuum %r (id %d) 激活=%s 运动激活=%s: %d 个模型" % (
                c["name"], c["id"], c["is_active"], c["is_motion_active"],
                len(c["models"])))
            for m2 in c["models"]:
                ps = " ".join("%s=%s" % (k, g7fmt(v))
                              for k, v in m2["params"].items())
                print("    %s%s%s" % (
                    m2["class"].rsplit(".", 1)[-1],
                    (" %r" % m2["name"]) if m2["name"] else "",
                    ("：%s" % ps) if ps else ""))
        for mt in ph["materials"]:
            print("  材料 %r (%s): %d 属性" % (
                mt["name"], mt["class"].rsplit(".", 1)[-1],
                len(mt["properties"])))
            for p in mt["properties"]:
                line = "    %s [%s/%s]" % (p["name"], p["method"],
                                           p.get("method_tag", ""))
                if "value" in p:
                    line += " = %s" % g7fmt(
                        {"value": p["value"], "units": p.get("units", "")})
                print(line)
        if ph["motion"]:
            print("  运动规格（MotionSpecification）: %d 条" % len(ph["motion"]))
            for m3 in ph["motion"]:
                extras = " ".join(
                    "%s=%s" % (k, g7fmt(m3[k]))
                    for k in ("RotationRate", "AxisVector", "OriginVector")
                    if k in m3)
                print("    Region %r ← %s%s @ %s%s" % (
                    m3["region"],
                    m3["motion_class"].rsplit(".", 1)[-1],
                    (" %r" % m3["motion_name"]) if m3["motion_name"] else "",
                    m3["ref_frame_class"].rsplit(".", 1)[-1],
                    ("（%s）" % extras) if extras else ""))

    if args.scenes:
        s8 = sim.extract_scene_display()
        print("\n== 场景显示参数（G8） ==")
        if not s8.get("ok"):
            print("  诚实拒绝：%s" % s8.get("reason"))
        for s in s8["scenes"]:
            bg = s["background"]
            ann = s.get("annotations") or {}
            print("  Scene %r (id %d) 背景 mode=%s%s: %d 显示器, %d 灯光, %d 注记显示" % (
                s["name"], s["id"], bg.get("mode"),
                (" solid=%s" % (bg.get("solid"),)) if bg.get("solid") else "",
                len(s["displayers"]), len(s["lights"]),
                len(ann.get("shown") or [])))
            for lt in s["lights"]:
                print("    灯光 %r 方位=%s 仰角=%s 强度=%s 色=%s 启用=%s" % (
                    lt["name"], lt["azimuth"], lt["elevation"],
                    lt["intensity"], lt["color"], lt["enabled"]))
            for d2 in s["displayers"]:
                line = "    %s %r 透明度=%s" % (
                    d2["class"], d2["name"], d2["opacity"])
                if d2.get("color"):
                    line += " 色=%s" % (tuple(round(c, 3) for c in d2["color"]),)
                if d2.get("parts"):
                    line += " 部件=%s" % ",".join(
                        "%s(%s)" % (p["name"], p["class"])
                        for p in d2["parts"][:4])
                print(line)
                if d2.get("field"):
                    f = d2["field"]
                    print("      场=%r 单位=%r 全局范围=%s" % (
                        f.get("name"), f.get("units"),
                        f.get("range") or (f.get("min"), f.get("max"))))
                if d2.get("legend"):
                    lg = d2["legend"]
                    print("      图例 LUT=%r(%s) 标签数=%s 格式=%r 位置=%s" % (
                        lg.get("lut"), lg.get("lut_class"), lg.get("labels"),
                        lg.get("format"), lg.get("position")))
                if d2.get("colormap"):
                    cm = d2["colormap"]
                    bps = cm.get("breakpoints") or []
                    if bps:
                        b0 = (round(bps[0]["pos"], 3),
                              tuple(round(c, 3) for c in bps[0]["rgb"]))
                        b1 = (round(bps[-1]["pos"], 3),
                              tuple(round(c, 3) for c in bps[-1]["rgb"]))
                    else:
                        b0 = b1 = "-"
                    print("      颜色映射 %r：%d 组断点(位置,R,G,B) 首=%s 末=%s alpha=%s" % (
                        cm.get("name"), len(bps), b0, b1, cm.get("alphas")))
            if ann.get("props"):
                print("    注记显示属性：%s" % "; ".join(
                    "%s→%s 可见=%s 位置=%s 高=%s" % (
                        p["class"], p["annotation"] or "-", p["visible"],
                        p["position"], p["height"])
                    for p in ann["props"]))
            if ann.get("shown"):
                print("    注记组（AnnotationGroup.Keys 解引用）：%s" % ",".join(
                    ann["shown"]))
        defs8 = s8.get("annotations") or {}
        if defs8:
            print("  全局注记定义 %d 项：%s" % (len(defs8), "; ".join(
                "%s(%r)" % (d["class"], d["name"]) for d in defs8.values())))

    if args.binary_decode:
        print("== 二进制状态表数值流解码（启发式：2 字节整 + 8 字节双精度段） ==")
        n = 0
        for rec in sim.records:
            if rec.get("values_decoded") and n < args.binary_decode:
                vals = rec["values_decoded"]
                ds = rec.get("decode_stats") or {}
                head = " ".join("%s:%s" % (k, ("%.5g" % v) if k == "double" else v)
                                for k, _o, v in vals[:14])
                print("  rec %3d fmt=%s bits=%s stats=%s" % (
                    rec.get("token_index", -1), rec.get("fmt"), rec.get("bits"), ds))
                print("       %s..." % head)
                n += 1
        if n == 0:
            print("  无带 raw 的二进制记录（文件可能为 ASCII 编码）")

    if args.binary_verify:
        import re as _re
        print("== 二进制状态表完整文法（G9）无损往返校验 ==")
        if sim.state_mode != "binary":
            print("  状态表为 ASCII 编码，跳过 binary 文法校验")
        else:
            body = sim.state_text.encode("latin-1")
            bm = _re.search(
                rb"TRANSMIT FILE created by modeller version (\d+).{0,24}?SCH_([A-Za-z0-9_]+)",
                body)
            i0 = bm.end() if bm else 0
            tok, recs, magic, banner = parse_state_table_binary(sim.state_text)
            rb = serialize_binary_records(recs)
            orig = body[i0:]
            lossless = (rb == orig)
            named = [r for r in recs if r["kind"] == "named"]
            anon = [r for r in recs if r["kind"] == "anonymous"]
            data = [r for r in recs if r["kind"] == "data"]
            print("  banner=%r" % (banner or ""))
            print("  记录流 %d 字节：named=%d anonymous=%d data=%d" % (
                len(orig), len(named), len(anon), len(data)))
            print("  无损往返: %s" % ("通过（逐字节一致）" if lossless else "失败"))
            for r in named[:6]:
                print("    named %-22s id=%-6s flags=%s version=%s fmt=%s" % (
                    r["name"], r["id"], r["flags"], r["version"], r["fmt"]))
            if not lossless:
                for k in range(min(len(rb), len(orig))):
                    if rb[k] != orig[k]:
                        print("    首个差异 @%d 原=%02x 新=%02x" % (
                            k, orig[k], rb[k]))
                        break

    if args.state_edit:
        print("== 状态表安全编辑（W2：只动已确证记录，差分验证） ==")
        if sim.state_mode != "binary":
            print("  状态表为 ASCII 编码；W2 安全编辑作用于 binary 记录流，跳过")
        else:
            edits = []
            for spec in args.state_edit:
                idx, _, rest = spec.partition(":")
                fld, _, val = rest.partition("=")
                val = int(val)
                edits.append({"index": int(idx), fld: val})
            new_bytes, changed = edit_binary_state_records(
                sim.state_text, edits)
            verify_binary_state_edit(sim.state_text, new_bytes, edits)
            if len(new_bytes) != len(sim.state_text.encode("latin-1")):
                raise SystemExit("等宽编辑应保持记录流总长不变（变长留给 W1）")
            dest = args.edit_out or os.path.join(
                os.path.dirname(os.path.abspath(sim.path)),
                os.path.splitext(os.path.basename(sim.path))[0] + "_edit.sim")
            src = sim.path
            import shutil as _shutil
            shutil_err = None
            if src and os.path.isfile(src):
                _shutil.copy2(src, dest)
                with open(dest, "rb") as f:
                    blob = bytearray(f.read())
                # 原位替换存状态表的 Character1 数组载荷（同宽，偏移不变）
                for _a in sim.arrays:
                    if _a["type"] != "Character1":
                        continue
                    if _a.get("data") is None:
                        continue
                    if len(_a["data"]) == len(new_bytes):
                        st = int(_a["start"])
                        blob[st:st + len(new_bytes)] = new_bytes
                        break
                else:
                    shutil_err = "未定位状态表 Character1 数组"
            if shutil_err:
                print("  ! %s（仅写出状态载荷到临时文件）" % shutil_err)
                dest = os.path.join(
                    os.path.dirname(os.path.abspath(dest)),
                    os.path.basename(dest) + ".payload")
                with open(dest, "wb") as f:
                    f.write(new_bytes)
            else:
                with open(dest, "wb") as f:
                    f.write(bytes(blob))
            recs = parse_state_table_binary(sim.state_text)[1]
            for e in edits:
                r = recs[int(e["index"])]
                print("  记录 %d (%s) id=%s flags=%s version=%s -> %s" % (
                    e["index"], r["name"], r["id"], r["flags"], r["version"], e))
            print("  差分区间（仅这些记录字节变动）: %s" % changed)
            print("  写出 -> %s（其余字节逐字不变，重解析通过）" % dest)

    if args.array_op:
        import importlib
        _w = importlib.import_module("sim_writer")
        print("== 数组块变长替换/删除（W1：全量重定位 + StatePosition 重算） ==")
        ops, dest = [], (args.edit_out or os.path.join(
            os.path.dirname(os.path.abspath(sim.path)),
            os.path.splitext(os.path.basename(sim.path))[0] + "_arr.sim"))
        for spec in args.array_op:
            kind, _, rest = spec.partition(":")
            if kind == "replace":
                idx, _, n = rest.partition("=")
                ops.append({"op": "replace", "index": int(idx), "count": int(n)})
            elif kind == "delete":
                ops.append({"op": "delete", "index": int(rest)})
            else:
                raise SystemExit("未知 array-op: %s（replace:IDX=N / delete:IDX）" % spec)
        _tsize = {a["type"]: (a["dict"].get("sizeof<T>") or 1)
                  for a in sim.arrays}
        for op in ops:
            if op["op"] == "replace":
                sz = _tsize.get(sim.arrays[op["index"]]["type"], 1)
                op["payload"] = b"\x00" * (op["count"] * int(sz))
        src = sim.path
        _blob = open(src, "rb").read()
        _nb, _info = _w.apply_array_ops(_blob, sim, ops)
        import shutil as _sh
        _sh.copy2(src, dest)
        with open(dest, "wb") as _fh:
            _fh.write(_nb)
        for e in _info["edits"]:
            print("  arr[%d] %s delta=%+d range=%s" % (
                e["index"], e["op"], e["delta"], e["abs_range"]))
        print("  StatePosition %d -> %d" % (_info["old_state_position"],
                                            _info["new_state_position"]))
        print("  重开校验 -> %s" % dest)
        _r = SimFile(dest)
        ok = (_r.header["StatePosition"] == _info["new_state_position"]
              and len(_r.objects) == len(sim.objects))
        print("  重开一致: %s（%d 对象, %d 数组）" % (
            "通过" if ok else "失败", len(_r.objects), len(_r.arrays)))

    if args.boundaries:
        bf = sim.boundary_faces()
        print("== 边界 → 面片映射 ==")
        if bf is None:
            print("  无网格或无法定位每面 patch 数组")
        else:
            assigned = 0
            for name, per in sorted(bf["by_boundary"].items()):
                n = sum(len(v) for v in per.values())
                assigned += n
                print("  %-22s faces=%d parts=%s" % (name, n, list(per.keys())))
            print("  已指派面数: %d / %d" % (assigned, bf["total"]))
            print("  （未指派 = part 表面 patch 未被列出的 region 边界引用，见文档说明）")

    if args.volume_boundaries or args.boundary_csv:
        bf4 = sim.extract_boundary_faces()
        print("\n== 体网格边界映射（G4：FvBoundary → Boundary 链）==")
        if not bf4.get("ok"):
            print("  诚实拒绝：%s" % bf4.get("reason"))
        elif not bf4.get("boundaries"):
            print("  %s" % (bf4.get("reason") or "无边界"))
        else:
            for b in bf4["boundaries"]:
                extra = []
                if b["part_surface_index"] is not None:
                    extra.append("psi=%s" % b["part_surface_index"])
                if b["prostar_boun_id"] is not None:
                    extra.append("prostar=%s" % b["prostar_boun_id"])
                print("  %-24s region=%-20s Index=%-4s faces=%-6d %s" % (
                    b["name"], b["region_name"] or "-",
                    b["index"] if b["index"] is not None else "-",
                    b["face_count"], " ".join(extra)))
            print("  合计：%d 边界 / %d 面" % (
                len(bf4["boundaries"]), bf4["total_faces"]))
        if args.boundary_csv:
            print("  CSV -> %s" % (sim.export_boundary_csv(
                args.boundary_csv, bf4) or "-"))

    if args.solution_fields or args.solution_csv:
        sf5 = sim.extract_solution_fields()
        print("\n== 内嵌解场（G5：SolutionRepresentation → FvRegion cells 组）==")
        if not sf5.get("ok"):
            print("  诚实拒绝：%s" % sf5.get("reason"))
        else:
            print("  单元数: %d  字段数: %d  区域: %s" % (
                sf5["cell_count"], sf5["n_fields"],
                sf5.get("region_name") or "-"))
            for f in sf5["fields"]:
                vec = "x%d" % f["components"] if f["components"] > 1 else "  "
                print("  %-34s %-26s n=%-7d [%+.4g .. %+.4g]%s" % (
                    f["name"], f["type"], f["count"], f["min"], f["max"], vec))
        if args.solution_csv:
            print("  CSV -> %s" % (sim.export_solution_csv(
                args.solution_csv, sf5) or "-"))

    if args.curves or args.curves_csv:
        mc6 = sim.extract_monitor_curves()
        print("\n== 监视器曲线（G6：MonitorManager → XAxisData/YAxisValues 双 MasterArray）==")
        if not mc6.get("ok"):
            print("  诚实拒绝：%s" % mc6.get("reason"))
        else:
            for e in mc6["monitors"]:
                cv = e.get("cur_value")
                eq = ""
                if isinstance(cv, float) and (
                        cv == e["last"] or
                        abs(cv - e["last"]) <= 1e-9 * max(1.0, abs(cv))):
                    eq = "  ==CurrentValue"
                print("  %-32s %-24s n=%-7d y[%+.4g .. %+.4g] last=%+.4g%s" % (
                    e["name"], e["class"].split(".")[-1], e["n"],
                    e["y_min"], e["y_max"], e["last"], eq))
            pl6 = sim.extract_plots(mc6)
            print("  绘图关联（G2 标注：标题/轴标题/单位/图例）：")
            for q in pl6.get("plots", []):
                print("   ◇ %s (%s) X轴=%r 单位=%r" % (
                    q["title"], q["class"].split(".")[-1],
                    q["x_title"], q["x_units"]))
                for s in q["series"]:
                    if s["kind"] == "monitor":
                        print("      - %-34s X←%s Y←%s" % (
                            s["name"], s["x_monitor"] or "-",
                            s["y_monitor"] or "-"))
                    elif s["kind"] == "tabular":
                        print("      - %-34s [tabular] X列=%r Y列=%r%s" % (
                            s["name"], s.get("x_column"), s.get("y_column"),
                            (" 表文件=%r" % s["table_file"])
                            if s.get("table_file") else ""))
            if args.curves_csv:
                for q in pl6.get("plots", []):
                    safe = re.sub(r"[^\w\-]+", "_", q["title"]).strip("_") or "plot"
                    out = "%s_%s.csv" % (args.curves_csv, safe)
                    r = sim.export_plot_csv(out, mc6, q)
                    if r:
                        sk = getattr(sim, "_last_plot_export_skipped", None)
                        print("  CSV -> %s%s" % (
                            r, ("（跳过：%s）" % "; ".join(sk)) if sk else ""))

    if args.mesh:
        m = sim.extract_mesh()
        meta = m["meta"]
        print("\n== 网格抽取 ==")
        print("  各 part 三角数: %s" % sorted(meta["TriangleCount"], reverse=True))
        print("  VertexCount 来源: %s" % meta["VertexCount"])
        print("  元数据来源对象: %d 个" % len(meta["sources"]))
        if m["vertices"] is not None:
            print("  顶点: %d (来源=%s)" % (m["vertices"].shape[0], m["vertex_flag"]))
        if m["faces"] is not None:
            print("  面: %d (来源=%s)" % (m["faces"].shape[0], m["face_flag"]))
        if m["vertices"] is not None and m["faces"] is not None:
            print("  最大索引 %d vs 顶点数 %d -> %s" % (
                m["max_index"], m["n_vertices"], "一致" if m["consistent"] else "不一致"))

    if args.mesh_export:
        out = sim.export_stl(args.mesh_export)
        print("\nSTL 已导出到 %s" % out)

    if args.volume_mesh:
        vol = sim.extract_volume_mesh()
        print("\n== 体网格抽取（G3 存储体系驱动） ==")
        if not vol.get("ok"):
            print("  未匹配: %s" % vol.get("reason"))
        else:
            pts = vol.get("points")
            fv = vol.get("face_verts")
            loops = vol.get("cell_loops") or []
            nfaces = len(vol.get("cell_faces") or [])
            print("  顶点: %d  面: %d  单元: %d" % (pts.shape[0], nfaces,
                                                    vol.get("count")))
            print("  存储组: %s" % vol.get("groups"))
            if fv is not None:
                print("  面顶点跨度: 0..%d  索引总数: %d" % (int(fv.max()) if fv.size else -1,
                                                        int(fv.size)))
            shapes = {}
            for cell in loops:
                k = (len(cell), sum(len(l) for l in cell))
                shapes[k] = shapes.get(k, 0) + 1
            top = ", ".join("%d面/%d点 x%d" % (k[0], k[1], v)
                            for k, v in sorted(shapes.items(),
                                               key=lambda kv: -kv[1])[:8])
            print("  单元形状(面数/节点数): %s" % top)
            et = vol.get("elem_types")
            if et:
                print("  ElemType 编码: %r" % sorted(set(et)))

    if args.volume_export:
        out = sim.export_volume_vtu(args.volume_export)
        print("\n体网格 VTU 已导出到 %s" % out)

    if args.export:
        sim.export(args.export)
        print("\n已导出到 %s" % args.export)


if __name__ == "__main__":
    main()

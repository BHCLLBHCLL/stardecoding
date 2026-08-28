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
    """解析"二进制编码"状态表变体（部分老版本 .sim，如 airfoil.sim）。

    与 ASCII 变体的差异：魔数块后不再是 80 列折行文本，而是一条字节流——
    名字/格式字母仍是 ASCII，id/flags/version/尾值等数字字段为二进制
    （id=3 字节小端、flags/version/尾值=1 字节）。数值流（T 块等）的具体
    二进制文法未完全还原，按原始字节保留。
    返回 (tokens, records, magic, banner)。
    """
    lines = text.split("\n")
    magic = lines[:4]
    blob = text.encode("latin-1")[len("\n".join(magic)) + 1:]

    # banner：从字节流中正则提取（NUL 填充的版本号/SCH 串）
    bm = re.search(rb"TRANSMIT FILE created by modeller version (\d+).{0,24}?SCH_([A-Za-z0-9_]+)", blob)
    banner = ""
    if bm:
        banner = "T51 : TRANSMIT FILE created by modeller version %s SCH_%s" % (
            bm.group(1).decode("latin-1"), bm.group(2).decode("latin-1"))
        body = blob[bm.end():]
    else:
        body = blob

    NAME_BYTES = set(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")
    records = []
    i = 0
    n = len(body)
    tokens = []

    def scan_name(i):
        j = i
        while j < n and body[j] in NAME_BYTES:
            j += 1
        return body[i:j].decode("latin-1"), j

    while i < n:
        b = body[i]
        if b in NAME_BYTES:
            run, j = scan_name(i)
            is_bitmap = bool(re.fullmatch(r"[FTPB?]+", run))
            is_fmt = run in KNOWN_BIN_FMTS or (len(run) == 1 and run.isupper())
            if is_bitmap or is_fmt:
                rec = {"kind": "bitmap" if is_bitmap else "anonymous",
                       "name": None, "id": None, "flags": None,
                       "version": None, "fmt": run if not is_bitmap else None,
                       "bits": run if is_bitmap else None,
                       "value": None, "values": [], "token_index": len(tokens),
                       "raw": None}
                tokens.append(run)
                i = j
                if i < n and body[i] not in NAME_BYTES:
                    rec["value"] = body[i]; i += 1
                # 数值流：原始字节直到下一个字母 run
                k = i
                while k < n and body[k] not in NAME_BYTES:
                    k += 1
                if k > i:
                    rec["raw"] = body[i:k].hex(" ")
                i = k
                records.append(rec)
                continue
            # 名字记录
            rec = {"kind": "named", "name": run, "id": None, "flags": None,
                   "version": None, "fmt": None, "value": None, "values": [],
                   "token_index": len(tokens), "raw": None}
            tokens.append(run)
            i = j
            # id = 3 字节大端（= ASCII 模式 id × 256）
            if i + 3 <= n and body[i] not in NAME_BYTES:
                rec["id"] = int.from_bytes(body[i:i+3], "big") // 256
                i += 3
                # flags = 1 字节
                if i < n:
                    rec["flags"] = body[i]; i += 1
                if rec["flags"] > 2:
                    # flags 异常（>2）说明该字母 run 是数值流中浮点字节的假阳性
                    k = i
                    while k < n and body[k] not in NAME_BYTES:
                        k += 1
                    records.append({"kind": "data", "raw": body[i:k].hex(" "),
                                    "token_index": len(tokens) - 1, "values": []})
                    i = k
                    continue
                # 可选 version：非打印字节且后面跟字母
                if (i + 1 < n and body[i] not in NAME_BYTES
                        and body[i+1] in NAME_BYTES):
                    rec["version"] = body[i]; i += 1
                # fmt 字母 + 1 字节尾值
                fm, j2 = scan_name(i)
                if fm and (fm in KNOWN_BIN_FMTS or (len(fm) == 1 and fm.isupper())):
                    rec["fmt"] = fm
                    i = j2
                    if i < n and body[i] not in NAME_BYTES:
                        rec["value"] = body[i]; i += 1
                    # 数值流：原始字节直到下一个字母 run
                    k = i
                    while k < n and body[k] not in NAME_BYTES:
                        k += 1
                    if k > i:
                        rec["raw"] = body[i:k].hex(" ")
                    i = k
                else:
                    i = j2
            records.append(rec)
        else:
            # 非字母字节（数值流/位图等）：原样累计
            k = i
            while k < n and body[k] not in NAME_BYTES:
                k += 1
            tokens.append("<bytes:%d>" % (k - i))
            records.append({"kind": "data", "raw": body[i:k].hex(" "),
                            "token_index": len(tokens) - 1, "values": []})
            i = k

    if banner:
        records.insert(0, {"kind": "banner", "fmt": "T", "value": 51,
                           "message": banner.split(": ", 1)[-1], "token_index": -1})
    # 对带 raw 的记录追加尽力数值解码（2 字节整 + 8 字节双精度交替段）
    for rec in records:
        if rec.get("raw"):
            vals, stats = decode_binary_values(rec["raw"])
            rec["values_decoded"] = vals
            rec["decode_stats"] = stats
    return tokens, records, magic, banner


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
    ap.add_argument("--fingerprint", action="store_true", help="输出版本指纹（banner/release/编码/头部）")
    ap.add_argument("--check-length", action="store_true", help="状态表长度自校验")
    ap.add_argument("--report", action="store_true", help="语义层报告（Region/Continuum/Scene/Part）")
    ap.add_argument("--boundaries", action="store_true", help="边界→面片映射统计（by_part 面索引/覆盖）")
    ap.add_argument("--binary-decode", type=int, default=0, help="二进制状态表：解码前 N 个带 raw 记录（int/double 段显示）")
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
                                args.fingerprint, args.check_length,
                                args.report, args.export, args.state_tree,
                                args.grammar, args.arrays]):
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

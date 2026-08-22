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
          - 顶点坐标 = Float8 数组，count/3 与面索引最大值吻合（1 基 => == max_index，
            0 基 => == max_index+1）；否则取 count%3==0 且数值绝对值 >1 的最大
            Float8 数组（法向量数组的值 ≤1，可区分）。
        返回 dict: vertices(N,3), faces(M,3), meta, flags(推断置信度)。
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
        # 顶点（按面索引最大值确定规模）
        cands = [(i, a) for i, a in enumerate(self.arrays) if a["type"] == "Float8"]
        if faces is not None and faces.size:
            mx = int(faces.max())
            for want in (mx, mx + 1):
                for i, a in cands:
                    if a["count"] == want * 3:
                        verts = self.array_data(i).reshape(-1, 3)
                        vflags = "face-max:%d" % want
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
            res["max_index"] = int(faces.max()) if faces.size else 0
            res["n_vertices"] = int(verts.shape[0])
            res["consistent"] = bool(res["max_index"] <= res["n_vertices"])
        return res

    def export_stl(self, out_path):
        """把抽取出的网格写成 ASCII STL（三角形面片）。"""
        m = self.extract_mesh()
        if m["vertices"] is None or m["faces"] is None:
            raise RuntimeError("无法抽取网格（缺少顶点/面数组）")
        v, f = m["vertices"], m["faces"]
        one_based = bool(f.size and int(f.min()) >= 1)
        idx = f - 1 if one_based else f
        idx = _np.clip(idx, 0, v.shape[0] - 1)
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
    ap.add_argument("--export", metavar="DIR", help="导出到目录（数组 .npy/.csv + JSON）")
    ap.add_argument("--max-records", type=int, default=0, help="--state 最多输出的记录数（0=全部）")
    args = ap.parse_args(argv)

    sim = SimFile(args.file)

    if args.summary or not any([args.sections, args.arrays, args.state,
                                args.objects, args.tree, args.layers,
                                args.validate, args.mesh, args.mesh_export,
                                args.export]):
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

    if args.export:
        sim.export(args.export)
        print("\n已导出到 %s" % args.export)


if __name__ == "__main__":
    main()

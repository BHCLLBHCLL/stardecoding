# -*- coding: utf-8 -*-
"""P 波 P2：场函数表达式求值器（与 STAR-CCM+ 官方表达式语法对齐）。

实现场函数表达式语言的**词法 / 语法 / 求值**三件套，覆盖：
- 算术   : + - * / %
- 逻辑   : == != > < >= <= && || （求值为 1.0 / 0.0）
- 条件   : (a ? b : c) 三元，可嵌套
- 数学   : sin cos tan asin acos atan atan2 sinh cosh tanh
           ceil exp abs floor log log10 sqrt pow fmod mod min max clamp
- 矢量   : mag mag2 unit dot cross （分量用 [i] 下标，Position[0/1/2] 取 x/y/z）
- 张量   : eigValue eigVector trace norm mag mag2 （$$$A.eigValue(i) 点方法）
- 插值   : interpolateTable(@Table("name"), "col", SPLINE|LINEAR, "units", ${Position}[i])
- 替代值 : alternateValue(expr1, expr2)，expr 求值失败/非有限时取 expr2
- 变量   : ${Name}（标量/参数）、$$Name（矢量场函数）、$$$Name（张量）
           $$Position、$${Position}[i]、$${Time} 等

值域：
- 标量 => float ；矢量 => tuple[3] ；张量 => tuple[tuple[3] x 3]；字符串 => str
全部用 Python 原生类型，避免 numpy/BLAS 在 occ/scdm 环境的 `0xc06d007f` 崩溃。

用法（无 GUI）：
    from field_fn import FieldFunction, EvalContext
    ff = FieldFunction("mag($$Velocity) + 0.1")
    print(ff.evaluate({"Velocity": (3.0, 4.0, 0.0)}))   # 5.1
"""
import math
import re


# ---------------------------------------------------------------------------
# 词法分析
# ---------------------------------------------------------------------------
class Token:
    """词法单元。kind 取：NUMBER/STRING/IDENT/DOLLAR/OP/LPAREN/RPAREN/LBRACKET/
    RBRACKET/DOT/COMMA/AT/EOF。dollars 记录 $ 引用类型（1 标量 / 2 矢量 / 3 张量）。"""
    __slots__ = ("kind", "value", "dollars", "pos")

    def __init__(self, kind, value, pos, dollars=0):
        self.kind = kind
        self.value = value
        self.pos = pos
        self.dollars = dollars

    def __repr__(self):
        return "Token(%s, %r, dollars=%d)" % (self.kind, self.value, self.dollars)


# 运算符（按最长匹配优先排序）
_OPERATORS = [
    "&&", "||", "==", "!=", ">=", "<=", "?", ":", ">", "<",
    "+", "-", "*", "/", "%", "!", "(", ")", "[", "]", ".", ",", "@", "$",
]

# 结构符映射为独立 token 种类，供解析器按名匹配
_STRUCTURAL_KINDS = {
    "(": "LPAREN", ")": "RPAREN", "[": "LBRACKET",
    "]": "RBRACKET", ".": "DOT", ",": "COMMA",
}


class _Lexer:
    """将源码切分为词法单元流。"""

    _WS = frozenset(" \t\r\n")
    _DIGIT = frozenset("0123456789")
    _IDENT_START = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
    _IDENT_CHARS = frozenset(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789"
    )

    def __init__(self, text):
        self.text = text
        self.i = 0
        self.n = len(text)

    def _peek(self, k=0):
        j = self.i + k
        return self.text[j] if j < self.n else ""

    def _advance(self):
        c = self.text[self.i]
        self.i += 1
        return c

    def tokenize(self):
        toks = []
        while self.i < self.n:
            c = self._peek()
            if c in self._WS:
                self.i += 1
                continue
            pos = self.i
            if c in self._DIGIT or (c == "." and self._peek(1) in self._DIGIT):
                toks.append(self._number(pos))
            elif c == "$":
                toks.append(self._dollar(pos))
            elif c == '"':
                toks.append(self._string(pos))
            elif c == "@":
                self.i += 1
                toks.append(Token("AT", "@", pos))
            elif c in self._IDENT_START:
                toks.append(self._ident(pos))
            else:
                op = self._operator()
                if op is None:
                    raise SyntaxError("P2 无法识别字符 %r (offset %d)" % (c, pos))
                # 结构符独立成种（LPAREN/RPAREN/LBRACKET/RBRACKET/DOT/COMMA），
                # 便于解析器按名匹配；其余运算符保留 OP 种类。
                kind = _STRUCTURAL_KINDS.get(op, "OP")
                toks.append(Token(kind, op, pos))
        toks.append(Token("EOF", "", self.n))
        return toks

    def _number(self, pos):
        start = self.i
        if self._peek() == ".":
            self.i += 1
            while self._peek() in self._DIGIT:
                self.i += 1
        else:
            while self._peek() in self._DIGIT:
                self.i += 1
            if self._peek() == "." and self._peek(1) in self._DIGIT:
                self.i += 1
                while self._peek() in self._DIGIT:
                    self.i += 1
        if self._peek() in ("e", "E"):
            exp_pos = self.i
            self.i += 1
            if self._peek() in ("+", "-"):
                self.i += 1
            if self._peek() in self._DIGIT:
                while self._peek() in self._DIGIT:
                    self.i += 1
            else:
                self.i = exp_pos
        txt = self.text[start:self.i]
        if txt == ".":
            raise SyntaxError("P2 非法数字 '.' (offset %d)" % pos)
        return Token("NUMBER", float(txt), pos)

    def _dollar(self, pos):
        cnt = 0
        while self._peek() == "$":
            cnt += 1
            self.i += 1
        if self._peek() == "{":
            self.i += 1
            name = self._read_until("}")
            if not name:
                raise SyntaxError("P2 空变量引用 $ (offset %d)" % pos)
            return Token("DOLLAR", name, pos, dollars=cnt)
        name = self._read_ident_text()
        if not name:
            raise SyntaxError("P2 $ 后缺少变量名 (offset %d)" % pos)
        return Token("DOLLAR", name, pos, dollars=cnt)

    def _string(self, pos):
        self.i += 1
        buf = []
        while self.i < self.n:
            c = self._advance()
            if c == "\\":
                if self.i < self.n:
                    buf.append(self._advance())
                continue
            if c == '"':
                return Token("STRING", "".join(buf), pos)
            buf.append(c)
        raise SyntaxError("P2 字符串未闭合 (offset %d)" % pos)

    def _ident(self, pos):
        name = self._read_ident_text()
        return Token("IDENT", name, pos)

    def _read_ident_text(self):
        start = self.i
        while self._peek() in self._IDENT_CHARS:
            self.i += 1
        return self.text[start:self.i]

    def _read_until(self, ch):
        start = self.i
        while self.i < self.n and self._peek() != ch:
            self.i += 1
        if self.i >= self.n:
            raise SyntaxError("P2 期望 %r，遇到结尾 (offset %d)" % (ch, start))
        out = self.text[start:self.i].strip()
        self.i += 1  # 消费 ch
        return out

    def _operator(self):
        for op in _OPERATORS:
            if self.text.startswith(op, self.i):
                self.i += len(op)
                return op
        return None


# ---------------------------------------------------------------------------
# 语法树节点
# ---------------------------------------------------------------------------
class _Node:
    __slots__ = ()

    def eval(self, ctx):  # pragma: no cover - 抽象
        raise NotImplementedError


class _Num(_Node):
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def eval(self, ctx):
        return self.value


class _Str(_Node):
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def eval(self, ctx):
        return self.value


class _Ref(_Node):
    __slots__ = ("name", "dollars")

    def __init__(self, name, dollars=1):
        self.name = name
        self.dollars = dollars

    def eval(self, ctx):
        return ctx.lookup(self.name, self.dollars)


class _VecLit(_Node):
    __slots__ = ("items",)

    def __init__(self, items):
        self.items = items

    def eval(self, ctx):
        return tuple(_as_number(it.eval(ctx)) for it in self.items)


class _Unary(_Node):
    __slots__ = ("op", "operand")

    def __init__(self, op, operand):
        self.op = op
        self.operand = operand

    def eval(self, ctx):
        v = self.operand.eval(ctx)
        return _unary(self.op, v)


class _BinOp(_Node):
    __slots__ = ("op", "left", "right")

    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

    def eval(self, ctx):
        if self.op == "&&":
            return 1.0 if (_truthy(self.left.eval(ctx)) and _truthy(self.right.eval(ctx))) else 0.0
        if self.op == "||":
            return 1.0 if (_truthy(self.left.eval(ctx)) or _truthy(self.right.eval(ctx))) else 0.0
        a = self.left.eval(ctx)
        b = self.right.eval(ctx)
        return _binary(self.op, a, b)


class _Ternary(_Node):
    __slots__ = ("cond", "then", "otherwise")

    def __init__(self, cond, then, otherwise):
        self.cond = cond
        self.then = then
        self.otherwise = otherwise

    def eval(self, ctx):
        return self.then.eval(ctx) if _truthy(self.cond.eval(ctx)) else self.otherwise.eval(ctx)


class _Call(_Node):
    __slots__ = ("name", "args")

    def __init__(self, name, args):
        self.name = name
        self.args = args

    def eval(self, ctx):
        if self.name in _LAZY_FUNCS:
            return _call_lazy(self.name, self.args, ctx)
        args = [a.eval(ctx) for a in self.args]
        return _call_function(self.name, args, ctx)


class _Index(_Node):
    __slots__ = ("target", "index")

    def __init__(self, target, index):
        self.target = target
        self.index = index

    def eval(self, ctx):
        t = self.target.eval(ctx)
        i = self.index.eval(ctx)
        return _index(t, i)


class _Method(_Node):
    __slots__ = ("target", "name", "args")

    def __init__(self, target, name, args):
        self.target = target
        self.name = name
        self.args = args

    def eval(self, ctx):
        t = self.target.eval(ctx)
        args = [a.eval(ctx) for a in self.args]
        return _method(t, self.name, args)


class _TableRef(_Node):
    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name

    def eval(self, ctx):
        return _TableHandle(self.name)


# ---------------------------------------------------------------------------
# 递归下降解析器
# ---------------------------------------------------------------------------
class _Parser:
    def __init__(self, text):
        self.tokens = _Lexer(text).tokenize()
        self.i = 0

    def _cur(self, k=0):
        return self.tokens[min(self.i + k, len(self.tokens) - 1)]

    def _next(self):
        t = self._cur()
        self.i += 1
        return t

    def _expect(self, kind, val=None):
        t = self._cur()
        if t.kind != kind or (val is not None and t.value != val):
            raise SyntaxError(
                "P2 语法错误 offset %d：期望 %s，得到 %s" % (t.pos, kind, t.kind)
            )
        return self._next()

    def parse(self):
        node = self._ternary()
        eof = self._expect("EOF")
        return node

    def _ternary(self):
        cond = self._logical_or()
        if self._cur().kind == "OP" and self._cur().value == "?":
            self._next()
            then = self._ternary()
            self._expect("OP", ":")
            otherwise = self._ternary()
            return _Ternary(cond, then, otherwise)
        return cond

    def _logical_or(self):
        node = self._logical_and()
        while self._cur().kind == "OP" and self._cur().value == "||":
            self._next()
            node = _BinOp("||", node, self._logical_and())
        return node

    def _logical_and(self):
        node = self._equality()
        while self._cur().kind == "OP" and self._cur().value == "&&":
            self._next()
            node = _BinOp("&&", node, self._equality())
        return node

    def _equality(self):
        node = self._relational()
        while self._cur().kind == "OP" and self._cur().value in ("==", "!="):
            op = self._next().value
            node = _BinOp(op, node, self._relational())
        return node

    def _relational(self):
        node = self._additive()
        while self._cur().kind == "OP" and self._cur().value in (">", "<", ">=", "<="):
            op = self._next().value
            node = _BinOp(op, node, self._additive())
        return node

    def _additive(self):
        node = self._multiplicative()
        while self._cur().kind == "OP" and self._cur().value in ("+", "-"):
            op = self._next().value
            node = _BinOp(op, node, self._multiplicative())
        return node

    def _multiplicative(self):
        node = self._unary()
        while self._cur().kind == "OP" and self._cur().value in ("*", "/", "%"):
            op = self._next().value
            node = _BinOp(op, node, self._unary())
        return node

    def _unary(self):
        t = self._cur()
        if t.kind == "OP" and t.value in ("-", "+", "!"):
            self._next()
            return _Unary(t.value, self._unary())
        return self._postfix()

    def _postfix(self):
        node = self._primary()
        while True:
            t = self._cur()
            if t.kind == "LBRACKET":
                self._next()
                idx = self._ternary()
                self._expect("RBRACKET")
                node = _Index(node, idx)
            elif t.kind == "DOT":
                self._next()
                name = self._expect("IDENT").value
                args = []
                if self._cur().kind == "LPAREN":
                    self._next()
                    args = self._args()
                    self._expect("RPAREN")
                node = _Method(node, name, args)
            else:
                break
        return node

    def _primary(self):
        t = self._cur()
        if t.kind == "NUMBER":
            self._next()
            return _Num(t.value)
        if t.kind == "STRING":
            self._next()
            return _Str(t.value)
        if t.kind == "DOLLAR":
            self._next()
            return _Ref(t.value, t.dollars)
        if t.kind == "AT":
            self._next()
            self._expect("IDENT")  # Table / GeometryPart
            self._expect("LPAREN")
            name = self._expect("STRING").value
            self._expect("RPAREN")
            return _TableRef(name)
        if t.kind == "IDENT":
            return self._ident_or_call()
        if t.kind == "LBRACKET":
            self._next()
            items = [self._ternary()]
            while self._cur().kind == "COMMA":
                self._next()
                items.append(self._ternary())
            self._expect("RBRACKET")
            return _VecLit(items)
        if t.kind == "LPAREN":
            self._next()
            node = self._ternary()
            self._expect("RPAREN")
            return node
        raise SyntaxError("P2 语法错误 offset %d：无法解析" % t.pos)

    def _ident_or_call(self):
        name = self._expect("IDENT").value
        if self._cur().kind == "LPAREN":
            self._next()
            args = self._args()
            self._expect("RPAREN")
            return _Call(name, args)
        # 裸枚举/常量：插值法 SPLINE/LINEAR 与布尔 true/false 不作为变量引用
        if name in ("SPLINE", "LINEAR"):
            return _Str(name)
        if name in ("true", "false"):
            return _Num(1.0 if name == "true" else 0.0)
        return _Ref(name, 1)

    def _args(self):
        args = []
        if self._cur().kind == "RPAREN":
            return args
        args.append(self._ternary())
        while self._cur().kind == "COMMA":
            self._next()
            args.append(self._ternary())
        return args


# ---------------------------------------------------------------------------
# 值操作（标量/矢量/张量/字符串，纯 Python，避免 BLAS）
# ---------------------------------------------------------------------------
def _is_scalar(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_vector(v):
    return isinstance(v, (tuple, list)) and len(v) == 3 and all(
        _is_scalar(x) for x in v
    )


def _is_tensor(v):
    return (
        isinstance(v, (tuple, list))
        and len(v) == 3
        and all(_is_vector(x) for x in v)
    )


def _as_number(v):
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if _is_scalar(v):
        return float(v)
    raise TypeError("P2 期望标量，得到 %s" % _type_name(v))


def _vector_map(v, fn):
    return tuple(fn(x) for x in v)


def _scalarize(v):
    """把张量/矢量的标量化视图用于 mag/norm：张量取矩阵范数，矢量取欧氏范数。"""
    if _is_scalar(v):
        return float(v)
    if _is_vector(v):
        return math.sqrt(sum(x * x for x in v))
    if _is_tensor(v):
        return math.sqrt(sum(x * x for row in v for x in row))
    raise TypeError("P2 无法标量化 %s" % _type_name(v))


def _type_name(v):
    if _is_tensor(v):
        return "张量"
    if _is_vector(v):
        return "矢量"
    if isinstance(v, str):
        return "字符串"
    if _is_scalar(v):
        return "标量"
    return type(v).__name__


def _truthy(v):
    if isinstance(v, bool):
        return v
    if _is_scalar(v):
        return v != 0.0
    if _is_vector(v):
        return any(x != 0.0 for x in v)
    if _is_tensor(v):
        return any(x != 0.0 for row in v for x in row)
    return bool(v)


def _binary(op, a, b):
    if op in ("==", "!="):
        eq = _equal(a, b)
        return 1.0 if (eq == (op == "==")) else 0.0
    if op in (">", "<", ">=", "<="):
        if _is_scalar(a) and _is_scalar(b):
            return 1.0 if _ordered(op, a, b) else 0.0
        raise TypeError("P2 关系运算仅支持标量：%s %s %s" % (_type_name(a), op, _type_name(b)))

    # 算术：+ - * / %
    if _is_scalar(a) and _is_scalar(b):
        return _scalar_arith(op, a, b)
    if _is_vector(a) and _is_vector(b):
        return _vector_arith(op, a, b)
    if _is_tensor(a) and _is_tensor(b):
        return _tensor_arith(op, a, b)
    # 标量 <-> 矢量/张量（广播：标量作用于每个分量）
    if _is_scalar(a) and _is_vector(b):
        return _vector_map(b, lambda x: _scalar_arith(op, a, x))
    if _is_vector(a) and _is_scalar(b):
        return _vector_map(a, lambda x: _scalar_arith(op, x, b))
    if _is_scalar(a) and _is_tensor(b):
        return tuple(_vector_map(row, lambda x: _scalar_arith(op, a, x)) for row in b)
    if _is_tensor(a) and _is_scalar(b):
        return tuple(_vector_map(row, lambda x: _scalar_arith(op, x, b)) for row in a)
    raise TypeError("P2 不支持 %s %s %s" % (_type_name(a), op, _type_name(b)))


def _equal(a, b):
    if _is_scalar(a) and _is_scalar(b):
        return a == b
    if _is_vector(a) and _is_vector(b):
        return tuple(a) == tuple(b)
    if _is_tensor(a) and _is_tensor(b):
        return tuple(tuple(r) for r in a) == tuple(tuple(r) for r in b)
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    if isinstance(a, _TableHandle) and isinstance(b, _TableHandle):
        return a.name == b.name
    return False


def _ordered(op, a, b):
    if op == ">":
        return a > b
    if op == "<":
        return a < b
    if op == ">=":
        return a >= b
    return a <= b


def _scalar_arith(op, a, b):
    if op == "+":
        return float(a + b)
    if op == "-":
        return float(a - b)
    if op == "*":
        return float(a * b)
    if op == "/":
        if b == 0:
            raise ZeroDivisionError("P2 除零")
        return float(a / b)
    if op == "%":
        if b == 0:
            raise ZeroDivisionError("P2 除零")
        return float(math.fmod(a, b))
    raise TypeError("P2 未知标量运算符 %s" % op)


def _vector_arith(op, a, b):
    if op == "*":
        return tuple(x * y for x, y in zip(a, b))
    return tuple(_scalar_arith(op, x, y) for x, y in zip(a, b))


def _tensor_arith(op, a, b):
    return tuple(_vector_arith(op, ra, rb) for ra, rb in zip(a, b))


def _unary(op, v):
    if op == "+":
        return v
    if op == "-":
        if _is_scalar(v):
            return float(-v)
        if _is_vector(v):
            return tuple(-x for x in v)
        if _is_tensor(v):
            return tuple(tuple(-x for x in row) for row in v)
        raise TypeError("P2 无法对 %s 取负" % _type_name(v))
    if op == "!":
        return 0.0 if _truthy(v) else 1.0
    raise TypeError("P2 未知一元运算符 %s" % op)


def _index(t, i):
    if isinstance(i, bool):
        i = int(i)
    if not _is_scalar(i):
        raise TypeError("P2 下标必须是整数")
    i = int(i)
    if _is_vector(t) or isinstance(t, (tuple, list)):
        if not (0 <= i < len(t)):
            raise IndexError("P2 下标 %d 越界 (len=%d)" % (i, len(t)))
        return t[i]
    raise TypeError("P2 无法对 %s 取下标" % _type_name(t))


def _method(t, name, args):
    if _is_vector(t) and name in ("x", "y", "z"):
        if args:
            raise TypeError("P2 分量访问 %s 不接受参数" % name)
        return t[{"x": 0, "y": 1, "z": 2}[name]]
    if _is_tensor(t) and name in _TENSOR_FUNCS:
        return _call_function(name, [t] + list(args), None)
    if name in ("length", "size") and (isinstance(t, (tuple, list))):
        if args:
            raise TypeError("P2 %s 不接受参数" % name)
        return float(len(t))
    raise TypeError("P2 %s 类型无方法 / 属性 %s" % (_type_name(t), name))


# ---------------------------------------------------------------------------
# 表 / 插值
# ---------------------------------------------------------------------------
class _TableHandle:
    """@Table("name") 解析产物，携带表名；不持有数据，数据由 EvalContext 提供。"""

    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "@Table(%r)" % self.name


class Table:
    """插值用一维回归表：首列为独立变量（如 x/时间），其余列为待插值列。"""

    def __init__(self, name, x, columns):
        self.name = name
        self.x = [float(v) for v in x]
        self.columns = {k: [float(v) for v in vals] for k, vals in columns.items()}

    def column(self, name):
        if name not in self.columns:
            raise KeyError("P2 表 %r 无列 %r（可用：%s）" % (self.name, name, ", ".join(self.columns)))
        return self.columns[name]

    def interpolate(self, col_name, method, x):
        xs = self.x
        ys = self.column(col_name)
        if len(xs) != len(ys):
            raise ValueError("P2 表 %r 列 %r 长度不一致" % (self.name, col_name))
        if not xs:
            raise ValueError("P2 表 %r 为空" % self.name)
        if method == "LINEAR":
            return _interp_linear(xs, ys, x)
        if method == "SPLINE":
            return _interp_spline(xs, ys, x)
        raise ValueError("P2 未知插值法 %r（用 SPLINE / LINEAR）" % method)


def _interp_linear(xs, ys, x):
    n = len(xs)
    if n == 1:
        return float(ys[0])
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    for i in range(1, n):
        if xs[i - 1] <= x <= xs[i]:
            x0, x1 = xs[i - 1], xs[i]
            if x1 == x0:
                return float(ys[i])
            t = (x - x0) / (x1 - x0)
            return float(ys[i - 1] * (1.0 - t) + ys[i] * t)
    return float(ys[-1])


def _interp_spline(xs, ys, x):
    """Natural cubic spline（连续二阶导）。STAR-CCM+ SPLINE 用于平滑表插值。"""
    n = len(xs)
    if n == 1:
        return float(ys[0])
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    # 二次导数 h/delta
    h = [xs[i + 1] - xs[i] for i in range(n - 1)]
    if any(v == 0 for v in h):
        return _interp_linear(xs, ys, x)
    alpha = [0.0] * n
    for i in range(1, n - 1):
        alpha[i] = 3.0 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1])
    c = [0.0] * n
    l = [1.0] * n
    mu = [0.0] * n
    z = [0.0] * n
    for i in range(1, n - 1):
        l[i] = 2.0 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * mu[i - 1]
        if l[i] == 0:
            return _interp_linear(xs, ys, x)
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]
    b = [0.0] * (n - 1)
    c[n - 1] = 0.0
    d = [0.0] * (n - 1)
    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (ys[j + 1] - ys[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])
    for i in range(n - 1):
        if xs[i] <= x <= xs[i + 1]:
            dx = x - xs[i]
            return float(
                ys[i]
                + b[i] * dx
                + c[i] * dx * dx
                + d[i] * dx * dx * dx
            )
    return float(ys[-1])


# ---------------------------------------------------------------------------
# 内置函数表
# ---------------------------------------------------------------------------
def _trig(fn, name):
    def impl(args):
        if len(args) != 1:
            raise TypeError("P2 %s 需要 1 个参数" % name)
        return float(fn(_as_number(args[0])))
    return impl


def _atan2(args):
    if len(args) != 2:
        raise TypeError("P2 atan2 需要 2 个参数")
    return float(math.atan2(_as_number(args[0]), _as_number(args[1])))


def _pow(args):
    if len(args) != 2:
        raise TypeError("P2 pow 需要 2 个参数")
    return float(pow(_as_number(args[0]), _as_number(args[1])))


def _fmod(args):
    if len(args) != 2:
        raise TypeError("P2 fmod/mod 需要 2 个参数")
    a, b = _as_number(args[0]), _as_number(args[1])
    if b == 0:
        raise ZeroDivisionError("P2 除零")
    return float(math.fmod(a, b))


def _mod(args):
    return _fmod(args)


def _min(args):
    if len(args) != 2:
        raise TypeError("P2 min 需要 2 个参数")
    return float(min(_as_number(args[0]), _as_number(args[1])))


def _max(args):
    if len(args) != 2:
        raise TypeError("P2 max 需要 2 个参数")
    return float(max(_as_number(args[0]), _as_number(args[1])))


def _clamp(args):
    if len(args) != 3:
        raise TypeError("P2 clamp 需要 3 个参数")
    v, lo, hi = _as_number(args[0]), _as_number(args[1]), _as_number(args[2])
    return float(min(max(v, lo), hi))


def _mag(args):
    if len(args) != 1:
        raise TypeError("P2 mag/mag2 需要 1 个参数")
    v = args[0]
    s = _scalarize(v)
    return float(s)


def _mag2(args):
    if len(args) != 1:
        raise TypeError("P2 mag/mag2 需要 1 个参数")
    v = args[0]
    if _is_tensor(v):
        return float(sum(x * x for row in v for x in row))
    if _is_vector(v):
        return float(sum(x * x for x in v))
    return float(_as_number(v) * _as_number(v))


def _unit(args):
    if len(args) != 1 and len(args) != 2:
        raise TypeError("P2 unit(u[,d]) 需要 1/2 个参数")
    v = args[0]
    if not _is_vector(v):
        raise TypeError("P2 unit 需要矢量")
    m = math.sqrt(sum(x * x for x in v))
    if m == 0:
        return tuple(0.0 for _ in v)
    u = tuple(x / m for x in v)
    if len(args) == 2:
        return float(u[int(_as_number(args[1]))])
    return u


def _dot(args):
    if len(args) != 2:
        raise TypeError("P2 dot 需要 2 个参数")
    a, b = args
    if _is_tensor(a) and _is_tensor(b):
        return float(sum(x * y for ra, rb in zip(a, b) for x, y in zip(ra, rb)))
    if _is_vector(a) and _is_vector(b):
        return float(sum(x * y for x, y in zip(a, b)))
    raise TypeError("P2 dot 需要矢量/张量，得到 %s/%s" % (_type_name(a), _type_name(b)))


def _cross(args):
    if len(args) != 2:
        raise TypeError("P2 cross 需要 2 个参数")
    a, b = args
    if not (_is_vector(a) and _is_vector(b)):
        raise TypeError("P2 cross 需要矢量")
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _curl(args):
    if len(args) != 1:
        raise TypeError("P2 curl 需要 1 个参数（矢量场）")
    raise NotImplementedError("P2 curl 依赖网格/场梯度，暂未内置")


def _div(args):
    if len(args) != 1:
        raise TypeError("P2 div 需要 1 个参数（矢量场）")
    raise NotImplementedError("P2 div 依赖网格/场梯度，暂未内置")


def _grad(args):
    if len(args) != 1:
        raise TypeError("P2 grad 需要 1 个参数")
    raise NotImplementedError("P2 grad 依赖网格/场梯度，暂未内置")


def _trace(args):
    if len(args) != 1:
        raise TypeError("P2 trace 需要 1 个参数")
    a = args[0]
    if not _is_tensor(a):
        raise TypeError("P2 trace 需要张量")
    return float(a[0][0] + a[1][1] + a[2][2])


def _norm(args):
    if len(args) != 1 and len(args) != 2:
        raise TypeError("P2 norm(A[,type]) 需要 1/2 个参数")
    a = args[0]
    if not _is_tensor(a):
        raise TypeError("P2 norm 需要张量")
    if len(args) == 1:
        return float(math.sqrt(sum(x * x for row in a for x in row)))
    spec = args[1]
    if isinstance(spec, str):
        s = spec.lower()
        if s in ("frobenius", "frob", "fro"):
            return float(math.sqrt(sum(x * x for row in a for x in row)))
        if s == "infinity" or s == "inf":
            return float(max(sum(abs(x) for x in row) for row in a))
        raise ValueError("P2 未知张量范数 %r" % spec)
    k = _as_number(spec)
    if k == 1:
        return float(max(sum(abs(a[r][c]) for r in range(3)) for c in range(3)))
    if k == 2:
        return float(math.sqrt(sum(x * x for row in a for x in row)))
    raise ValueError("P2 未知张量范数 %s" % k)


def _eig_value(args):
    if len(args) != 2:
        raise TypeError("P2 eigValue 需要 2 个参数")
    a, i = args[0], int(_as_number(args[1]))
    if not _is_tensor(a):
        raise TypeError("P2 eigValue 需要张量")
    evals, _ = _jacobi(a)
    if not (0 <= i < 3):
        raise IndexError("P2 eigValue 下标 %d 越界" % i)
    return float(evals[i])


def _eig_vector(args):
    if len(args) != 2:
        raise TypeError("P2 eigVector 需要 2 个参数")
    a, i = args[0], int(_as_number(args[1]))
    if not _is_tensor(a):
        raise TypeError("P2 eigVector 需要张量")
    _, evecs = _jacobi(a)
    if not (0 <= i < 3):
        raise IndexError("P2 eigVector 下标 %d 越界" % i)
    return tuple(evecs[i])


def _jacobi(a, tol=1e-9, max_n=100):
    """对称 3x3 矩阵 Jacobi 特征值分解，返回 (特征值列表, 特征向量行矩阵)。"""
    rows = [[float(a[r][c]) for c in range(3)] for r in range(3)]
    v = [[1.0 if r == c else 0.0 for c in range(3)] for r in range(3)]
    for _ in range(max_n):
        p, q = 0, 1
        mx = abs(rows[p][q])
        for i in range(3):
            for j in range(i + 1, 3):
                if abs(rows[i][j]) > mx:
                    mx, p, q = abs(rows[i][j]), i, j
        if mx < tol:
            break
        if rows[p][q] == 0:
            break
        theta = (rows[q][q] - rows[p][p]) / (2.0 * rows[p][q])
        t = 1.0 if theta >= 0 else -1.0
        t = t / (abs(theta) + math.sqrt(theta * theta + 1.0)) if theta != 0 else 0.0
        c = 1.0 / math.sqrt(1.0 + t * t)
        s = t * c
        for k in range(3):
            akp, akq = rows[k][p], rows[k][q]
            rows[k][p] = c * akp - s * akq
            rows[k][q] = s * akp + c * akq
        for k in range(3):
            pk, qk = rows[p][k], rows[q][k]
            rows[p][k] = c * pk - s * qk
            rows[q][k] = s * pk + c * qk
        for k in range(3):
            vkp, vkq = v[k][p], v[k][q]
            v[k][p] = c * vkp - s * vkq
            v[k][q] = s * vkp + c * vkq
    evals = [rows[i][i] for i in range(3)]
    evecs = [tuple(v[r][i] for r in range(3)) for i in range(3)]
    order = sorted(range(3), key=lambda i: evals[i])
    evals = [evals[i] for i in order]
    evecs = [evecs[i] for i in order]
    return evals, evecs


def _alternate_value(args, ctx=None):
    if len(args) != 2:
        raise TypeError("P2 alternateValue 需要 2 个参数")
    return args[1]


def _finite(v):
    """判断值是否有限（非 NaN/Inf）。对标量/矢量/张量逐一检查。"""
    if _is_scalar(v):
        return math.isfinite(v)
    if _is_vector(v):
        return all(math.isfinite(x) for x in v)
    if _is_tensor(v):
        return all(math.isfinite(x) for row in v for x in row)
    return True


def _call_lazy(name, arg_nodes, ctx):
    if name in ("alternateValue", "altValue"):
        if not arg_nodes:
            raise TypeError("P2 alternateValue 需要至少 1 个参数")
        # 官方语义：依次求值各表达式，取首个可求值且有限者；全部失败则抛错
        for node in arg_nodes:
            try:
                v = node.eval(ctx)
            except Exception:
                continue
            if _finite(v):
                return v
        raise ValueError("P2 alternateValue 所有参数均无法求值/非有限")
    raise NameError("P2 未知惰性函数 %s" % name)


def _inside_part(args):
    raise NotImplementedError("P2 insidePart 依赖几何部件，暂未内置")


def _distance_to_part(args):
    raise NotImplementedError("P2 distanceToPart 依赖几何部件，暂未内置")


def _interpolate_table(args):
    if len(args) < 5:
        raise TypeError("P2 interpolateTable 需要 5 个参数")
    table = args[0]
    col = args[1]
    method = args[2]
    x = args[4]
    if not isinstance(table, _TableHandle):
        raise TypeError("P2 interpolateTable 第 1 参数需为 @Table(...)")
    if not isinstance(col, str):
        raise TypeError("P2 interpolateTable 第 2 参数需为列名字符串")
    m = _to_interp_method(method)
    if isinstance(table, _TableHandle):
        tbl = _ctx_current_tables.get(table.name)
        if tbl is None:
            raise KeyError("P2 上下文中无表 %r" % table.name)
        return float(tbl.interpolate(col, m, _as_number(x)))
    raise TypeError("P2 interpolateTable 第 1 参数需为 @Table(...)")


def _to_interp_method(method):
    if isinstance(method, str):
        s = method.strip().lower()
        if s in ("spline", "natural", "cubic"):
            return "SPLINE"
        if s == "linear":
            return "LINEAR"
        raise ValueError("P2 未知插值法 %r" % method)
    if isinstance(method, str):
        return method
    raise TypeError("P2 插值法参数需为字符串")


_MATH_FUNCS = {
    "sin": _trig(math.sin, "sin"),
    "cos": _trig(math.cos, "cos"),
    "tan": _trig(math.tan, "tan"),
    "asin": _trig(math.asin, "asin"),
    "acos": _trig(math.acos, "acos"),
    "atan": _trig(math.atan, "atan"),
    "sinh": _trig(math.sinh, "sinh"),
    "cosh": _trig(math.cosh, "cosh"),
    "tanh": _trig(math.tanh, "tanh"),
    "atan2": _atan2,
    "ceil": _trig(math.ceil, "ceil"),
    "exp": _trig(math.exp, "exp"),
    "abs": _trig(abs, "abs"),
    "floor": _trig(math.floor, "floor"),
    "log": _trig(math.log, "log"),
    "log10": _trig(math.log10, "log10"),
    "sqrt": _trig(math.sqrt, "sqrt"),
    "pow": _pow,
    "fmod": _fmod,
    "mod": _mod,
    "min": _min,
    "max": _max,
    "clamp": _clamp,
}

_VECTOR_FUNCS = {
    "mag": _mag,
    "mag2": _mag2,
    "unit": _unit,
    "dot": _dot,
    "cross": _cross,
}

_TENSOR_FUNCS = {
    "eigValue": _eig_value,
    "eigVector": _eig_vector,
    "trace": _trace,
    "norm": _norm,
    "mag": _mag,
    "mag2": _mag2,
}

_SPECIAL_FUNCS = {
    "altValue": _alternate_value,
    "alternateValue": _alternate_value,
    "interpolateTable": _interpolate_table,
    "insidePart": _inside_part,
    "distanceToPart": _distance_to_part,
    "curl": _curl,
    "div": _div,
    "grad": _grad,
}

# 惰性求值函数：参数以 AST 节点传入，而非先求值（供 alternateValue 回退语义）
_LAZY_FUNCS = frozenset({"alternateValue", "altValue"})

_FUNCTIONS = {}
_FUNCTIONS.update(_MATH_FUNCS)
_FUNCTIONS.update(_VECTOR_FUNCS)
_FUNCTIONS.update(_TENSOR_FUNCS)
_FUNCTIONS.update(_SPECIAL_FUNCS)


# 全局表注册表（_ctx_current_tables 在求值期间由 EvalContext 临时托管）
_ctx_current_tables = {}


def _call_function(name, args, ctx):
    fn = _FUNCTIONS.get(name)
    if fn is None:
        raise NameError("P2 未知函数 %s" % name)
    return fn(args)


# ---------------------------------------------------------------------------
# 求值上下文
# ---------------------------------------------------------------------------
class EvalContext:
    """表达式求值上下文。

    变量按名称存储：名字 -> 标量 / 矢量 / 张量 / 表。内置 {Position}、{Time}。
    tables 为 ``名字 -> Table`` 映射，供 interpolateTable 使用。可用
    ``validate=False`` 跳过运行时表/变量预检，仅做语法编译。
    """

    def __init__(self, variables=None, tables=None, position=(0.0, 0.0, 0.0), time=0.0):
        self.variables = dict(variables or {})
        self.tables = dict(tables or {})
        self.position = tuple(position)
        self.time = float(time)

    def lookup(self, name, dollars=1):
        if name == "Position":
            return self.position
        if name == "Time":
            return self.time
        if name in self.variables:
            return self.variables[name]
        raise NameError("P2 未定义变量 $%s%s" % ("$" * (dollars - 1), name))


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------
class FieldFunction:
    """已编译的场函数表达式。

    ``compile(expr)`` 解析并缓存 AST；``evaluate(**vars)``（或 ``eval``）在给定
    变量上下文中求值。``numeric(expr, ctx)`` 便捷函数自动判断标量/矢量/张量。
    """

    __slots__ = ("source", "ast")

    def __init__(self, expression):
        self.source = expression
        self.ast = _Parser(expression).parse()

    def evaluate(self, variables=None, tables=None, position=(0.0, 0.0, 0.0), time=0.0):
        global _ctx_current_tables
        ctx = EvalContext(variables, tables, position, time)
        old = _ctx_current_tables
        _ctx_current_tables = ctx.tables
        try:
            return self.ast.eval(ctx)
        finally:
            _ctx_current_tables = old

    def eval(self, variables=None, tables=None, position=(0.0, 0.0, 0.0), time=0.0):
        return self.evaluate(variables, tables, position, time)

    def __call__(self, variables=None, tables=None, position=(0.0, 0.0, 0.0), time=0.0):
        return self.evaluate(variables, tables, position, time)

    def __repr__(self):
        return "FieldFunction(%r)" % self.source


def compile_expression(expression):
    """仅解析为 AST，不依赖运行时上下文（用于批量预检）。"""
    return _Parser(expression).parse()


def numeric(expression, variables=None, tables=None, position=(0.0, 0.0, 0.0), time=0.0):
    """单次表达式求值，返回标量/矢量/张量。"""
    return FieldFunction(expression).evaluate(variables, tables, position, time)

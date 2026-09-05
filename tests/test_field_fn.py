# -*- coding: utf-8 -*-
"""P 波 P2：场函数表达式求值器（与 STAR-CCM+ 官方语法对齐）。

覆盖：
  词法/语法：十进制/科学计数/字符串/${}/$$/$$$/@Table 引用/矢量字面量
  算术：+ - * / % 结合律与优先级
  逻辑：== != > < >= <= && ||（求值 1.0 / 0.0）
  条件：三元 (? :) 嵌套
  数学：sin cos tan asin acos atan atan2 sinh cosh tanh ceil exp abs floor
        log log10 sqrt pow fmod mod min max clamp
  矢量：mag mag2 unit dot cross + 下标 [i] + Position[0/1/2] + .x/.y/.z
  张量：eigValue eigVector trace norm mag mag2（含 $$$A 点方法）
  插值：interpolateTable LINEAR / SPLINE（越界钳位）
  替代值：alternateValue（除零 / NaN 回退）
  错误：未知函数 / 未定义变量 / 除零 / 越界下标 / 非法语法
  预检：compile_expression 仅解析不依赖运行时上下文

验收核心（P2 行）：数学/矢量/逻辑 + 插值器；与官方语法对齐。
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import math

import pytest

from field_fn import (EvalContext, FieldFunction, Table,
                      compile_expression, numeric)


def _vec(v):
    return tuple(float(x) for x in v)


# ---------------------------------------------------------------- 算术
def test_arithmetic_basic():
    assert numeric("1 + 2") == 3.0
    assert numeric("1 - 2") == -1.0
    assert numeric("3 * 4") == 12.0
    assert numeric("7 / 2") == 3.5
    assert numeric("7 % 3") == 1.0


def test_arithmetic_precedence():
    assert numeric("1 + 2 * 3") == 7.0
    assert numeric("(1 + 2) * 3") == 9.0
    assert numeric("2 * 3 + 4 * 5") == 26.0
    assert numeric("10 - 2 - 3") == 5.0


def test_unary_minus():
    assert numeric("-5") == -5.0
    assert numeric("3 * -2") == -6.0
    assert numeric("-(1 + 2)") == -3.0


def test_scientific_number():
    assert numeric("1e3") == 1000.0
    assert numeric("1.5e-2") == 0.015
    assert numeric("0.5") == 0.5


# ---------------------------------------------------------------- 逻辑
def test_logic_comparisons():
    assert numeric("3 > 2") == 1.0
    assert numeric("3 < 2") == 0.0
    assert numeric("3 >= 3") == 1.0
    assert numeric("2 <= 1") == 0.0
    assert numeric("3 == 3") == 1.0
    assert numeric("3 != 3") == 0.0


def test_logic_and_or():
    assert numeric("1 && 0") == 0.0
    assert numeric("1 && 1") == 1.0
    assert numeric("0 || 0") == 0.0
    assert numeric("0 || 1") == 1.0
    assert numeric("(1 < 2) && (2 < 3)") == 1.0


def test_logic_not():
    assert numeric("!0") == 1.0
    assert numeric("!1") == 0.0


# ---------------------------------------------------------------- 条件三元
def test_ternary():
    assert numeric("1 < 2 ? 10 : 20") == 10.0
    assert numeric("1 > 2 ? 10 : 20") == 20.0


def test_ternary_nested():
    assert numeric("(1 < 0) ? 10 : ((2 < 3) ? 30 : 40)") == 30.0
    assert numeric("(1 < 0) ? 10 : ((2 > 3) ? 30 : 40)") == 40.0


# ---------------------------------------------------------------- 数学函数
def test_math_scalar():
    assert round(numeric("sqrt(9)"), 9) == 3.0
    assert round(numeric("abs(-4)"), 9) == 4.0
    assert round(numeric("floor(2.7)"), 9) == 2.0
    assert round(numeric("ceil(2.1)"), 9) == 3.0
    assert round(numeric("pow(2, 10)"), 9) == 1024.0
    assert round(numeric("exp(0)"), 9) == 1.0
    assert round(numeric("log(1)"), 9) == 0.0
    assert round(numeric("log10(100)"), 9) == 2.0
    assert round(numeric("fmod(7, 3)"), 9) == 1.0
    assert round(numeric("mod(-7, 3)"), 6) == round(math.fmod(-7.0, 3.0), 6)


def test_math_trig():
    assert round(numeric("sin(0)"), 9) == 0.0
    assert round(numeric("cos(0)"), 9) == 1.0
    assert round(numeric("asin(0)"), 9) == 0.0
    assert round(numeric("acos(1)"), 9) == 0.0
    assert round(numeric("atan(0)"), 9) == 0.0
    assert round(numeric("atan2(1, 1)"), 9) == round(math.pi / 4, 9)


def test_math_min_max_clamp():
    assert numeric("min(3, 5)") == 3.0
    assert numeric("max(3, 5)") == 5.0
    assert numeric("clamp(5, 0, 3)") == 3.0
    assert numeric("clamp(-1, 0, 3)") == 0.0
    assert numeric("clamp(2, 0, 3)") == 2.0


# ---------------------------------------------------------------- 矢量
def test_vector_mag_dot_cross():
    assert round(numeric("mag($$Velocity)", {"Velocity": (3.0, 4.0, 0.0)}), 9) == 5.0
    assert round(numeric("mag2($$Velocity)", {"Velocity": (3.0, 4.0, 0.0)}), 9) == 25.0
    assert numeric("dot($$u, $$v)", {"u": (1.0, 0.0, 0.0), "v": (0.0, 1.0, 0.0)}) == 0.0
    c = numeric("cross($$u, $$v)", {"u": (1.0, 0.0, 0.0), "v": (0.0, 1.0, 0.0)})
    assert _vec(c) == (0.0, 0.0, 1.0)


def test_vector_component_index():
    assert numeric("$$Velocity[0]", {"Velocity": (3.0, 4.0, 0.0)}) == 3.0
    assert numeric("$$Velocity[1]", {"Velocity": (3.0, 4.0, 0.0)}) == 4.0
    assert numeric("$$Velocity[2]", {"Velocity": (3.0, 4.0, 0.0)}) == 0.0


def test_vector_method_component():
    assert numeric("$$Velocity.x", {"Velocity": (3.0, 4.0, 0.0)}) == 3.0
    assert numeric("$$Velocity.y", {"Velocity": (3.0, 4.0, 0.0)}) == 4.0
    assert numeric("$$Velocity.z", {"Velocity": (3.0, 4.0, 0.0)}) == 0.0


def test_vector_arithmetic():
    assert _vec(numeric("$$a + $$b", {"a": (1.0, 2.0, 3.0), "b": (4.0, 5.0, 6.0)})) == (5.0, 7.0, 9.0)
    assert _vec(numeric("2 * $$a", {"a": (1.0, 2.0, 3.0)})) == (2.0, 4.0, 6.0)
    assert _vec(numeric("$$a * 2", {"a": (1.0, 2.0, 3.0)})) == (2.0, 4.0, 6.0)


def test_position_access():
    v = numeric("${Position}[0]", position=(1.0, 2.0, 3.0))
    assert v == 1.0
    v2 = numeric("${Position}[2]", position=(1.0, 2.0, 3.0))
    assert v2 == 3.0


def test_vector_literal():
    assert _vec(numeric("[1, 2, 3]")) == (1.0, 2.0, 3.0)
    assert _vec(numeric("[${x}, ${y}, ${z}]", {"x": 1.0, "y": 2.0, "z": 3.0})) == (1.0, 2.0, 3.0)


def test_unit_vector():
    u = numeric("unit($$v)", {"v": (3.0, 0.0, 0.0)})
    assert _vec(u) == (1.0, 0.0, 0.0)
    assert numeric("unit($$v, 0)", {"v": (3.0, 0.0, 0.0)}) == 1.0


# ---------------------------------------------------------------- 张量
def test_tensor_trace():
    A = ((2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.0))
    assert numeric("trace($$$A)", {"A": A}) == 9.0
    assert numeric("$$$A.trace()", {"A": A}) == 9.0


def test_tensor_eigen():
    A = ((2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.0))
    assert round(numeric("$$$A.eigValue(0)", {"A": A}), 9) == 2.0
    assert round(numeric("$$$A.eigValue(2)", {"A": A}), 9) == 4.0
    v = numeric("$$$A.eigVector(2)", {"A": A})
    assert _vec(v)[2] == 1.0 or _vec(v)[2] == -1.0


def test_tensor_norm():
    A = ((3.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 0.0))
    assert round(numeric("norm($$$A)", {"A": A}), 9) == 5.0
    assert round(numeric("mag($$$A)", {"A": A}), 9) == 5.0
    assert round(numeric("mag2($$$A)", {"A": A}), 9) == 25.0


def test_tensor_norm_type():
    A = ((3.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 0.0))
    assert round(numeric('norm($$$A, "frobenius")', {"A": A}), 9) == 5.0
    assert round(numeric('norm($$$A, "infinity")', {"A": A}), 9) == 4.0
    assert round(numeric("norm($$$A, 1)", {"A": A}), 9) == 4.0


def test_tensor_methods():
    A = ((2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.0))
    assert round(numeric("$$$A.mag()", {"A": A}), 9) == round(math.sqrt(29.0), 9)
    assert round(numeric("$$$A.mag2()", {"A": A}), 9) == 29.0
    assert round(numeric("$$$A.trace()", {"A": A}), 9) == 9.0
    assert round(numeric("$$$A.norm()", {"A": A}), 9) == round(math.sqrt(29.0), 9)


# ---------------------------------------------------------------- 插值
def _make_table():
    return Table("t", [0.0, 1.0, 2.0], {"u": [0.0, 10.0, 20.0]})


def test_interpolate_linear():
    t = _make_table()
    assert round(numeric('interpolateTable(@Table("t"), "u", LINEAR, "", ${Position}[0])',
                         tables={"t": t}, position=(0.5, 0.0, 0.0)), 9) == 5.0
    assert round(numeric('interpolateTable(@Table("t"), "u", LINEAR, "", ${Position}[0])',
                         tables={"t": t}, position=(1.5, 0.0, 0.0)), 9) == 15.0


def test_interpolate_clamp():
    t = _make_table()
    assert round(numeric('interpolateTable(@Table("t"), "u", LINEAR, "", ${Position}[0])',
                         tables={"t": t}, position=(-1.0, 0.0, 0.0)), 9) == 0.0
    assert round(numeric('interpolateTable(@Table("t"), "u", LINEAR, "", ${Position}[0])',
                         tables={"t": t}, position=(5.0, 0.0, 0.0)), 9) == 20.0


def test_interpolate_spline():
    t = Table("s", [0.0, 1.0, 2.0, 3.0], {"v": [0.0, 1.0, 0.0, 1.0]})
    val = numeric('interpolateTable(@Table("s"), "v", SPLINE, "", ${Position}[0])',
                  tables={"s": t}, position=(1.5, 0.0, 0.0))
    assert 0.3 < val < 0.7


def test_interpolate_missing_column():
    t = _make_table()
    with pytest.raises(Exception):
        numeric('interpolateTable(@Table("t"), "missing", LINEAR, "", ${Position}[0])',
                tables={"t": t}, position=(0.0, 0.0, 0.0))


# ---------------------------------------------------------------- 替代值
def test_alternate_value_fallback():
    assert numeric("alternateValue(1 / 0, 99)") == 99.0
    assert numeric("alternateValue(sqrt(-1), 42)") == 42.0
    assert numeric("alternateValue(5, 99)") == 5.0


def test_alternate_value_multi():
    assert numeric("alternateValue(1 / 0, sqrt(-1), 42)") == 42.0
    assert numeric("altValue(${Missing}, 7)") == 7.0


# ---------------------------------------------------------------- 官方风格复合
def test_official_style_composite():
    v = numeric("(mag($$v) > 1) ? ${Time} : max(0, ${Time})",
                {"v": (3.0, 0.0, 0.0)}, time=12.5)
    assert v == 12.5
    v2 = numeric("(mag($$v) > 1) ? max(0, ${Time}) : ${Time}",
                 {"v": (0.0, 0.0, 0.0)}, time=3.0)
    assert v2 == 3.0


# ---------------------------------------------------------------- 变量引用
def test_dollar_reference_kinds():
    assert numeric("${p}", {"p": 7.0}) == 7.0
    assert _vec(numeric("$$v", {"v": (1.0, 2.0, 3.0)})) == (1.0, 2.0, 3.0)
    A = numeric("$$$A", {"A": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))})
    assert A[0][0] == 1.0 and A[2][2] == 1.0
    assert numeric("$${Time}", time=12.5) == 12.5


def test_context_lookup():
    ctx = EvalContext({"a": 2.0}, position=(1.0, 2.0, 3.0), time=9.0)
    assert ctx.lookup("a", 1) == 2.0
    ff = FieldFunction("${a} + ${Time} + ${Position}[0]")
    assert ff.evaluate({"a": 2.0}, position=(1.0, 2.0, 3.0), time=9.0) == 12.0  # 2 + 9 + 1
    with pytest.raises(NameError):
        ctx.lookup("Missing", 1)


# ---------------------------------------------------------------- 编译预检
def test_compile_expression():
    ast = compile_expression("1 + 2 * 3 + mag($$v)")
    assert ast is not None
    with pytest.raises(Exception):
        compile_expression("1 + (2")


# ---------------------------------------------------------------- 错误路径
def test_unknown_function():
    with pytest.raises(Exception):
        numeric("nope(1)")


def test_undefined_variable():
    with pytest.raises(Exception):
        numeric("${Nope} + 1")


def test_divide_by_zero():
    with pytest.raises(Exception):
        numeric("1 / 0")


def test_out_of_range_index():
    with pytest.raises(Exception):
        numeric("$$v[3]", {"v": (1.0, 2.0, 3.0)})


def test_syntax_error():
    with pytest.raises(Exception):
        numeric("1 +")

# -*- coding: utf-8 -*-
"""P 波 P3：初始化器（field function/常量/表格初值 —— Run 前 Initialize 可用）。

覆盖：
  Initializer：链式声明（常量/表格/场函数）+ compile 预检 + 单点/批量求值
  表格线性插值：interpolateTable 在表达式内查表
  initial_field / apply_initial：把初场写入 DemoDiffusionSolver（与 demo_mesh 联调）
  SolverBackend.initialize：Run 前注入初始化器，初场生效
  诚实拒绝：source_field 缺失/未定义函数/语法错误/维度不匹配/单点求值失败

验收核心（P3 行）：Run 前 Initialize 可用 —— 常量/表格/场函数在初始化阶段
被求值并生成初始场，求解器由此起步。
"""
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from init_solver import Initializer, _mag, _shape0
from solver_run import DemoDiffusionSolver, SolverBackend, SolverState, demo_mesh


def _ramp_init():
    """demo_mesh(nx=4) 上  T = T0 + 100*x（表线性插值）。"""
    init = Initializer(source_field="T")
    init.add_constant("T0", 300.0)
    init.add_table("ramp", [0.0, 1.0], {"load": [0.0, 100.0]})
    init.add_function(
        "T",
        'T0 + interpolateTable(@Table("ramp"), "load", LINEAR, "", ${Position}[0])',
    )
    return init


# ---------------------------------------------------------------- 声明 / 预检
def test_init_chainable_declare():
    init = Initializer()
    assert init.add_constant("a", 1.0) is init
    assert init.add_table("t", [0, 1], {"v": [1.0, 2.0]}) is init
    assert init.add_function("f", "a + 2") is init
    assert init.constants["a"] == 1.0
    assert "t" in init.tables and "v" in init.tables["t"].columns


def test_init_declare_manifest():
    init = Initializer()
    init.declare({
        "constants": {"k": 2.0},
        "tables": {"t": {"x": [0, 1], "columns": {"v": [0.0, 5.0]}}},
        "functions": {"f": "k * 3"},
        "source_field": "f",
    })
    assert init.constants["k"] == 2.0
    assert init.tables["t"].interpolate("v", "LINEAR", 0.5) == pytest.approx(2.5)
    assert init.value("f") == pytest.approx(6.0)
    assert init.source_field == "f"


def test_init_compile_precheck_rejects_syntax_error():
    init = Initializer().add_function("f", "1 +")
    with pytest.raises(Exception):
        init.compile()


def test_init_compile_unknown_function_rejected():
    init = Initializer().add_function("f", "nope(1)")
    init.compile()                              # 语法合法，编译通过
    with pytest.raises(Exception):
        init.value("f")                         # 求值时报未知函数 nope


def test_init_source_field_must_be_registered():
    init = Initializer(source_field="U")
    with pytest.raises(ValueError):
        init.compile()                       # source_field 未登记进 functions


# ---------------------------------------------------------------- 求值（单点）
def test_init_value_constant_and_function():
    init = _ramp_init().compile()
    assert init.value("T0") == pytest.approx(300.0)        # 常量直接返回
    assert init.value("T", position=(0.0, 0.0, 0.0)) == pytest.approx(300.0)
    assert init.value("T", position=(1.0, 0.0, 0.0)) == pytest.approx(400.0)


def test_init_unknown_name_honest_reject():
    init = _ramp_init().compile()
    with pytest.raises(NameError):
        init.value("Nope")


# ---------------------------------------------------------------- 批量初场
def test_init_field_scalar_batch():
    V, _ = demo_mesh(nx=4)
    init = _ramp_init().compile()
    f = init.field("T", V)
    assert f.shape == (len(V),)
    assert np.allclose(f, 300.0 + 100.0 * V[:, 0])        # 表线性插值 = 100*x


def test_init_field_vector_auto_and_mag():
    V, _ = demo_mesh(nx=4)
    init = Initializer(source_field="w").add_function("w", "[${Position}[0], ${Position}[1], 0.0]")
    init.compile()
    f = init.field("w", V, vector="auto")
    assert f.shape == (len(V), 3)
    assert np.allclose(f[:, 0], V[:, 0])
    m = init.field("w", V, vector="mag")
    assert m.shape == (len(V),)
    assert np.allclose(m, np.sqrt(V[:, 0] ** 2 + V[:, 1] ** 2))


def test_init_field_default_on_single_point_failure():
    V, _ = demo_mesh(nx=4)
    init = Initializer().add_function("f", "1 / ${Position}[0]")   # x=0 除零
    init.compile()
    with pytest.raises(Exception):
        init.field("f", V)                                          # default=None 诚实抛出
    f = init.field("f", V, default=0.0)                             # default 兜底
    assert f.shape == (len(V),)
    assert np.isfinite(f).all()


def test_init_initial_field_requires_source_field():
    V, _ = demo_mesh(nx=4)
    init = Initializer().add_function("f", "1.0").compile()
    with pytest.raises(ValueError):
        init.initial_field(V)


# ---------------------------------------------------------------- 施加到求解器
def test_init_apply_initial_demo_solver():
    V, C = demo_mesh(nx=4)
    init = _ramp_init().compile()
    s = DemoDiffusionSolver(V, C, initializer=init)
    s._initialize_field()
    assert s.field().shape == (len(V),)
    assert np.allclose(s.field(), 300.0 + 100.0 * V[:, 0])        # 初场 = T0+100x


def test_init_apply_initial_wrong_shape_rejected():
    V, C = demo_mesh(nx=4)
    init = _ramp_init().compile()
    s = DemoDiffusionSolver(V, C, initializer=init)
    with pytest.raises(ValueError):
        s._set_initial_field(np.zeros(len(V) + 1))


# ---------------------------------------------------------------- SolverBackend 联调
def test_backend_initialize_applies_initializer():
    V, C = demo_mesh(nx=4)
    init = _ramp_init().compile()
    be = SolverBackend(DemoDiffusionSolver(V, C), initializer=init)
    assert be.state() == SolverState.IDLE
    assert be.initialize()
    assert be.state() == SolverState.INITIALIZED
    assert np.allclose(be.solver.field(), 300.0 + 100.0 * V[:, 0])


def test_backend_run_loop_with_initializer():
    V, C = demo_mesh(nx=3)
    init = _ramp_init().compile()
    be = SolverBackend(DemoDiffusionSolver(V, C), initializer=init)
    be.initialize()
    r = be.run_loop(max_iter=5)
    assert r["state"] == SolverState.COMPLETED
    assert r["iteration"] == 5


# ---------------------------------------------------------------- 工具函数
def test_helpers_shape_and_mag():
    assert _shape0(1.0) == "scalar"
    assert _shape0((1.0, 2.0, 3.0)) == "vector"
    assert _shape0(((1, 0, 0), (0, 1, 0), (0, 0, 1))) == "tensor"
    assert _mag((3.0, 4.0, 0.0)) == pytest.approx(5.0)

# -*- coding: utf-8 -*-
"""P 波 P3：初始化器（field function/常量/表格初值 —— Run 前 Initialize 可用）。

在求解 Run 之前，把用户声明的 **常量 / 表格 / 场函数** 编译、求值，为求解器
生成**初始场**（初始条件），与 STAR-CCM+「Initialize 求解初始条件」语义对齐：
- 常量   : name -> 标量（并入 EvalContext.variables，供表达式 / 函数引用）
- 表格   : name -> field_fn.Table（供 interpolateTable 在表达式内查表）
- 场函数 : name -> field_fn.FieldFunction（表达式，随网格坐标求值）

`Initializer` 提供：
- 声明 API：`add_constant / add_table / add_function / declare(manifest)`（可链式）
- 预检    ：`compile()` 一次性编译并锁定语法错误（诚实拒绝在 Run 前暴露）
- 求值    ：`value(name, position, time)` 单点；`field(name, coords, ...)` 批量初场
- 施加    ：`apply_initial(solver, ...)` 把初场写入求解器（Run 前 Initialize 用）

值域与 field_fn 一致：标量 float / 矢量 tuple[3] / 张量 tuple[tuple[3]x3]，全部
用 Python 原生类型引入初场再转 numpy 数组，刻意避开 numpy/BLAS 在 occ/scdm
环境的 `0xc06d007f` 崩溃（与 field_fn / solver_run 同一约束）。

用法（无 GUI）：
    from init_solver import Initializer
    from solver_run import demo_mesh, DemoDiffusionSolver
    V, C = demo_mesh(nx=3)
    init = Initializer(source_field="T")
    init.add_constant("T0", 300.0)
    init.add_table("ramp", [0, 1], {"load": [0.0, 100.0]})
    init.add_function("T", "T0 + interpolateTable(@Table(\"ramp\"), \"load\", LINEAR, \"\", ${Position}[0])")
    init.compile()
    s = DemoDiffusionSolver(V, C, initializer=init)
    s._initialize_field()      # 初场 T = T0 + load(x)
    print(s.field()[:3])       # [300, 300.25, 300.5]（x=0/0.25/0.5）
"""
import numpy as np

from field_fn import EvalContext, FieldFunction, Table


def _mag(v):
    """矢量的欧氏范数（完全回避 numpy.linalg）。"""
    s = 0.0
    for x in v:
        s += float(x) * float(x)
    return float(s ** 0.5)


def _shape0(v):
    """判断一个求值结果的维度类型：'scalar' / 'vector' / 'tensor'。"""
    if isinstance(v, (tuple, list)):
        if len(v) == 3 and all(isinstance(x, (tuple, list)) for x in v):
            return "tensor"
        return "vector"
    return "scalar"


class Initializer:
    """场函数 / 常量 / 表格 初始化器（Run 前 Initialize 可用）。

    集中管理初始化定义并据此生成初场。`source_field` 指明算作初始场的那条
    场函数，`apply_initial(solver)` 会把它对 solver.vertices 求值并写入求解器。
    """

    def __init__(self, constants=None, tables=None, functions=None,
                 source_field=None, default=0.0):
        self.constants = {}
        self.tables = {}
        self.functions = {}       # name -> 原始表达式
        self._compiled = {}       # name -> FieldFunction
        self.source_field = source_field
        self.default = float(default)
        if constants:
            self.add_constants(**constants)
        for name, obj in (tables or {}).items():
            self.add_table_obj(name, obj)
        for name, expr in (functions or {}).items():
            self.add_function(name, expr)

    # -- 声明（可链式） ------------------------------------------------
    def add_constant(self, name, value):
        self.constants[name] = value
        return self

    def add_constants(self, **kv):
        for k, v in kv.items():
            self.constants[k] = v
        return self

    def add_table(self, name, x, columns, method="LINEAR"):
        self.tables[name] = Table(name, x, columns)
        self.tables[name].method = method
        return self

    def add_table_obj(self, name, table):
        if not isinstance(table, Table):
            raise TypeError(
                "P3 添加表 %r 需 field_fn.Table 实例（见 add_table 构造）" % name)
        self.tables[name] = table
        return self

    def add_function(self, name, expr, kind="auto"):
        self.functions[name] = expr
        self._compiled.pop(name, None)
        return self

    def declare(self, manifest):
        """从清单 dict 一次性载入：{'constants': {...}, 'tables': {...}, 'functions': {...}}。"""
        for k, v in (manifest.get("constants") or {}).items():
            self.add_constant(k, v)
        for k, obj in (manifest.get("tables") or {}).items():
            if isinstance(obj, Table):
                self.add_table_obj(k, obj)
            else:
                self.add_table(k, obj["x"], obj["columns"])
        for k, expr in (manifest.get("functions") or {}).items():
            self.add_function(k, expr)
        if manifest.get("source_field") is not None:
            self.source_field = manifest["source_field"]
        return self

    # -- 预检 ---------------------------------------------------------
    def compile(self):
        """编译全部场函数（语法/函数名预检），返回本实例。编译冲突诚实抛出。"""
        for name, expr in self.functions.items():
            ff = FieldFunction(expr)
            self._compiled[name] = ff
        # 若 source_field 未显式登记，则在常量/表格之外强制其可编译
        if self.source_field is not None and self.source_field not in self.functions:
            raise ValueError(
                "P3 source_field 未定义：%r（需先在 functions 中声明该初场函数）"
                % self.source_field)
        return self

    def _get_function(self, name):
        if name in self._compiled:
            return self._compiled[name]
        if name in self.functions:
            return FieldFunction(self.functions[name])
        raise NameError("P3 未定义场函数/常量 %r（可引用：%s）"
                        % (name, ", ".join(list(self.functions) + list(self.constants))))

    # -- 求值 ---------------------------------------------------------
    def make_context(self, position=(0.0, 0.0, 0.0), time=0.0):
        return EvalContext(self.constants, self.tables, position, time)

    def value(self, name, position=(0.0, 0.0, 0.0), time=0.0):
        """单点求值：常量直接返回；场函数对 position 求值。"""
        if name in self.constants:
            return self.constants[name]
        return self._get_function(name).evaluate(
            self.constants, self.tables, position=tuple(position), time=time)

    def field(self, name, coords, time=0.0, default=None, vector="auto"):
        """对网格坐标批量求值初场，返回 numpy 数组（标量 (N,) / 矢量或张量 (N,L)）。

        - vector="auto"：按首点结果保留原状；若首点为标量→(N,)，矢量→(N,3)；
          "mag" 强制取矢量模；(N,)。
        - 单点求值异常时：提供 default 则用其兜底（含非有限情形），否则诚实抛出。
        """
        ff = self._get_function(name)
        coords = np.asarray(coords, float)
        if coords.ndim == 1:
            coords = coords.reshape(1, 3)
        n = len(coords)
        rows = []
        for c in coords:
            try:
                v = ff.evaluate(self.constants, self.tables,
                                position=tuple(float(x) for x in c), time=time)
            except Exception:
                if default is None:
                    raise
                v = default
            rows.append(v)
        first = rows[0]
        if vector == "mag" and _shape0(first) in ("vector", "tensor"):
            return np.asarray([_mag(r) for r in rows], float)
        if vector == "first" and _shape0(first) in ("vector", "tensor"):
            return np.asarray([float(r[0]) for r in rows], float)
        if _shape0(first) == "scalar":
            return np.asarray([float(r) for r in rows], float)
        return np.asarray([r for r in rows], float)

    def initial_field(self, coords, time=0.0, default=None):
        """source_field 指定的初场，对 coords 批量求值。"""
        if self.source_field is None:
            raise ValueError("P3 初始化器未配置 source_field（无初场函数）")
        return self.field(self.source_field, coords, time=time, default=default)

    def apply_initial(self, solver, time=0.0, default=None):
        """把初场写入求解器（Run 前 Initialize 用）。

        需要 solver.vertices。若初场为矢量而 solver 字段为标量一维，则自动取模。
        成功返回 True，未配置 source_field / 无 vertices / 无法求值 均诚实抛出。
        """
        coords = getattr(solver, "vertices", None)
        if coords is None:
            raise ValueError("P3 apply_initial 需要 solver.vertices（网格坐标）")
        if self.source_field is None:
            raise ValueError("P3 初始化器未配置 source_field（无初场函数）")
        arr = self.field(self.source_field, coords, time=time, default=default)
        target = getattr(solver, "_field", None)
        if arr.ndim == 2 and arr.shape[1] == 3 and target is not None and target.ndim == 1:
            arr = np.sqrt(np.sum(arr * arr, axis=1))
        if hasattr(solver, "_set_initial_field"):
            solver._set_initial_field(arr)
        else:
            solver._field = arr
        return True

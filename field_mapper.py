# -*- coding: utf-8 -*-
"""R 波 R2：跨网格数据映射与插值器（纯 numpy + 标准库，无 Qt）。

对齐官方 `star.cosimulation.common`（`FieldMapper` / `MapperManager` /
`FieldTreatment`，见 `doc_javadoc_catalog.md` §15）与 `star.mapping` 语义层
（`semantic_dict` 中 `star.mapping -> mapping / 数据映射`），提供跨网格场数据
映射与一维表格插值：

  跨网格映射 —— 把源网格（P4 `FVM` 四面体）上的**单元中心场**或**顶点场**映射到
    目标网格单元中心或任意目标点上。先对每个目标点做重心定位得到插值权重
    （`build_weights`），再以权重作用于源场（`apply_weights`）；权重只依赖几何，
    可一次构建、复用于多个场（`FieldMapper` 缓存）。
  域外处理 —— `FieldTreatment` 提供 nan / constant / zero / nearest 四种策略，
    诚实区分「插值所得」与「回退填充」（回退均单独计入 `n_outside`）。
  表格插值 —— `TableInterpolator` 复用 `field_fn.Table` 的 LINEAR / SPLINE 内核
    （与 P2 表达式 `interpolateTable` 同一实现），提供逐点与批量求值。
  映射器管理 —— `MapperManager` 按名注册 / 检索 / 应用，对应官方 MapperManager。

**诚实边界**：源网格为四面体（`FVM`），故跨网格插值为**一阶重心线性**；域外点的
结果取决于 `FieldTreatment`，默认保留 NaN 不伪造数值；本模块不提供官方 .sim 中
`FieldMapper` 对象的落盘读写（语料中亦无该对象，见 `doc_javadoc_catalog.md` §三），
只提供与官方列同一命名空间的计算内核。

纯 numpy 约束：不新增 scipy 依赖；范数统一走 `solver_run._safe_norm`（规避 occ
环境 `0xc06d007f`）。
"""
import numpy as np

from field_fn import Table, _interp_linear, _interp_spline
from solver_run import _safe_norm

EPS = 1.0e-12
LOC_TOL = 1.0e-7

TREAT_NAN = "nan"
TREAT_CONSTANT = "constant"
TREAT_ZERO = "zero"
TREAT_NEAREST = "nearest"
TREATMENTS = (TREAT_NAN, TREAT_CONSTANT, TREAT_ZERO, TREAT_NEAREST)

INTERP_LINEAR = "LINEAR"
INTERP_SPLINE = "SPLINE"


# ===========================================================================
# 几何核心：四面体重心定位与插值权重
# ===========================================================================
def _sixvol(a, b, c, d):
    """带号体积的 6 倍 det(b-a, c-a, d-a)，支持广播（与 postprocess 同源）。"""
    e1 = b - a
    e2 = c - a
    e3 = d - a
    return (e1[..., 0] * (e2[..., 1] * e3[..., 2] - e2[..., 2] * e3[..., 1])
            - e1[..., 1] * (e2[..., 0] * e3[..., 2] - e2[..., 2] * e3[..., 0])
            + e1[..., 2] * (e2[..., 0] * e3[..., 1] - e2[..., 1] * e3[..., 0]))


def locate_points(fv, points, tol=LOC_TOL):
    """为每个目标点定位源四面体并给出重心权重。

    返回 `(cells, weights)`：`cells` 为 `(n,)` 源单元索引（域外 -1），`weights`
    为 `(n,4)` 重心插值权重（域外全 0）。多点命中（点恰落在共享面/棱/顶点）时
    取**最小单元索引**以保证确定性。
    """
    pts = np.atleast_2d(np.asarray(points, float))
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("locate_points 需要 (n,3) 目标点")
    V = fv.vertices
    C = fv.cells
    A = V[C[:, 0]]
    B = V[C[:, 1]]
    Cc = V[C[:, 2]]
    D = V[C[:, 3]]
    ref = _sixvol(A, B, Cc, D)
    safe = np.where(np.abs(ref) > 1e-14, ref, 1.0)
    n = len(pts)
    cells = np.full(n, -1, np.int64)
    W = np.zeros((n, 4), float)
    for i in range(n):
        p = pts[i]
        la = _sixvol(p, B, Cc, D) / safe
        lb = _sixvol(A, p, Cc, D) / safe
        lc = _sixvol(A, B, p, D) / safe
        ld = _sixvol(A, B, Cc, p) / safe
        ok = (la >= -tol) & (lb >= -tol) & (lc >= -tol) & (ld >= -tol)
        idx = np.where(ok)[0]
        if idx.size == 0:
            continue
        k = int(idx[0])
        w = np.array([la[k], lb[k], lc[k], ld[k]], float)
        s = w.sum()
        if abs(s) > EPS:
            w = w / s
        cells[i] = k
        W[i] = w
    return cells, W


def build_weights(fv, points, tol=LOC_TOL):
    """构建目标点 → 源网格的重心插值权重（一次构建、可复用）。

    返回 dict：`cells`/`weights`/`inside`/`n_inside`/`n_outside`/`n_points`。
    """
    cells, W = locate_points(fv, points, tol=tol)
    inside = cells >= 0
    return {"cells": cells, "weights": W, "inside": inside,
            "n_inside": int(inside.sum()),
            "n_outside": int((~inside).sum()),
            "n_points": int(len(cells))}


def nearest_cells(fv, points):
    """每个目标点最近的源**单元**索引（按单元质心欧氏距离）。

    域外回退（`FieldTreatment("nearest")`）与粗网格到细网格的兜底共用。
    返回 `(n,)` 索引数组；空点集返回空数组。
    """
    pts = np.atleast_2d(np.asarray(points, float))
    if len(pts) == 0:
        return np.zeros(0, np.int64)
    out = np.empty(len(pts), np.int64)
    cent = fv.centroids
    for i, p in enumerate(pts):
        out[i] = int(np.argmin(_safe_norm(cent - p, axis=1)))
    return out


def _as_vertex_field(fv, field):
    """源场统一为逐顶点长度：单元中心场按体积加权折算到顶点。

    与 `postprocess.sample_scalar` 口径一致（长度 == 单元数且 != 顶点数时折算）。
    """
    arr = np.asarray(field, float)
    if arr.ndim == 1:
        if arr.size == fv.n_cells and arr.size != fv.n_vertices:
            return _cell_to_vertex(fv, arr)
        if arr.size != fv.n_vertices:
            raise ValueError("源场长度 %d 与源网格（%d 单元 / %d 顶点）不匹配"
                             % (arr.size, fv.n_cells, fv.n_vertices))
        return arr
    if arr.shape[0] == fv.n_cells and arr.shape[0] != fv.n_vertices:
        return np.column_stack([_cell_to_vertex(fv, arr[:, d])
                                for d in range(arr.shape[1])])
    if arr.shape[0] != fv.n_vertices:
        raise ValueError("源矢量场长度 %d 与源网格（%d 单元 / %d 顶点）不匹配"
                         % (arr.shape[0], fv.n_cells, fv.n_vertices))
    return arr


def _cell_to_vertex(fv, field):
    """体积加权单元中心场 → 顶点场（标量或矢量，纯 numpy `np.add.at`）。"""
    arr = np.asarray(field, float)
    scalar = arr.ndim == 1
    a = arr.reshape(fv.n_cells, -1)
    ncomp = a.shape[1]
    wc = np.asarray(fv.volumes, float)[:, None] * np.ones((1, 4), float)
    vid = fv.cells.reshape(-1)
    vals = np.zeros((fv.n_vertices, ncomp), float)
    wts = np.zeros(fv.n_vertices, float)
    flat = (wc[:, :, None] * a[:, None, :]).reshape(-1, ncomp)
    np.add.at(vals, vid, flat)
    np.add.at(wts, vid, wc.reshape(-1))
    good = wts > EPS
    vals[good] /= wts[good][:, None]
    return vals[:, 0] if scalar else vals


def apply_weights(weights, source_field, fv, cells=None):
    """把已构建的权重作用于源场，返回逐目标点的插值结果。

    `source_field` 可为逐顶点场（长度 == 顶点数）或单元中心场（长度 == 单元数，
    自动体积加权折算到顶点）；`cells` 缺省取 `weights["cells"]`，可显式传入最近
    单元索引（nearest 回退时复用同一权重表）。
    域外点返回 NaN（回退策略由 `FieldMapper` 归一化）。
    """
    if cells is None:
        cells = weights["cells"]
    W = weights["weights"]
    vf = _as_vertex_field(fv, source_field)
    n = len(cells)
    trailing = vf.shape[1:]
    out = np.full((n,) + trailing, np.nan, float)
    for i in range(n):
        k = int(cells[i])
        if k < 0:
            continue
        out[i] = W[i] @ vf[fv.cells[k]]
    return out


# ===========================================================================
# 域外处理策略（官方 FieldTreatment）
# ===========================================================================
class FieldTreatment:
    """域外值处理策略（对应官方 `FieldTreatment`）。

    kind：
      - `"nan"`      : 域外保留 NaN（默认，诚实不伪造）；
      - `"constant"` : 域外填 `value` 常量；
      - `"zero"`     : 域外填 0；
      - `"nearest"`  : 域外取最近源单元场值（`nearest_cells`）。
    """

    def __init__(self, kind=TREAT_NAN, value=0.0):
        kind = str(kind).strip().lower()
        if kind not in TREATMENTS:
            raise ValueError("未知 FieldTreatment %r（用 %s）"
                             % (kind, " / ".join(TREATMENTS)))
        self.kind = kind
        self.value = float(value)

    def describe(self):
        if self.kind == TREAT_CONSTANT:
            return "constant(%g)" % self.value
        return self.kind

    def __repr__(self):
        return "FieldTreatment(%s)" % self.describe()


def _nearest_values(fv, points, source_field, outside_mask):
    """域外点取最近源值的向量（仅对 outside 点计算）。"""
    n = len(outside_mask)
    if not np.any(outside_mask):
        return None
    vf = _as_vertex_field(fv, source_field)
    pts = np.atleast_2d(np.asarray(points, float))
    cent = fv.centroids
    out = np.zeros((n,) + vf.shape[1:], float)
    for i in np.where(outside_mask)[0]:
        k = int(np.argmin(_safe_norm(cent - pts[i], axis=1)))
        out[i] = vf[fv.cells[k]].mean(axis=0)
    return out


# ===========================================================================
# 跨网格映射器（官方 FieldMapper）
# ===========================================================================
class FieldMapper:
    """跨网格场数据映射器（对应官方 `FieldMapper`）。

    `source_fv` 为源网格（P4 `FVM`）；`treatment` 决定域外点结果；`target_points`
    给定时预构建权重（后续 `map_scalar`/`map_vector` 不传 points 即复用该权重）。

    返回的 `map_*` 载荷为 dict：`values`/`inside`/`cells`/`weights`/`n_inside`/
    `n_outside`/`n_points`/`treatment`/`field`。
    """

    def __init__(self, source_fv, treatment=None, target_points=None,
                 tol=LOC_TOL):
        self.source_fv = source_fv
        self.treatment = treatment if treatment is not None else FieldTreatment()
        if not isinstance(self.treatment, FieldTreatment):
            self.treatment = FieldTreatment(self.treatment)
        self.tol = tol
        self._weights = None
        self._points = None
        if target_points is not None:
            self.set_target(target_points)

    # -- 权重缓存 --------------------------------------------------------
    def set_target(self, points):
        self._points = np.atleast_2d(np.asarray(points, float))
        self._weights = build_weights(self.source_fv, points, tol=self.tol)
        return self._weights

    @property
    def weights(self):
        return self._weights

    def clear_target(self):
        self._weights = None
        self._points = None

    def _resolve(self, points):
        if points is not None:
            return build_weights(self.source_fv, points, tol=self.tol), \
                np.atleast_2d(np.asarray(points, float))
        if self._weights is None:
            raise ValueError("未设置目标点：构造时给 target_points 或调用 "
                             "set_target()，或把 points 传给 map_*")
        return self._weights, self._points

    def _treatment_fill(self, values, inside, fv, points, source_field):
        kind = self.treatment.kind
        if kind == TREAT_NAN:
            return values
        if kind == TREAT_CONSTANT:
            values[~inside] = self.treatment.value
            return values
        if kind == TREAT_ZERO:
            values[~inside] = 0.0
            return values
        near = _nearest_values(fv, points, source_field, ~inside)
        if near is not None:
            values[~inside] = near[~inside]
        return values

    # -- 映射 ------------------------------------------------------------
    def map_scalar(self, field, points=None):
        """标量场映射到目标点（或预置目标点）。"""
        w, pts = self._resolve(points)
        values = apply_weights(w, field, self.source_fv)
        values = self._treatment_fill(values, w["inside"], self.source_fv, pts,
                                      field)
        return self._payload(field, values, w)

    def map_vector(self, vec, points=None):
        """矢量场 (M,3) 映射到目标点（或预置目标点）。"""
        w, pts = self._resolve(points)
        values = apply_weights(w, vec, self.source_fv)
        values = self._treatment_fill(values, w["inside"], self.source_fv, pts,
                                      vec)
        return self._payload(vec, values, w)

    def map_to_cells(self, field, target_fv):
        """映射到目标网格**单元中心**（目标点 = `target_fv.centroids`）。"""
        return self.map_scalar(field, points=target_fv.centroids)

    def map_to_vertices(self, field, target_fv):
        """映射到目标网格**顶点**（目标点 = `target_fv.vertices`）。"""
        return self.map_scalar(field, points=target_fv.vertices)

    def map_vector_to_cells(self, vec, target_fv):
        """矢量场映射到目标网格单元中心。"""
        return self.map_vector(vec, points=target_fv.centroids)

    def _payload(self, field, values, w):
        name = getattr(field, "name", None)
        payload = dict(w)
        payload["values"] = values
        payload["treatment"] = self.treatment.describe()
        payload["field"] = (name if name is not None
                            else ("vector" if np.asarray(field).ndim > 1
                                  else "scalar"))
        return payload

    def summary(self):
        return {"source_cells": self.source_fv.n_cells,
                "source_vertices": self.source_fv.n_vertices,
                "treatment": self.treatment.describe(),
                "n_points": (0 if self._weights is None
                             else self._weights["n_points"]),
                "n_inside": (0 if self._weights is None
                             else self._weights["n_inside"]),
                "n_outside": (0 if self._weights is None
                              else self._weights["n_outside"])}


# ===========================================================================
# 映射器注册表（官方 MapperManager）
# ===========================================================================
class MapperManager:
    """命名映射器注册表（对应官方 `MapperManager`）。"""

    def __init__(self):
        self._mappers = {}

    def register(self, name, mapper):
        if not isinstance(mapper, FieldMapper):
            raise TypeError("MapperManager 只接受 FieldMapper 实例")
        self._mappers[str(name)] = mapper
        return mapper

    def unregister(self, name):
        if str(name) not in self._mappers:
            raise KeyError("未注册映射器 %r" % name)
        return self._mappers.pop(str(name))

    def get(self, name):
        try:
            return self._mappers[str(name)]
        except KeyError:
            raise KeyError("未注册映射器 %r（已注册：%s）"
                           % (name, ", ".join(sorted(self._mappers)) or "无"))

    def has(self, name):
        return str(name) in self._mappers

    def names(self):
        return sorted(self._mappers)

    def map_scalar(self, name, field, points=None):
        return self.get(name).map_scalar(field, points=points)

    def map_vector(self, name, vec, points=None):
        return self.get(name).map_vector(vec, points=points)

    def summary(self):
        return {n: m.summary() for n, m in self._mappers.items()}

    def __len__(self):
        return len(self._mappers)

    def __contains__(self, name):
        return self.has(name)

    def __iter__(self):
        return iter(self.names())


# ===========================================================================
# 一维表格插值器（复用 field_fn.Table 内核）
# ===========================================================================
class TableInterpolator:
    """一维表格插值器（对应官方 Table / InternalTable / FileTable 的插值用法）。

    `table` 为 `field_fn.Table`；`column` 为待插值列名；`method` 为
    `"LINEAR"` / `"SPLINE"`（与 P2 表达式 `interpolateTable` 同一内核，
    越界钳位到端点）。
    """

    def __init__(self, table, column, method=INTERP_LINEAR):
        if not isinstance(table, Table):
            raise TypeError("TableInterpolator 需要 field_fn.Table 实例")
        method = str(method).upper()
        if method not in (INTERP_LINEAR, INTERP_SPLINE):
            raise ValueError("未知插值法 %r（用 LINEAR / SPLINE）" % method)
        if column not in table.columns:
            raise KeyError("表 %r 无列 %r（可用：%s）"
                           % (table.name, column, ", ".join(table.columns)))
        xs = [float(v) for v in table.x]
        ys = [float(v) for v in table.columns[column]]
        if len(xs) != len(ys):
            raise ValueError("表 %r 列 %r 长度不一致" % (table.name, column))
        if not xs:
            raise ValueError("表 %r 为空" % table.name)
        self.table = table
        self.column = column
        self.method = method
        self.xs = xs
        self.ys = ys

    @classmethod
    def from_arrays(cls, xs, ys, method=INTERP_LINEAR, name="inline"):
        """由坐标数组直接构造（无 field_fn.Table 亦可）。"""
        return cls(Table(name, list(xs), {"y": list(ys)}), "y", method=method)

    def value(self, x):
        """单点求值。"""
        if self.method == INTERP_LINEAR:
            return _interp_linear(self.xs, self.ys, float(x))
        return _interp_spline(self.xs, self.ys, float(x))

    def values(self, xs):
        """批量求值，返回 `(n,)` 数组。"""
        arr = np.asarray(xs, float).reshape(-1)
        return np.array([self.value(v) for v in arr], float)

    def range(self):
        return (self.xs[0], self.xs[-1])

    def summary(self):
        return {"table": self.table.name, "column": self.column,
                "method": self.method, "n_points": len(self.xs),
                "range": self.range()}


# ===========================================================================
# 便捷函数
# ===========================================================================
def map_scalar(source_fv, field, points, treatment=None, tol=LOC_TOL):
    """一次性跨网格标量映射（无权重缓存）。"""
    return FieldMapper(source_fv, treatment=treatment, tol=tol).map_scalar(
        field, points=points)


def map_vector(source_fv, vec, points, treatment=None, tol=LOC_TOL):
    """一次性跨网格矢量映射（无权重缓存）。"""
    return FieldMapper(source_fv, treatment=treatment, tol=tol).map_vector(
        vec, points=points)

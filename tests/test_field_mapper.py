# -*- coding: utf-8 -*-
"""R 波 R2：跨网格数据映射与插值器 —— 重心映射 + 域外处理 + 表格插值 + 映射器管理。

覆盖：
  常量：EPS/LOC_TOL/TREAT_*/INTERP_*
  定位：locate_points（域内/域外、权重和为 1、最小单元索引确定性）
  权重：build_weights 计数、nearest_cells 最近单元
  映射：FieldMapper 跨网格标量/矢量映射（线性场精确复现）、单元中心源场折算、
        map_to_cells/map_to_vertices、预置目标点复用/clear_target 诚实拒绝
  域外：FieldTreatment nan/constant/zero/nearest 四策略 + 未知策略 ValueError
  管理：MapperManager 注册/检索/注销/names/len/contains/iter + 类型与缺失 KeyError
  表格：TableInterpolator LINEAR/SPLINE/from_arrays/range/summary + 错误路径
  工具：map_scalar/map_vector 便捷函数、源场长度不匹配诚实拒绝

验收核心（R2 行）：跨网格数据映射/插值器 L0 → 目标 R2。
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

from field_fn import Table
from field_mapper import (
    EPS, INTERP_LINEAR, INTERP_SPLINE, LOC_TOL, TREAT_CONSTANT, TREAT_NAN,
    TREAT_NEAREST, TREAT_ZERO, TREATMENTS, FieldMapper, FieldTreatment,
    MapperManager, TableInterpolator, apply_weights, build_weights,
    locate_points, map_scalar, map_vector, nearest_cells,
)
from fvm_core import FVM, cube_tet_mesh


# ---------------------------------------------------------------- 工具
def _fv(nx=2):
    V, C = cube_tet_mesh(nx)
    return FVM(V, C)


def _lin(points):
    """解析线性标量 f = 2x + 3y - z + 1（可在四面体内被一阶插值精确复现）。"""
    p = np.atleast_2d(np.asarray(points, float))
    return 2.0 * p[:, 0] + 3.0 * p[:, 1] - p[:, 2] + 1.0


# ---------------------------------------------------------------- 常量
def test_field_mapper_constants():
    assert EPS > 0.0 and LOC_TOL > 0.0
    assert TREAT_NAN == "nan" and TREAT_CONSTANT == "constant"
    assert TREAT_ZERO == "zero" and TREAT_NEAREST == "nearest"
    assert set(TREATMENTS) == {"nan", "constant", "zero", "nearest"}
    assert INTERP_LINEAR == "LINEAR" and INTERP_SPLINE == "SPLINE"


# ---------------------------------------------------------------- 定位与权重
def test_locate_points_inside_and_outside():
    fv = _fv(2)
    inside = np.array([[0.25, 0.25, 0.25], [0.75, 0.5, 0.5]])
    cells, W = locate_points(fv, inside)
    assert (cells >= 0).all()
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-12)
    assert (W >= -LOC_TOL).all()
    outside = np.array([[2.0, 0.5, 0.5], [-1.0, 0.0, 0.0]])
    cells2, W2 = locate_points(fv, outside)
    assert (cells2 == -1).all()
    assert np.allclose(W2, 0.0)


def test_locate_points_rejects_bad_shape():
    fv = _fv(2)
    with pytest.raises(ValueError):
        locate_points(fv, np.zeros((3, 2)))


def test_locate_points_deterministic_min_cell():
    """点恰落在共享面/顶点上时取最小单元索引（确定性）。"""
    fv = _fv(2)
    p = np.array([[0.5, 0.5, 0.5]])
    c1, _ = locate_points(fv, p)
    c2, _ = locate_points(fv, p)
    assert c1[0] == c2[0] and c1[0] >= 0


def test_build_weights_counts():
    fv = _fv(2)
    pts = fv.vertices.copy()
    extra = np.array([[3.0, 3.0, 3.0]])
    w = build_weights(fv, np.vstack([pts, extra]))
    assert w["n_points"] == len(pts) + 1
    assert w["n_inside"] == len(pts)
    assert w["n_outside"] == 1
    assert w["inside"][:len(pts)].all() and not w["inside"][-1]


def test_nearest_cells_picks_closest_centroid():
    fv = _fv(2)
    target = fv.centroids[3] + 1e-4
    got = nearest_cells(fv, np.array([target]))
    assert got[0] == 3
    assert nearest_cells(fv, np.zeros((0, 3))).size == 0


# ---------------------------------------------------------------- 跨网格映射
def test_map_scalar_linear_field_exact_cross_mesh():
    """粗网格顶点线性场 → 细网格顶点：域内一阶插值精确复现。"""
    src, dst = _fv(2), _fv(3)
    mapper = FieldMapper(src)
    res = mapper.map_scalar(_lin(src.vertices), points=dst.vertices)
    assert res["n_outside"] == 0
    assert np.allclose(res["values"], _lin(dst.vertices), atol=1e-9)
    assert res["field"] == "scalar"
    assert res["treatment"] == "nan"


def test_map_scalar_constant_field_exact():
    src, dst = _fv(2), _fv(3)
    const = np.full(src.n_cells, 7.5)
    res = FieldMapper(src).map_scalar(const, points=dst.centroids)
    assert np.allclose(res["values"], 7.5)


def test_map_scalar_cell_field_bounded():
    src, dst = _fv(2), _fv(3)
    field = _lin(src.centroids)
    res = FieldMapper(src).map_scalar(field, points=dst.vertices)
    finite = np.isfinite(res["values"])
    assert finite.all()
    assert res["values"].min() >= field.min() - 1e-9
    assert res["values"].max() <= field.max() + 1e-9


def test_map_vector_linear_exact():
    src, dst = _fv(2), _fv(3)
    vec = np.column_stack([_lin(src.vertices),
                           0.5 * _lin(src.vertices),
                           -_lin(src.vertices)])
    res = FieldMapper(src).map_vector(vec, points=dst.vertices)
    assert res["values"].shape == (len(dst.vertices), 3)
    expect = np.column_stack([_lin(dst.vertices),
                              0.5 * _lin(dst.vertices),
                              -_lin(dst.vertices)])
    assert np.allclose(res["values"], expect, atol=1e-9)
    assert res["field"] == "vector"


def test_map_to_cells_and_vertices():
    src, dst = _fv(2), _fv(3)
    mapper = FieldMapper(src)
    rc = mapper.map_to_cells(_lin(src.vertices), dst)
    rv = mapper.map_to_vertices(_lin(src.vertices), dst)
    assert rc["values"].shape == (dst.n_cells,)
    assert rv["values"].shape == (dst.n_vertices,)
    assert np.allclose(rc["values"], _lin(dst.centroids), atol=1e-9)
    assert np.allclose(rv["values"], _lin(dst.vertices), atol=1e-9)


def test_preset_target_reuse_and_clear():
    src, dst = _fv(2), _fv(3)
    mapper = FieldMapper(src, target_points=dst.vertices)
    a = mapper.map_scalar(_lin(src.vertices))
    b = mapper.map_scalar(np.ones(src.n_vertices))
    assert np.allclose(a["values"], _lin(dst.vertices), atol=1e-9)
    assert np.allclose(b["values"], 1.0)
    assert mapper.summary()["n_points"] == dst.n_vertices
    mapper.clear_target()
    with pytest.raises(ValueError):
        mapper.map_scalar(_lin(src.vertices))


def test_map_without_target_honest_reject():
    src = _fv(2)
    with pytest.raises(ValueError):
        FieldMapper(src).map_scalar(_lin(src.vertices))


def test_field_length_mismatch_rejected():
    src, dst = _fv(2), _fv(3)
    mapper = FieldMapper(src)
    with pytest.raises(ValueError):
        mapper.map_scalar(np.zeros(5), points=dst.vertices)


# ---------------------------------------------------------------- 域外处理
def test_treatment_nan_constant_zero():
    src = _fv(2)
    outside = np.array([[4.0, 4.0, 4.0]])
    field = _lin(src.vertices)
    assert np.isnan(map_scalar(src, field, outside)["values"][0])
    assert np.allclose(
        map_scalar(src, field, outside,
                   treatment=FieldTreatment("constant", -3.0))["values"], -3.0)
    assert np.allclose(map_scalar(src, field, outside,
                                  treatment="zero")["values"], 0.0)


def test_treatment_nearest_matches_cell_mean():
    src = _fv(2)
    outside = np.array([[4.0, 4.0, 4.0]])
    field = _lin(src.vertices)
    res = map_scalar(src, field, outside, treatment="nearest")
    k = int(nearest_cells(src, outside)[0])
    assert np.allclose(res["values"], field[src.cells[k]].mean())
    assert res["treatment"] == "nearest"


def test_treatment_invalid_rejected():
    with pytest.raises(ValueError):
        FieldTreatment("bogus")
    assert FieldTreatment("CONSTANT", 1.0).kind == "constant"
    assert "constant(1)" in repr(FieldTreatment("constant", 1.0))


def test_treatment_accepts_string_and_instance():
    src = _fv(2)
    m1 = FieldMapper(src, treatment="zero")
    assert isinstance(m1.treatment, FieldTreatment) and m1.treatment.kind == "zero"
    t = FieldTreatment("constant", 2.0)
    m2 = FieldMapper(src, treatment=t)
    assert m2.treatment is t


# ---------------------------------------------------------------- 权重复用
def test_apply_weights_reuse_across_fields():
    src, dst = _fv(2), _fv(3)
    w = build_weights(src, dst.vertices)
    f1 = apply_weights(w, _lin(src.vertices), src)
    f2 = apply_weights(w, np.ones(src.n_vertices), src)
    assert np.allclose(f1, _lin(dst.vertices), atol=1e-9)
    assert np.allclose(f2, 1.0)


def test_apply_weights_outside_is_nan():
    src = _fv(2)
    pts = np.array([[9.0, 9.0, 9.0]])
    w = build_weights(src, pts)
    out = apply_weights(w, _lin(src.vertices), src)
    assert np.isnan(out[0])


# ---------------------------------------------------------------- MapperManager
def test_manager_register_lookup_apply():
    src, dst = _fv(2), _fv(3)
    mgr = MapperManager()
    mgr.register("air2fine", FieldMapper(src, target_points=dst.vertices))
    assert mgr.names() == ["air2fine"] and len(mgr) == 1
    assert mgr.has("air2fine") and "air2fine" in mgr
    assert list(iter(mgr)) == ["air2fine"]
    res = mgr.map_scalar("air2fine", _lin(src.vertices))
    assert np.allclose(res["values"], _lin(dst.vertices), atol=1e-9)
    assert "air2fine" in mgr.summary()


def test_manager_unregister_and_errors():
    src = _fv(2)
    mgr = MapperManager()
    mgr.register("m", FieldMapper(src))
    mgr.unregister("m")
    assert len(mgr) == 0
    with pytest.raises(KeyError):
        mgr.get("m")
    with pytest.raises(KeyError):
        mgr.unregister("m")
    with pytest.raises(TypeError):
        mgr.register("bad", object())


# ---------------------------------------------------------------- 表格插值
def test_table_interpolator_linear():
    t = Table("ramp", [0.0, 1.0, 2.0], {"load": [0.0, 10.0, 20.0]})
    it = TableInterpolator(t, "load", method="LINEAR")
    assert it.value(0.5) == pytest.approx(5.0)
    assert it.value(-1.0) == pytest.approx(0.0)
    assert it.value(5.0) == pytest.approx(20.0)
    assert np.allclose(it.values([0.0, 1.5, 3.0]), [0.0, 15.0, 20.0])
    assert it.range() == (0.0, 2.0)
    s = it.summary()
    assert s["table"] == "ramp" and s["n_points"] == 3


def test_table_interpolator_spline_natural_cubic():
    it = TableInterpolator.from_arrays([0.0, 1.0, 2.0], [0.0, 1.0, 4.0],
                                       method="SPLINE")
    assert it.method == "SPLINE"
    assert it.value(0.5) == pytest.approx(0.3125)
    assert it.value(1.0) == pytest.approx(1.0)


def test_table_interpolator_errors():
    t = Table("t", [0.0, 1.0], {"a": [0.0, 1.0]})
    with pytest.raises(TypeError):
        TableInterpolator([0, 1], "a")
    with pytest.raises(KeyError):
        TableInterpolator(t, "missing")
    with pytest.raises(ValueError):
        TableInterpolator(t, "a", method="NEAREST")
    empty = Table("e", [], {"a": []})
    with pytest.raises(ValueError):
        TableInterpolator(empty, "a")


# ---------------------------------------------------------------- 便捷函数
def test_convenience_functions():
    src, dst = _fv(2), _fv(3)
    rs = map_scalar(src, _lin(src.vertices), dst.vertices)
    assert np.allclose(rs["values"], _lin(dst.vertices), atol=1e-9)
    vec = np.column_stack([_lin(src.vertices)] * 3)
    rv = map_vector(src, vec, dst.vertices)
    assert rv["values"].shape == (len(dst.vertices), 3)

# -*- coding: utf-8 -*-
"""C 波 C2：草图/拉伸/旋转/扫掠/放样/管道（OCC 构造算子）验收。

依赖 conda 环境 occ（pythonocc-core），默认 Python 3.14 无 OCP → 整体 skip
（occ_builder.has_occ()==False），不影响 run_all 基线。真跑：
    <conda>/envs/occ/python.exe -m pytest tests/test_occ_builder.py -q
"""
import os
import tempfile

import pytest

from occ_builder import (Sketch, wire3d, extrude, revolve, loft, pipe, sweep,
                         _HAS_OCC)
from occ_bridge import (has_occ, shape_counts, tessellate, export_shape,
                        import_shape)

pytestmark = pytest.mark.skipif(
    not has_occ(), reason="OCC(OpenCascade) 不可用：需 conda 环境 occ")


def _square(z=0.0, x0=0.0, y0=0.0, s=1.0):
    """带 z/起点/边长参数的正方形闭合草图。"""
    return (Sketch(z=z).move_to(x0, y0).line_to(x0 + s, y0)
            .line_to(x0 + s, y0 + s).line_to(x0, y0 + s).close())


def test_c2_sketch_wire_and_face():
    """草图：闭合折线 → 线框（点数含回起点）+ 平面面（非空）。"""
    sk = _square()
    assert len(sk.points) == 5          # 4 点 + 闭合回起点
    w = sk.wire()
    assert shape_counts(w)["edges"] == 4
    f = sk.face()
    assert not f.IsNull() and shape_counts(f)["faces"] == 1


def test_c2_extrude_box():
    """拉伸：正方形面沿 Z → 六面体（6 面，三角化 12 三角）。"""
    sol = extrude(_square().face(), 0, 0, 3)
    assert shape_counts(sol)["faces"] == 6
    v, tr = tessellate(sol)
    assert len(tr) == 12 and len(v) >= 8


def test_c2_revolve_ring():
    """旋转：矩形剖面绕 Y 远轴 → 实心回转体（闭合剖面→面数稳定非空）。"""
    rect = (Sketch().move_to(3, 0).line_to(5, 0).line_to(5, 1)
            .line_to(3, 1).close().face())
    rev = revolve(rect, (0, 0, 0), (0, 1, 0))
    assert not rev.IsNull()
    assert shape_counts(rev)["faces"] >= 3
    assert len(tessellate(rev)[1]) > 0


def test_c2_loft_frustum():
    """放样：两个错位同形截面 → 6 面实体（斜台）。"""
    s1 = _square(x0=0, y0=0, s=1).wire()
    s2 = _square(x0=2, y0=0, s=1).wire()
    lo = loft([s1, s2])
    assert not lo.IsNull()
    assert shape_counts(lo)["faces"] == 6


def test_c2_pipe_sweep_mesh():
    """管道/扫掠：方形剖面沿直线 spine → 扫掠网格非空。"""
    pro = _square().wire()
    spine = wire3d([(0, 0, 0), (0, 0, 3)])
    for fn in (pipe, sweep):
        sh = fn(pro, spine)
        assert not sh.IsNull()
        v, tr = tessellate(sh)
        assert len(tr) > 0 and len(v) > 0


def test_c2_from_sketch_to_step_roundtrip():
    """C2↔C6 联动：拉伸体 → STEP 导出 → 重读 → 面数不变。"""
    sol = extrude(_square().face(), 0, 0, 3)
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "c2.step")
        export_shape(sol, p)
        assert os.path.getsize(p) > 0
        back = import_shape(p)
        assert shape_counts(back)["faces"] == 6


def test_c2_mesh_consistency_export():
    """C2 产物三角化与重读一致：拉伸六面体三角恒 12。"""
    sol = extrude(_square().face(), 0, 0, 3)
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "c2.brep")
        export_shape(sol, p)
        back = import_shape(p)
        v2, f2 = tessellate(back)
        assert len(f2) == 12
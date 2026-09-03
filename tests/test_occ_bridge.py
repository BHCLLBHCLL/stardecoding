# -*- coding: utf-8 -*-
"""C 波 B-Rep（OCC）能力：C1 三角化入图 + C6 STEP/IGES/BREP 格式往返面数一致。

依赖 conda 环境 occ（pythonocc-core），默认 Python 3.14 无 OCP → 本文件整体 skip
（occ_bridge.has_occ()==False），不影响 run_all 基线。真正执行用：
    <conda>/envs/occ/python.exe -m pytest tests/test_occ_bridge.py -q
"""
import os
import tempfile

import pytest

from occ_bridge import has_occ, import_surface, export_shape, import_shape, \
    tessellate, shape_counts

pytestmark = pytest.mark.skipif(
    not has_occ(), reason="OCC(OpenCascade) 不可用：需 conda 环境 occ")


@pytest.fixture()
def box():
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    return BRepPrimAPI_MakeBox(2.0, 3.0, 4.0).Shape()


def test_c1_tessellate_box(box):
    """C1：box 三角化入图，面数 12（6 面×2 三角）、顶点≥8。"""
    verts, faces = tessellate(box)
    assert faces.shape[1] == 3
    assert len(faces) == 12
    assert len(verts) >= 8


def test_c1_tessellate_cone_nonempty():
    """C1：曲面锥体三角化得到非零三角网（曲面才有细分价值）。"""
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCone
    shape = BRepPrimAPI_MakeCone(1.0, 0.5, 3.0).Shape()
    verts, faces = tessellate(shape)
    assert len(faces) > 16 and len(verts) > 16


def test_c6_shape_counts_box(box):
    """C6 锚点：box 的 B-Rep 面数恒为 6（边/顶因未缝合可翻倍，面数是验收锚）。"""
    c = shape_counts(box)
    assert c["faces"] == 6


def test_c6_step_roundtrip(box):
    """C6：STEP 写出→导入→三角化，面数与三角数一致。"""
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "box.step")
        export_shape(box, p)
        assert os.path.getsize(p) > 0
        sh = import_shape(p)
        assert shape_counts(sh)["faces"] == 6
        v2, f2 = tessellate(sh)
        assert len(f2) == 12 and len(v2) >= 8


def test_c6_iges_roundtrip(box):
    """C6：IGES 写出→导入，B-Rep 面数一致（严格 6 面）。"""
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "box.igs")
        export_shape(box, p)
        sh = import_shape(p)
        assert shape_counts(sh)["faces"] == 6


def test_c6_brep_roundtrip(box):
    """C6：BREP 写出→导入，box 面数一致（边/顶可能因未缝合而翻倍，仅面数验收）。"""
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "box.brep")
        export_shape(box, p)
        sh = import_shape(p)
        c = shape_counts(sh)
        assert c["faces"] == 6
        assert not sh.IsNull()


def test_c6_step_export_reimport_surface_import(box):
    """C6 一条龙：import_surface 读自写 STEP，顶点/三角/面数一致。"""
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "box.stp")
        export_shape(box, p)
        out = import_surface(p)
        assert out["ok"]
        assert out["n_triangles"] == 12
        assert out["counts"]["faces"] == 6


def test_occ_unsupported_format_rejected(box):
    """非 B-Rep 扩展名明确拒绝（不落入 OCC 读写分派）。"""
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError):
            export_shape(box, os.path.join(tmp, "box.xxx"))
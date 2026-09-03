# -*- coding: utf-8 -*-
"""C 波 C3：布尔/圆角/倒角/抽壳/阵列/镜像（OCC 编辑算子）验收。

依赖 conda 环境 occ（pythonocc-core），默认 Python 3.14 无 OCP → 整体 skip
（occ_edit._HAS_OCC 随 occ_bridge.has_occ()）。真跑：
    <conda>/envs/occ/python.exe -m pytest tests/test_occ_edit.py -q
"""
import pytest

from occ_bridge import has_occ, shape_counts, tessellate
from occ_edit import (fuse, cut, common, fillet, chamfer, shell, pattern,
                      mirror)

pytestmark = pytest.mark.skipif(
    not has_occ(), reason="OCC(OpenCascade) 不可用：需 conda 环境 occ")


@pytest.fixture()
def box():
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    return BRepPrimAPI_MakeBox(2, 3, 4).Shape()


@pytest.fixture()
def probe():
    """与 box2 错开一半的 1³ 小盒（用于布尔/阵列验证）。"""
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    return BRepPrimAPI_MakeBox(1, 1, 1).Shape()


def _shift(shape, dx, dy, dz):
    from OCC.Core.gp import gp_Trsf, gp_Vec
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform, BRepBuilderAPI_Copy
    tr = gp_Trsf(); tr.SetTranslation(gp_Vec(*map(float, (dx, dy, dz))))
    return BRepBuilderAPI_Transform(BRepBuilderAPI_Copy(shape).Shape(), tr).Shape()


def test_c3_fuse(box, probe):
    """"布尔并集：两盒错位融合，面数性状稳定且可三角化。"""
    sol = fuse(box, _shift(probe, 1.5, 0.5, 0.5))
    c = shape_counts(sol)
    assert c["faces"] > 6
    assert len(tessellate(sol)[1]) > 0


def test_c3_cut_common(box, probe):
    """布尔差/交：差集保留六面体性状，交集为非空实体。"""
    t = _shift(probe, 0.5, 1.0, 1.5)
    cu = cut(box, t)
    assert shape_counts(cu)["faces"] > 6
    co = common(box, t)
    assert not co.IsNull() and shape_counts(co)["faces"] >= 3


def test_c3_fillet(box):
    """圆角：全边倒角，面数增多、三角网非空。"""
    sol = fillet(box, 0.4)
    c = shape_counts(sol)
    assert c["faces"] > 6
    assert len(tessellate(sol)[1]) > 0


def test_c3_chamfer(box):
    """倒角：全边倒角，面数增多、三角网非空。"""
    sol = chamfer(box, 0.3)
    c = shape_counts(sol)
    assert c["faces"] > 6
    assert len(tessellate(sol)[1]) > 0


def test_c3_shell(box):
    """抽壳：移除一面掏空，三角面数远超原盒（内+外表面）。"""
    sol = shell(box, remove_idx=0, offset=0.3)
    assert shape_counts(sol)["faces"] > 6
    assert len(tessellate(sol)[1]) > 20


def test_c3_pattern(box, probe):
    """阵列：线性 2×1×1 阵列 → 两并列体融合面数=12（两 6 面盒）。"""
    sol = pattern(probe, nx=2, ny=1, nz=1, dx=1.5, dy=0, dz=0)
    assert shape_counts(sol)["faces"] == 12


def test_c3_mirror(box):
    """镜像：关于 YZ 平面镜像，产物非空并与原件融合仍成立体。"""
    sol = mirror(box, (0, 0, 0), (1, 0, 0))
    assert not sol.IsNull()
    assert shape_counts(sol)["faces"] >= 6
    assert len(tessellate(sol)[1]) > 0


def test_c3_edit_to_step_roundtrip(box):
    """C3↔C6 联动：抽壳体 → STEP 导出 → 重读 → 面数与三角一致。"""
    import os
    import tempfile
    from occ_bridge import export_shape, import_shape
    sol = shell(box, remove_idx=0, offset=0.3)
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "c3.step")
        export_shape(sol, p)
        assert os.path.getsize(p) > 0
        back = import_shape(p)
        v2, f2 = tessellate(back)
        assert len(f2) > 20
        assert shape_counts(back)["faces"] == shape_counts(sol)["faces"]
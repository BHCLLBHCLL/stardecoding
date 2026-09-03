# -*- coding: utf-8 -*-
"""C 波 C2：草图/拉伸/旋转/扫掠/放样/管道（OCC 内核构造算子）。

复用 `occ_bridge` 的 DLL 解析与可用性检测；构造算子各自惰性装配所需的
OCC 符号。所有算子输入/输出为 OCC TopoDS_Shape 或二维草图 Wire/Face，
产物可直接送 `occ_bridge.tessellate` 入图，或经 `occ_bridge.export_shape`
落 STEP/IGES/BREP。

非 OCC 环境（默认 Python 3.14）下相关调用抛 `OccUnavailable`。

算子（对照 STAR 3D-CAD 功能名）：
  - 草图：`Sketch` 2D 折线构造 → `wire()/face()`；
  - 拉伸：`extrude(shape, dx, dy, dz)` → BRepPrimAPI_MakePrism；
  - 旋转：`revolve(shape, axis_point, axis_vector, angle)` → MakeRevol；
  - 放样：`loft(sections)` → BRepOffsetAPI_ThruSections；
  - 管道：`pipe(profile, spine)` → BRepOffsetAPI_MakePipe；
  - 扫掠：`sweep(profile, spine)` → MakePipeShell。
"""
import importlib

from occ_bridge import OccUnavailable, has_occ

_HAS_OCC = has_occ()


def _need():
    """惰性装配 OCC 构造符号表；非 OCC 环境抛 OccUnavailable。"""
    if not _HAS_OCC:
        raise OccUnavailable("C2 算子需 conda 环境 occ（pythonocc-core）；默认 Python 3.14 不可用")
    return _ensure()


_OCC = None


def _ensure():
    global _OCC
    if _OCC is None:
        import numpy as npx
        try:
            gp = importlib.import_module("OCC.Core.gp")
            mods = {k: importlib.import_module("OCC.Core." + k) for k in
                    ("BRepBuilderAPI", "BRepPrimAPI", "BRepOffsetAPI", "TopoDS")}
        except Exception:
            _HAS_OCC = False  # noqa: F824
            raise OccUnavailable("OCC 构造库不可用")
        _OCC = {
            "gp": gp, "np": npx,
            "MakeEdge": mods["BRepBuilderAPI"].BRepBuilderAPI_MakeEdge,
            "MakeWire": mods["BRepBuilderAPI"].BRepBuilderAPI_MakeWire,
            "MakeFace": mods["BRepBuilderAPI"].BRepBuilderAPI_MakeFace,
            "MakePrism": mods["BRepPrimAPI"].BRepPrimAPI_MakePrism,
            "MakeRevol": mods["BRepPrimAPI"].BRepPrimAPI_MakeRevol,
            "ThruSections": mods["BRepOffsetAPI"].BRepOffsetAPI_ThruSections,
            "MakePipe": mods["BRepOffsetAPI"].BRepOffsetAPI_MakePipe,
            "MakePipeShell": mods["BRepOffsetAPI"].BRepOffsetAPI_MakePipeShell,
            "TopoDS_Shape": mods["TopoDS"].TopoDS_Shape,
        }
    return _OCC


class Sketch:
    """2D 草图（XY 平面）折线构造。

    move_to 设起点，line_to 续点（开链），close() 闭合回起点。产出线框
    `wire()` 或平面面 `face()`（需闭合）。
    """

    def __init__(self, z=0.0):
        self.z = z
        self._pts = []

    def move_to(self, x, y):
        self._pts = [(float(x), float(y))]
        return self

    def line_to(self, x, y):
        self._pts.append((float(x), float(y)))
        return self

    def close(self):
        if len(self._pts) > 1 and self._pts[0] != self._pts[-1]:
            self._pts.append(self._pts[0])
        return self

    @property
    def points(self):
        return list(self._pts)

    def wire(self):
        """逐段 MakeEdge 装配成 TopoDS_Wire。"""
        m = _need()
        BEM, BWM = m["MakeEdge"], m["MakeWire"]
        w = BWM()
        pts = self._pts
        for i in range(len(pts) - 1):
            (x0, y0), (x1, y1) = pts[i], pts[i + 1]
            e = BEM(m["gp"].gp_Pnt(x0, y0, self.z),
                    m["gp"].gp_Pnt(x1, y1, self.z)).Edge()
            w.Add(e)
        if w.IsDone():
            return w.Wire()
        raise ValueError("草图线框生成失败：至少需两点")

    def face(self):
        """闭合草图 → 平面面。"""
        return _need()["MakeFace"](self.wire()).Face()


def wire3d(points):
    """任意 3D 折线 → TopoDS_Wire（管道/扫掠的路径 spine）。"""
    m = _need()
    BEM, BWM = m["MakeEdge"], m["MakeWire"]
    w = BWM()
    for (a, b) in zip(points, points[1:]):
        e = BEM(m["gp"].gp_Pnt(*a), m["gp"].gp_Pnt(*b)).Edge()
        w.Add(e)
    if w.IsDone():
        return w.Wire()
    raise ValueError("3D 线框生成失败")


def extrude(shape, dx=0.0, dy=0.0, dz=1.0):
    """拉伸：沿 (dx,dy,dz) 平移生成棱柱/棱体。"""
    m = _need()
    mp = m["MakePrism"](shape, m["gp"].gp_Vec(dx, dy, dz))
    mp.Build()
    if mp.IsDone():
        return mp.Shape()
    raise ValueError("拉伸失败")


def revolve(shape, axis_point=(0, 0, 0), axis_vector=(0, 1, 0), angle=None):
    """旋转：绕 axis_point+axis_vector 轴扫 angle（弧度；缺省 2π）。"""
    m = _need()
    ax = m["gp"].gp_Ax1(m["gp"].gp_Pnt(*axis_point), m["gp"].gp_Dir(*axis_vector))
    mr = m["MakeRevol"](shape, ax, angle) if angle is not None \
        else m["MakeRevol"](shape, ax)
    mr.Build()
    if mr.IsDone():
        return mr.Shape()
    raise ValueError("旋转失败")


def loft(sections, solid=True):
    """放样：穿过一组截面（闭合线框）生成实体/壳。"""
    m = _need()
    ts = m["ThruSections"](solid, False, 1.0e-06)
    for s in sections:
        ts.AddWire(s)
    ts.Build()
    if ts.IsDone():
        return ts.Shape()
    raise ValueError("放样失败")


def pipe(profile, spine):
    """管道：profile 沿 spine（线框）扫掠成管道体。"""
    m = _need()
    mp = m["MakePipe"](spine, profile)
    mp.Build()
    if mp.IsDone():
        return mp.Shape()
    raise ValueError("管道生成失败")


def sweep(profile, spine):
    """扫掠：MakePipeShell 一般扫掠。"""
    m = _need()
    sh = m["MakePipeShell"](spine)
    sh.Add(profile)
    sh.Build()
    if sh.IsDone():
        return sh.Shape()
    raise ValueError("扫掠失败")
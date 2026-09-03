# -*- coding: utf-8 -*-
"""C 波 C3：布尔/圆角/倒角/抽壳/阵列/镜像（OCC 编辑算子）。

复用 `occ_bridge` 的 DLL 解析 + `shape_counts/tessellate`；编辑算子各自
惰性装配 OCC 符号。输入/输出均为 TopoDS_Shape，可继续送 `tessellate` 入图
或 `export_shape` 落 STEP/IGES/BREP。

非 OCC 环境（默认 Python 3.14）下相关调用抛 `OccUnavailable`。

算子（对照 STAR 3D-CAD 功能名）：
  - 布尔：`fuse(a, *rest)` / `cut(base, tool)` / `common(a, b)` → BRepAlgoAPI；
  - 圆角：`fillet(shape, radius, edges=None)` → MakeFillet（默认全部边）；
  - 倒角：`chamfer(shape, dist, edges=None)` → MakeChamfer（默认全部边）；
  - 抽壳：`shell(shape, remove_idx=None, offset=0.3)` → MakeThickSolidByJoin；
  - 阵列：`pattern(shape, nx,ny,nz, dx,dy,dz)` 线性阵列后 fuse 成体；
  - 镜像：`mirror(shape, plane_point=(0,0,0), plane_normal=(1,0,0))`。
"""
import importlib

from occ_bridge import OccUnavailable, has_occ

_HAS_OCC = has_occ()

_OCC = None


def _need():
    """惰性装配 C3 编辑符号表。"""
    if not _HAS_OCC:
        raise OccUnavailable("C3 算子需 conda 环境 occ（pythonocc-core）；默认 Python 3.14 不可用")
    global _OCC
    if _OCC is None:
        try:
            gp = importlib.import_module("OCC.Core.gp")
            o = {k: importlib.import_module("OCC.Core." + k) for k in
                 ("BRepAlgoAPI", "BRepFilletAPI", "BRepOffsetAPI",
                  "BRepBuilderAPI", "TopExp", "TopAbs", "TopTools")}
            M = o["BRepAlgoAPI"]
            F = o["BRepFilletAPI"]
            L = o["BRepOffsetAPI"]
            B = o["BRepBuilderAPI"]
            TT = o["TopTools"]
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
            _OCC = {
                "gp": gp,
                "Fuse": M.BRepAlgoAPI_Fuse, "Common": M.BRepAlgoAPI_Common,
                "Cut": M.BRepAlgoAPI_Cut,
                "MakeFillet": F.BRepFilletAPI_MakeFillet,
                "MakeChamfer": F.BRepFilletAPI_MakeChamfer,
                "MakeThickSolid": L.BRepOffsetAPI_MakeThickSolid,
                "Transform": B.BRepBuilderAPI_Transform,
                "Copy": B.BRepBuilderAPI_Copy,
                "TopTools_ListOfShape": TT.TopTools_ListOfShape,
                "Explorer": TopExp_Explorer,
                "TopAbs_EDGE": TopAbs_EDGE, "TopAbs_FACE": TopAbs_FACE,
            }
        except Exception as e:  # pragma: no cover
            raise OccUnavailable("OCC 编辑库不可用: %s" % e)
    return _OCC


def _edges_of(m, shape):
    out = []
    e = m["Explorer"](shape, m["TopAbs_EDGE"])
    while e.More():
        out.append(e.Current())
        e.Next()
    return out


def _faces_of(m, shape):
    out = []
    e = m["Explorer"](shape, m["TopAbs_FACE"])
    while e.More():
        out.append(e.Current())
        e.Next()
    return out


def fuse(a, *rest):
    """布尔并集：fuse(a, b, ...) 依次融合（>2 个折叠 Fuse）。"""
    m = _need()
    out = a
    for b in rest:
        r = m["Fuse"](out, b)
        r.Build()
        if not r.IsDone():
            raise ValueError("布尔并集失败")
        out = r.Shape()
    return out


def cut(base, tool):
    """布尔差集：base 减去 tool。"""
    m = _need()
    r = m["Cut"](base, tool)
    r.Build()
    if not r.IsDone():
        raise ValueError("布尔差集失败")
    return r.Shape()


def common(a, b):
    """布尔交集。"""
    m = _need()
    r = m["Common"](a, b)
    r.Build()
    if not r.IsDone():
        raise ValueError("布尔交集失败")
    return r.Shape()


def fillet(shape, radius, edges=None):
    """圆角：对指定边（缺省全部边）倒 radius 圆角。"""
    m = _need()
    mk = m["MakeFillet"](shape)
    for e in (edges if edges is not None else _edges_of(m, shape)):
        mk.Add(float(radius), e)
    mk.Build()
    if not mk.IsDone():
        raise ValueError("圆角失败")
    return mk.Shape()


def chamfer(shape, dist, edges=None):
    """倒角：对指定边（缺省全部边）做 dist 等距倒角。"""
    m = _need()
    mk = m["MakeChamfer"](shape)
    for e in (edges if edges is not None else _edges_of(m, shape)):
        mk.Add(float(dist), e)
    mk.Build()
    if not mk.IsDone():
        raise ValueError("倒角失败")
    return mk.Shape()


def shell(shape, remove_idx=0, offset=0.3, tol=1e-6):
    """抽壳：移除第 remove_idx 个面（缺省第 0 张）后向内 offset 掏空。"""
    m = _need()
    faces = _faces_of(m, shape)
    if not faces:
        raise ValueError("抽壳需要至少一个面")
    lfs = m["TopTools_ListOfShape"]()
    lfs.Append(faces[remove_idx])
    th = m["MakeThickSolid"]()
    th.MakeThickSolidByJoin(shape, lfs, float(offset), tol)
    if not th.IsDone():
        raise ValueError("抽壳失败")
    return th.Shape()


def pattern(shape, nx=2, ny=1, nz=1, dx=0.0, dy=0.0, dz=0.0):
    """线性阵列：沿 X/Y/Z 各复制 nx*ny*nz 份（间距 dx,dy,dz），融合成体。"""
    m = _need()
    occur = [shape]
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if i == 0 and j == 0 and k == 0:
                    continue
                tr = m["gp"].gp_Trsf()
                tr.SetTranslation(m["gp"].gp_Vec(i * dx, j * dy, k * dz))
                cp = m["Transform"](m["Copy"](shape).Shape(), tr).Shape()
                occur.append(cp)
    out = occur[0]
    for s in occur[1:]:
        r = m["Fuse"](out, s)
        r.Build()
        if not r.IsDone():
            raise ValueError("阵列失败")
        out = r.Shape()
    return out


def mirror(shape, plane_point=(0, 0, 0), plane_normal=(1, 0, 0)):
    """镜像：关于过 plane_point、法向 plane_normal 的平面镜像。"""
    m = _need()
    ax2 = m["gp"].gp_Ax2(m["gp"].gp_Pnt(*plane_point), m["gp"].gp_Dir(*plane_normal))
    tr = m["gp"].gp_Trsf()
    tr.SetMirror(ax2)
    return m["Transform"](m["Copy"](shape).Shape(), tr).Shape()
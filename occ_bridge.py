# -*- coding: utf-8 -*-
"""OCC(BRep / OpenCascade) 集成层：Part→TopoDS 映射、STEP/IGES/BREP 读写、三角化入图。

前提：conda 环境 `occ`（pythonocc-core 7.9.3，`import OCC`）。本模块自含 DLL 路径解析：
基于 OCC 包位置推断 conda 环境根，先把 `<root>/Library/bin(,lib)` 加入 PATH 与
os.add_dll_directory（Windows），因此无需先 conda activate。
非 OCC 环境（如默认 Python 3.14）下 `has_occ()==False`，所有函数报 `OccUnavailable`。

功能（对应 C 波验收）：
  - C6：`import_shape/export_shape` —— STEP(.step/.stp)/IGES(.igs/.iges)/BREP(.brep) 双向；
  - C1：`tessellate(shape)` —— B-Rep → (verts, faces) numpy 三角网，可送本客户端
        `mesh_polydata` 重建显示；`import_surface(path)` 一条龙读文件→三角网→面数。
"""
import os
import sys


class OccUnavailable(RuntimeError):
    pass


def _locate_env_root():
    """从 OCC 包路径反推 conda 环境根（site-packages 上两级 = <root>/Lib/site-packages）。"""
    try:
        import OCC
    except Exception:
        return None
    pkg = os.path.dirname(os.path.abspath(OCC.__file__))     # <root>/Lib/site-packages/OCC
    sp = os.path.dirname(pkg)                                 # <root>/Lib/site-packages
    return os.path.dirname(os.path.dirname(sp))               # <root>


def _ensure_dll_path():
    root = _locate_env_root()
    if not root:
        return
    for sub in ("Library/bin", "Library/lib"):
        d = os.path.join(root, sub)
        if d not in os.environ.get("PATH", ""):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory") and os.path.isdir(d):
            try:
                os.add_dll_directory(d)
            except Exception:
                pass


_ensure_dll_path()


def has_occ():
    """OCC 内核是否可用（可 import OCC 且能加载 DLL）。"""
    try:
        import OCC  # noqa: F401
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox  # noqa: F401
        return True
    except Exception:
        return False


_HAS_OCC = has_occ()


def _occ():
    if not _HAS_OCC:
        raise OccUnavailable(
            "OCC(OpenCascade) 不可用：请在 conda 环境 occ 下运行 "
            "(pythonocc-core 7.9.3)。默认 Python 3.14 无可用 wheel。")
    import numpy as npx
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox  # noqa: F401 (自检)
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepTools import breptools
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
    from OCC.Core.TopoDS import TopoDS_Shape
    from OCC.Core.STEPControl import (STEPControl_Writer, STEPControl_AsIs,
                                       STEPControl_Reader)
    from OCC.Core.IGESControl import (IGESControl_Writer, IGESControl_Reader,
                                       IGESControl_Controller)
    return {
        "np": npx, "BRep_MakeBox": None, "BRep_Builder": BRep_Builder,
        "BRepMesh_IncrementalMesh": BRepMesh_IncrementalMesh,
        "BRep_Tool": BRep_Tool, "breptools": breptools,
        "TopExp_Explorer": TopExp_Explorer,
        "TopAbs": (TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX),
        "TopoDS_Shape": TopoDS_Shape,
        "STEP_Writer": STEPControl_Writer, "STEP_AsIs": STEPControl_AsIs,
        "STEP_Reader": STEPControl_Reader,
        "IGES_Writer": IGESControl_Writer, "IGES_Reader": IGESControl_Reader,
        "IGES_Controller": IGESControl_Controller,
    }


def shape_counts(shape, occ=None):
    """B-Rep 拓扑计数：{faces, edges, vertices}（C6 面数验收锚点）。"""
    m = occ or _occ()
    nxp = m["np"]
    FACE, EDGE, VERTEX = m["TopAbs"]
    out = {}
    for name, type_ in (("faces", FACE), ("edges", EDGE), ("vertices", VERTEX)):
        exp = m["TopExp_Explorer"](shape, type_)
        c = 0
        while exp.More():
            c += 1
            exp.Next()
        out[name] = c
    out["_np"] = nxp
    return out


def tessellate(shape, deflection=0.1, angular=0.5, occ=None):
    """B-Rep → (verts[N,3], faces[M,3]) 三角网（C1 三角化入图）。

    deflection 弦长偏差，angular 弧度角公差；越大越粗。空体返回 (空,空)。
    """
    m = occ or _occ()
    npx = m["np"]
    # 先清掉既有三角化：BRepMesh 在同一 shape 上是增量（只加密不加密精度下降），
    # clean 后才可独立按本次 deflection/angular 重新细分（coarse/fine 交替可靠）。
    try:
        m["breptools"].Clean(shape)
    except Exception:
        pass
    # 5 参构造：shape, linearDeflection, isRelative, angularDeflection, isRelativeAng
    m["BRepMesh_IncrementalMesh"](shape, deflection, False, angular, False)
    FACE = m["TopAbs"][0]
    verts = []
    faces = []
    exp = m["TopExp_Explorer"](shape, FACE)
    while exp.More():
        face = exp.Current()
        tri = m["BRep_Tool"].Triangulation(face, face.Location())
        if tri is None:
            exp.Next()
            continue
        base = len(verts)
        nb = tri.NbNodes()
        for i in range(1, nb + 1):
            p = tri.Node(i)
            verts.append([p.X(), p.Y(), p.Z()])
        for i in range(1, tri.NbTriangles() + 1):
            t = tri.Triangle(i)
            faces.append([base + t.Value(1) - 1, base + t.Value(2) - 1,
                          base + t.Value(3) - 1])
        exp.Next()
    return (npx.asarray(verts, dtype=float),
            npx.asarray(faces, dtype=npx.int64).reshape(-1, 3))


def import_shape(path, occ=None):
    """读 STEP/IGES/BREP → TopoDS_Shape；格式按扩展名分派。"""
    m = occ or _occ()
    ext = os.path.splitext(path)[1].lower()
    if ext in (".step", ".stp"):
        return _read_step(path, m)
    if ext in (".igs", ".iges"):
        return _read_iges(path, m)
    if ext == ".brep":
        return _read_brep(path, m)
    raise ValueError("不支持的 B-Rep 导入格式: %s（支持 .step/.stp/.igs/.iges/.brep）"
                     % ext)


def _read_step(path, m):
    rd = m["STEP_Reader"]()
    st = rd.ReadFile(path)
    if st != 1:
        raise ValueError("STEP 读取失败（ReadFile=%s）: %s" % (st, path))
    rd.TransferRoots()
    n = rd.NbShapes()
    return rd.Shape(1)


def _read_iges(path, m):
    m["IGES_Controller"].Init()
    rd = m["IGES_Reader"]()
    st = rd.ReadFile(path)
    if st != 1:
        raise ValueError("IGES 读取失败（ReadFile=%s）: %s" % (st, path))
    rd.TransferRoots()
    return rd.Shape(1)


def _read_brep(path, m):
    rs = m["TopoDS_Shape"]()
    m["breptools"].Read(rs, path, m["BRep_Builder"]())
    return rs


def export_shape(shape, path, occ=None):
    """TopoDS_Shape → STEP/IGES/BREP 文件；格式按扩展名分派。返回已写路径。"""
    m = occ or _occ()
    ext = os.path.splitext(path)[1].lower()
    if ext in (".step", ".stp"):
        w = m["STEP_Writer"]()
        if w.Transfer(shape, m["STEP_AsIs"]) != 1:
            raise ValueError("STEP 传输失败")
        if w.Write(path) != 1:
            raise ValueError("STEP 写出失败: %s" % path)
    elif ext in (".igs", ".iges"):
        m["IGES_Controller"].Init()
        w = m["IGES_Writer"]()
        w.AddShape(shape)
        if not w.Write(path):
            raise ValueError("IGES 写出失败: %s" % path)
    elif ext == ".brep":
        m["breptools"].Write(shape, path)
    else:
        raise ValueError("不支持的 B-Rep 导出格式: %s（支持 .step/.stp/.igs/.iges/.brep）"
                         % ext)
    return path


def import_surface(path, deflection=0.1, occ=None):
    """C1/C6 一条龙：读 B-Rep 文件 → 三角化 → (vertices, faces, counts)。

    返回 {vertices, faces, n_vertices, n_triangles, counts, ok}；空体或失败抛错
    由调用方处理（ok=False 保留给非 OCC 环境降级）。
    """
    if not _HAS_OCC:
        return {"ok": False, "reason": "OCC 不可用"}
    shape = import_shape(path, occ=occ)
    verts, faces = tessellate(shape, deflection, occ=occ)
    counts = shape_counts(shape, occ=occ)
    return {"ok": True, "vertices": verts, "faces": faces,
            "n_vertices": len(verts), "n_triangles": len(faces),
            "counts": counts}
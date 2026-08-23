# -*- coding: utf-8 -*-
"""star_gui_vtk.py — 场景图 → VTK（对齐 cabdecoding/cstdecoding 的 cab_vtk 路线）。

- mesh_polydata: (N,3) 顶点 + (M,3) 面 → vtkPolyData（三角形）
- build_mesh_actors: 按 Part 分组 actor（复用 extract_mesh 的面/顶点 + Part.TriangleCount）
- axes_actor / orientation_marker_widget: 全局轴 + 方向指示器
- render_to_png: 离屏渲染（测试/导出）
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np


def _numpy_support():
    import vtk.util.numpy_support as nps
    return nps


def _colors():
    """缺省 Part 调色板（STAR-CCM+ 网格场景常用青绿 + 分色）。"""
    return [
        (0.00, 0.55, 0.62), (0.95, 0.45, 0.25), (0.25, 0.80, 0.45),
        (0.90, 0.80, 0.20), (0.75, 0.35, 0.85), (0.30, 0.85, 0.85),
        (0.90, 0.50, 0.70), (0.55, 0.55, 0.60),
    ]


# STAR-CCM+ 图形窗口背景：上浅灰 → 下白（实机截图）
STARCCM_BG_BOTTOM = (1.00, 1.00, 1.00)
STARCCM_BG_TOP = (0.82, 0.84, 0.86)


def apply_actor_transform(actor, translate=(0, 0, 0), scale=(1, 1, 1),
                          rotate_deg=(0, 0, 0)):
    """会话变换预览：缩放 → 旋转 → 平移。"""
    import vtk
    tf = vtk.vtkTransform()
    tf.PostMultiply()
    tf.Scale(float(scale[0]), float(scale[1]), float(scale[2]))
    if rotate_deg[0]:
        tf.RotateX(float(rotate_deg[0]))
    if rotate_deg[1]:
        tf.RotateY(float(rotate_deg[1]))
    if rotate_deg[2]:
        tf.RotateZ(float(rotate_deg[2]))
    tf.Translate(float(translate[0]), float(translate[1]), float(translate[2]))
    actor.SetUserTransform(tf)
    return actor


def apply_starccm_background(renderer):
    """把 vtkRenderer 设为 STAR-CCM+ 浅色渐变背景。"""
    renderer.SetBackground(*STARCCM_BG_BOTTOM)
    try:
        renderer.SetBackground2(*STARCCM_BG_TOP)
        renderer.GradientBackgroundOn()
    except Exception:
        pass
    return renderer


def mesh_polydata(vertices, faces, one_based=True):
    """顶点/面 → vtkPolyData（三角形 + 点法向，对齐 cab_vtk._tris_to_polydata）。

    one_based 保留兼容；实际用 compact_face_indices 处理 0/1 基和分块偏移，
    避免超界下标被 clip 到末顶点（外框拧成星形）。
    """
    import vtk
    from sim_parser import compact_face_indices
    nps = _numpy_support()
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if f.size == 0 or v.size == 0:
        raise ValueError("空网格")
    f, _ok = compact_face_indices(f, v.shape[0])
    f = np.asarray(f, dtype=np.int64)
    _ = one_based  # API 兼容；映射由 compact_face_indices 完成
    pd = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.SetData(nps.numpy_to_vtk(np.ascontiguousarray(v), deep=True))
    pd.SetPoints(pts)
    n = int(f.shape[0])
    cells = np.empty(n * 4, dtype=np.int64)
    cells[0::4] = 3
    cells[1::4] = f[:, 0]
    cells[2::4] = f[:, 1]
    cells[3::4] = f[:, 2]
    ca = vtk.vtkCellArray()
    try:
        idarr = nps.numpy_to_vtkIdTypeArray(cells, deep=True)
        if hasattr(ca, "ImportLegacyFormat"):
            ca.ImportLegacyFormat(idarr)
        else:
            ca.SetCells(n, idarr)
    except Exception:
        for tri in f:
            ca.InsertNextCell(3, [int(tri[0]), int(tri[1]), int(tri[2])])
    pd.SetPolys(ca)
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(pd)
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    normals.SplittingOn()
    normals.SetFeatureAngle(45.0)
    normals.ConsistencyOn()
    try:
        normals.AutoOrientNormalsOn()
    except Exception:
        pass
    normals.Update()
    return normals.GetOutput()


def _face_vertex_wants(faces):
    """根据面下标推断配套顶点表规模（跨度优先，再 0/1 基）。"""
    lo, hi = int(faces.min()), int(faces.max())
    span = hi - lo + 1
    wants = []
    for w in (span, hi if lo >= 1 else 0, hi + 1):
        if w > 0 and w not in wants:
            wants.append(w)
    return lo, hi, wants


def _bind_part_vertices(sim, faces, face_i, used):
    """为某 Part 的面数组配对 Float8 顶点：规模吻合且数组下标靠近面数组。"""
    _lo, _hi, wants = _face_vertex_wants(faces)
    cands = []
    for i, a in enumerate(sim.arrays):
        if i in used or i == face_i:
            continue
        if a["type"] != "Float8" or a["count"] % 3:
            continue
        n = a["count"] // 3
        if n not in wants:
            continue
        data = sim.array_data(i)
        # 大表排除法向（|v|≤1）；8 顶点块体即使落在单位立方也保留
        if n > 32 and float(np.abs(data).max()) <= 1.0:
            continue
        cands.append((abs(i - face_i), i, n))
    if not cands:
        return None, None
    cands.sort()
    i = cands[0][1]
    return i, np.asarray(sim.array_data(i).reshape(-1, 3))


def part_meshes(sim):
    """按 Part 抽取网格：每个 Part 绑定自己的顶点表 + 面索引。

    不能共用 extract_mesh() 的全局顶点。12 三角形外框的面号是 1..8，若灌进
    风洞的十几万顶点表，会画成隧道前 8 个点；风洞面号若是 min=k 的分块偏移，
    clip 到末顶点会拧成星形。按 TriangleCount 从大到小认领尚未使用的数组。
    """
    cached = getattr(sim, "_part_meshes_cache", None)
    if cached is not None:
        return cached
    from sim_parser import compact_face_indices
    items = []
    seen = set()
    parts = []
    for o in sim.objects:
        if o.dict.get("ImportedVertices") and o.dict.get("ImportedFaces"):
            iv = np.asarray(o.dict["ImportedVertices"], dtype=float)
            iff = np.asarray(o.dict["ImportedFaces"], dtype=np.int64)
            parts.append({"name": o.name or "part%d" % o.id, "id": o.id,
                          "triangles": int(iff.shape[0]), "vertices": iv,
                          "faces": iff, "face_types": None, "one_based": False})
            if o.name:
                seen.add(o.name)
            continue
        t = o.dict.get("TriangleCount")
        if not isinstance(t, int) or t <= 0:
            continue
        if o.name in seen:
            continue
        seen.add(o.name)
        items.append((t, o))
    items.sort(key=lambda x: (-x[0], x[1].id))

    used = set()
    fallback = None
    for t, o in items:
        face_i = None
        faces = None
        for i, a in enumerate(sim.arrays):
            if i in used:
                continue
            if a["type"] in ("Unsigned4", "Integer4") and a["count"] == t * 3:
                faces = np.asarray(sim.array_data(i).reshape(-1, 3), dtype=np.int64)
                face_i = i
                break
        if faces is None:
            continue
        used.add(face_i)
        vert_i, verts = _bind_part_vertices(sim, faces, face_i, used)
        if verts is None:
            if fallback is None:
                m = sim.extract_mesh()
                fallback = m.get("vertices")
            verts = fallback
            vert_i = None
        if verts is None:
            continue
        if vert_i is not None:
            used.add(vert_i)
        faces, _ok = compact_face_indices(faces, verts.shape[0])
        face_types = _bind_face_types(sim, t, used)
        parts.append({"name": o.name or "part%d" % o.id,
                      "id": o.id, "triangles": t, "vertices": verts,
                      "faces": np.asarray(faces, dtype=np.int64),
                      "face_types": face_types,
                      "one_based": False})
    sim._part_meshes_cache = parts
    return parts


def _first_array(sim, types, count, used, near=None):
    if not isinstance(count, int) or count <= 0:
        return None
    cands = [i for i, a in enumerate(sim.arrays)
             if i not in used and a["type"] in types and a["count"] == count]
    if not cands:
        return None
    if near is not None:
        cands.sort(key=lambda i: abs(i - near))
    return cands[0]


def _bind_ssm(sim, ssm, used):
    """按 SerializableSurfaceMesh 的 *Length 字段绑定面/顶点/FaceTypes。"""
    d = ssm.dict or {}
    nfv = d.get("FaceVerticesLength")
    nvc = d.get("VertexCoordsLength")
    nft = d.get("FaceTypesLength")
    if not isinstance(nfv, int) or not isinstance(nvc, int) or nfv % 3 or nvc % 3:
        return None
    from sim_parser import compact_face_indices
    face_i = _first_array(sim, ("Unsigned4",), nfv, used)
    if face_i is None:
        face_i = _first_array(sim, ("Integer4",), nfv, used)
    if face_i is None:
        return None
    vert_i = _first_array(sim, ("Float8",), nvc, used, near=face_i)
    if vert_i is None:
        return None
    used.add(face_i)
    used.add(vert_i)
    faces = np.asarray(sim.array_data(face_i).reshape(-1, 3), dtype=np.int64)
    verts = np.asarray(sim.array_data(vert_i).reshape(-1, 3))
    faces, _ok = compact_face_indices(faces, verts.shape[0])
    ft = None
    if isinstance(nft, int) and nft == faces.shape[0]:
        ft_i = _first_array(sim, ("Integer4",), nft, used, near=face_i)
        if ft_i is None:
            ft_i = _first_array(sim, ("Unsigned4",), nft, used, near=face_i)
        if ft_i is not None:
            data = np.asarray(sim.array_data(ft_i), dtype=np.int64)
            nuniq = int(np.unique(data).size)
            if 1 <= nuniq < nft:
                used.add(ft_i)
                ft = data
    return {"vertices": verts, "faces": np.asarray(faces, dtype=np.int64),
            "face_types": ft, "one_based": False, "triangles": int(faces.shape[0]),
            "ssm_id": ssm.id}


def serializable_surface_meshes(sim):
    """全部 SerializableSurfaceMesh → 数组绑定（按对象 id 顺序认领，避免同规模块体抢数组）。"""
    cached = getattr(sim, "_ssm_meshes_cache", None)
    if cached is not None:
        return cached
    ssms = [o for o in sim.objects
            if o.class_name == "star.meshing.SerializableSurfaceMesh"]
    ssms.sort(key=lambda o: o.id)
    used = set()
    out = {}
    for ssm in ssms:
        bound = _bind_ssm(sim, ssm, used)
        if bound is not None:
            out[ssm.id] = bound
    sim._ssm_meshes_cache = out
    return out


def representation_source_id(sim, displayer):
    """显示器对应的表面描述：Mesh=True 用 Remesh/Latest，否则 CAD 三角化（Source=0）。

    Geometry Scene 的 Outline/Geometry 即使 Representation 字段仍指向 Remesh，
    官方画的是光滑 CAD 曲面 + 域边框，不能把计算面网格的每条边当轮廓。
    """
    if displayer is None:
        return 0
    if not bool((displayer.dict or {}).get("Mesh")):
        return 0
    om = sim.objmap
    rep = om.get(displayer.dict.get("Representation") or -1)
    if rep is None:
        return 0
    src = (rep.dict or {}).get("Source")
    if isinstance(src, int) and src in om:
        return src
    return 0


def _part_ssm_ids(sim, part_obj):
    """MeshPart.Descriptions → [(source_id, ssm_id), ...]。"""
    om = sim.objmap
    mgr = om.get((part_obj.dict or {}).get("Descriptions") or -1)
    out = []
    for k in (mgr.dict.get("Keys") if mgr is not None else None) or []:
        desc = om.get(k)
        if desc is None:
            continue
        src = desc.dict.get("Source") or 0
        if not isinstance(src, int):
            src = 0
        sm = om.get(desc.dict.get("SurfaceMesh") or -1)
        if sm is None:
            continue
        ssm = om.get(sm.dict.get("SerializableSurfaceMesh") or -1)
        if ssm is not None:
            out.append((src, ssm.id))
    return out


def mesh_bundle_for_part(sim, part_obj, source_id, tess_by_id=None):
    """按显示器 Representation 取 Part 的表面：Remesh/Latest 用计算面网格，否则 CAD 三角化。"""
    ssm_map = serializable_surface_meshes(sim)
    pairs = _part_ssm_ids(sim, part_obj) if part_obj is not None else []
    ssm_id = None
    for src, sid in pairs:
        if src == source_id:
            ssm_id = sid
            break
    if ssm_id is None and source_id:
        src_obj = sim.objmap.get(source_id)
        want = (src_obj.dict or {}).get("FaceCount") if src_obj is not None else None
        nonzero = [(src, sid) for src, sid in pairs if src]
        if want and nonzero:
            def _face_n(sid):
                b = ssm_map.get(sid)
                return b["triangles"] if b else 10 ** 9
            ssm_id = min(nonzero, key=lambda it: abs(_face_n(it[1]) - int(want)))[1]
        elif nonzero:
            ssm_id = nonzero[0][1]
    if ssm_id is None:
        for src, sid in pairs:
            if src == 0:
                ssm_id = sid
                break
    if ssm_id is not None and ssm_id in ssm_map:
        b = dict(ssm_map[ssm_id])
        if part_obj is not None:
            b["id"] = part_obj.id
            b["name"] = part_obj.name or b.get("name")
        return b
    if tess_by_id and part_obj is not None:
        return tess_by_id.get(part_obj.id)
    return None


def _bind_face_types(sim, ntri, used):
    """Integer4/Unsigned4、长度=三角面数、唯一值远少于面数 → CAD FaceTypes。"""
    cands = []
    for i, a in enumerate(sim.arrays):
        if i in used:
            continue
        if a["type"] not in ("Integer4", "Unsigned4") or a["count"] != ntri:
            continue
        data = np.asarray(sim.array_data(i))
        nuniq = int(np.unique(data).size)
        if 1 < nuniq < ntri:
            cands.append((nuniq, i, data.astype(np.int64, copy=False)))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0])
    used.add(cands[0][1])
    return cands[0][2]


# 超过该三角面数不再 CPU 抽取 vtkFeatureEdges（直升机风洞 24 万面会卡数分钟）
LARGE_EDGE_EXTRACT = 20000


def _actor(pd, color, opacity=1.0, wireframe=False, line_width=1.0, user_data=None,
           edge_color=None):
    import vtk
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(pd)
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(*color)
    prop.SetOpacity(opacity)
    prop.SetAmbient(0.28)
    prop.SetDiffuse(0.72)
    prop.SetSpecular(0.12)
    prop.SetSpecularPower(18)
    if wireframe:
        prop.SetRepresentationToWireframe()
    prop.SetLineWidth(line_width)
    if edge_color:
        prop.SetEdgeVisibility(1)
        prop.SetEdgeColor(*tuple(float(x) for x in edge_color[:3]))
    return actor


def build_mesh_actors(sim, wireframe=False, opacity=1.0):
    """按 Part 构建网格 actors。返回 [(key, name, part_id, actor)]。"""
    parts = part_meshes(sim)
    colors = _colors()
    actors = []
    for idx, p in enumerate(parts):
        pd = mesh_polydata(p["vertices"], p["faces"], one_based=p["one_based"])
        color = colors[idx % len(colors)]
        actor = _actor(pd, color, opacity=opacity, wireframe=wireframe)
        actors.append(("part:%s" % p["name"], p["name"], p["id"], actor))
    return actors


def edges_actor(pd, color=(0.15, 0.15, 0.18), line_width=1.0, mode="mesh",
                feature_angle=15.0):
    """网格边线 actor。

    mode='mesh'：边界+流形边（Mesh Scene）。mode='outline'：特征边+边界
    （Geometry Scene 的域边框）。outline 失败时不回退 ExtractEdges，
    否则会把每个三角面的边都画出来，风洞里全是蛛网。
    """
    import vtk
    edge_pd = None
    try:
        ext = vtk.vtkFeatureEdges()
        ext.SetInputData(pd)
        ext.BoundaryEdgesOn()
        ext.NonManifoldEdgesOff()
        ext.ColoringOff()
        if mode == "outline":
            ext.ManifoldEdgesOff()
            ext.FeatureEdgesOn()
            try:
                ext.SetFeatureAngle(float(feature_angle))
            except Exception:
                pass
        else:
            ext.ManifoldEdgesOn()
            ext.FeatureEdgesOff()
        ext.Update()
        edge_pd = ext.GetOutput()
        if edge_pd is None or edge_pd.GetNumberOfCells() == 0:
            raise RuntimeError("empty feature edges")
    except Exception:
        if mode == "outline":
            return None
        ext2 = vtk.vtkExtractEdges()
        ext2.SetInputData(pd)
        ext2.Update()
        edge_pd = ext2.GetOutput()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(edge_pd)
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(*color)
    prop.SetRepresentationToWireframe()
    prop.SetLineWidth(line_width)
    return actor


def outline_bounds_actor(pd, color=(0.12, 0.12, 0.14), line_width=1.0):
    """大网格 Outline：包围盒线框（vtkOutlineFilter），避免 FeatureEdges 卡死。"""
    import vtk
    of = vtk.vtkOutlineFilter()
    of.SetInputData(pd)
    of.Update()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(of.GetOutput())
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(*color)
    prop.SetRepresentationToWireframe()
    prop.SetLineWidth(line_width)
    return actor


def axes_actor(length=1.0):
    """全局坐标轴（RGB = XYZ）。"""
    import vtk
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(length, length, length)
    axes.SetShaftType(0)
    return axes


def orientation_marker_widget(interactor, size_frac=0.17):
    """左下角方向指示器（STAR-CCM+ 截图 / cab_vtk：viewport (0,0,s,s)）。"""
    import vtk
    marker = vtk.vtkOrientationMarkerWidget()
    marker.SetOrientationMarker(axes_actor(1.0))
    marker.SetInteractor(interactor)
    marker.SetViewport(0.0, 0.0, size_frac, size_frac)
    marker.SetEnabled(1)
    marker.InteractiveOff()
    return marker


# STAR-CCM+ 视图按钮：+X/-X/+Y/-Y/+Z/-Z / 等轴测（相机位置相对焦点的方向 + ViewUp）
VIEW_PRESETS = {
    "+x": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "-x": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "+y": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "-y": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "+z": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "-z": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    "iso": ((1.0, 1.0, 1.0), (0.0, 0.0, 1.0)),
}


def apply_view_preset(renderer, name):
    """正交/等轴测预设。name: +x/-x/+y/-y/+z/-z/iso。"""
    key = (name or "").lower().strip()
    if key not in VIEW_PRESETS:
        return False
    direction, up = VIEW_PRESETS[key]
    cam = renderer.GetActiveCamera()
    fp = cam.GetFocalPoint()
    cam.SetFocalPoint(*fp)
    cam.SetPosition(fp[0] + direction[0], fp[1] + direction[1], fp[2] + direction[2])
    cam.SetViewUp(*up)
    try:
        cam.ParallelProjectionOn()
    except Exception:
        pass
    renderer.ResetCamera()
    return True


def scene_displayers(sim, scene_obj):
    """场景的 PartDisplayer 列表（DisplayerManager.Keys → star.vis.*）。"""
    om = sim.objmap
    dm = om.get(scene_obj.dict.get("DisplayerManager") or -1)
    out = []
    if dm is None:
        return out
    for k in dm.dict.get("Keys") or []:
        d = om.get(k)
        if d is not None and (d.class_name or "").startswith("star.vis"):
            out.append(d)
    return out


def scene_background(sim, scene_obj):
    """场景背景：BackgroundColorMode + Solid/Gradient 颜色。"""
    om = sim.objmap
    mode = scene_obj.dict.get("BackgroundColorMode", 0)
    solid = scene_obj.dict.get("SolidBackgroundColor")
    grad = scene_obj.dict.get("GradientBackgroundColor")
    bg = STARCCM_BG_BOTTOM
    if isinstance(solid, int):
        o = om.get(solid)
        c = o.dict.get("Color") if o is not None else None
        if c:
            bg = tuple(float(x) for x in c[:3])
    return {"mode": mode, "solid": bg,
            "gradient": {"mode": 0, "color1": (1.0, 1.0, 1.0),
                         "color2": (0.93, 0.93, 0.93)}}


def scene_camera(sim, scene_obj):
    """从 Scene.CurrentView 读相机参数（Position/FocalPoint/ViewUp/ParallelScale）。"""
    om = sim.objmap
    view = om.get(scene_obj.dict.get("CurrentView") or -1)
    if view is None:
        return None

    def coord_val(ref):
        o = om.get(ref) if isinstance(ref, int) else None
        if o is not None:
            v = o.dict.get("Value")
            if isinstance(v, list) and len(v) == 3:
                return tuple(float(x) for x in v)
        return None

    cam = {
        "position": coord_val(view.dict.get("Position")),
        "focal": coord_val(view.dict.get("FocalPoint")),
        "view_up": coord_val(view.dict.get("ViewUp")),
        "parallel_scale": view.dict.get("ParallelScale"),
        "projection": view.dict.get("ProjectionMode", 0),
        "name": view.name,
    }
    return cam


def apply_camera(renderer, cam):
    """把场景相机参数应用到 vtkRenderer 的相机。"""
    if cam is None:
        return False
    camera = renderer.GetActiveCamera()
    if cam.get("projection") == 1:
        camera.ParallelProjectionOn()
    else:
        camera.ParallelProjectionOff()
    if cam.get("position") and cam.get("focal"):
        camera.SetPosition(*cam["position"])
        camera.SetFocalPoint(*cam["focal"])
        if cam.get("view_up"):
            camera.SetViewUp(*cam["view_up"])
    if cam.get("parallel_scale"):
        camera.SetParallelScale(float(cam["parallel_scale"]))
    return True


def _subset_part_polydata(p, cad_ids):
    """按 CAD FaceTypes 切出 PartSurface / Boundary 对应的三角面子块。"""
    faces = np.asarray(p["faces"], dtype=np.int64)
    verts = p["vertices"]
    ft = p.get("face_types")
    if cad_ids and ft is not None and len(ft) == faces.shape[0]:
        mask = np.isin(ft, np.fromiter(cad_ids, dtype=np.int64))
        if mask.any() and not bool(mask.all()):
            faces = faces[mask]
        elif not mask.any():
            return None
    if faces.shape[0] == 0:
        return None
    used = np.unique(faces)
    if used.size < verts.shape[0]:
        remap = np.full(int(faces.max()) + 1, -1, dtype=np.int64)
        remap[used] = np.arange(used.size, dtype=np.int64)
        verts = verts[used]
        faces = remap[faces]
    return mesh_polydata(verts, faces, one_based=False), int(faces.shape[0])


def _coord_xyz(sim, ref):
    o = sim.objmap.get(ref) if isinstance(ref, int) else None
    if o is None:
        return None
    v = o.dict.get("Value")
    if isinstance(v, (list, tuple)) and len(v) == 3:
        return tuple(float(x) for x in v)
    return None


def _scalar_value(sim, ref):
    o = sim.objmap.get(ref) if isinstance(ref, int) else None
    if o is None:
        return 0.0
    q = sim.objmap.get(o.dict.get("ValueQuantity") or -1)
    if q is None:
        return float(o.dict.get("Value") or 0.0)
    for k in ("Value", "SIValue", "RawValue"):
        v = q.dict.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return 0.0


def plane_section_actor(sim, plane_obj, bounds, line_width=2.0):
    """Derived-part 平面截面：用输入部件包围盒切出矩形轮廓（对齐 STAR-CCM+ Plane Section）。"""
    import vtk
    origin = _coord_xyz(sim, plane_obj.dict.get("Origin")) or (0.0, 0.0, 0.0)
    normal = _coord_xyz(sim, plane_obj.dict.get("Orientation")) or (0.0, 1.0, 0.0)
    offset = _scalar_value(sim, plane_obj.dict.get("SingleValue"))
    push = float(plane_obj.dict.get("PushAmount") or 0.0)
    origin = (origin[0] + normal[0] * (offset + push),
              origin[1] + normal[1] * (offset + push),
              origin[2] + normal[2] * (offset + push))
    color = tuple(float(x) for x in (plane_obj.dict.get("Color") or (1.0, 0.2, 0.55))[:3])
    if color[1] > 0.7 and color[2] > 0.55:
        color = (0.95, 0.15, 0.62)
    cube = vtk.vtkCubeSource()
    cube.SetBounds(bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5])
    plane = vtk.vtkPlane()
    plane.SetOrigin(*origin)
    plane.SetNormal(*normal)
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputConnection(cube.GetOutputPort())
    cutter.Update()
    cut_pd = cutter.GetOutput()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(cut_pd)
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(*color)
    prop.SetRepresentationToWireframe()
    prop.SetLineWidth(line_width)
    prop.SetEdgeVisibility(1)
    prop.SetEdgeColor(*color)
    prop.SetOpacity(1.0)
    return actor


def _input_parts_bounds(sim, plane_obj, parts_by_id):
    from star_gui_model import owning_mesh_part
    om = sim.objmap
    pg = om.get(plane_obj.dict.get("InputParts") or -1)
    xs, ys, zs = [], [], []
    for k in (pg.dict.get("Keys") if pg is not None else None) or []:
        src = om.get(k)
        part = owning_mesh_part(sim, src) if src is not None else None
        if part is None:
            continue
        p = parts_by_id.get(part.id)
        if p is None:
            continue
        v = p["vertices"]
        xs += [float(v[:, 0].min()), float(v[:, 0].max())]
        ys += [float(v[:, 1].min()), float(v[:, 1].max())]
        zs += [float(v[:, 2].min()), float(v[:, 2].max())]
    if not xs:
        return None
    return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))


def build_scene_actors(sim, scene_obj, fallback_palette=True):
    """按场景 Displayer.Collector 的源对象建 actors。

    Geometry Scene（Mesh=False）：CAD 三角化光滑着色 + Outline 域边框。
    Mesh Scene（Mesh=True）：Representation=Remesh 的计算面网格。
    Boundary/PartSurface 用 FaceTypes 切面子块。PlaneSection 画截面轮廓。
    """
    from star_gui_model import (
        collector_sources, owning_mesh_part, part_surface_cad_ids,
    )
    parts = part_meshes(sim)
    if not parts:
        return [], None
    by_id = {p["id"]: p for p in parts}
    colors = _colors()
    actors = []
    edges = []
    disp = scene_displayers(sim, scene_obj)
    trans_override = int((scene_obj.dict or {}).get("TransparencyOverrideMode") or 0)
    pd_cache = {}

    def poly_for_source(p, cad_ids):
        key = (p["id"], p.get("ssm_id"), tuple(sorted(cad_ids)) if cad_ids else None)
        if key in pd_cache:
            return pd_cache[key]
        got = _subset_part_polydata(p, cad_ids)
        pd_cache[key] = got
        return got

    filtered = False
    color_i = 0
    for d in disp:
        sources = collector_sources(sim, d)
        if not sources:
            continue
        want_surface = bool(d.dict.get("Surface"))
        want_outline = bool(d.dict.get("Outline"))
        want_mesh = bool(d.dict.get("Mesh"))
        source_id = representation_source_id(sim, d)
        dc = d.dict.get("DisplayerColor")
        mc = tuple(float(x) for x in (d.dict.get("MeshColor") or (0.12, 0.12, 0.14))[:3])
        lw = float(d.dict.get("LineWidth", 1.0) or 1.0)
        emode = "outline" if want_outline and not want_mesh else "mesh"
        for src in sources:
            cn = src.class_name or ""
            if "PlaneSection" in cn:
                b = _input_parts_bounds(sim, src, by_id)
                if b is None:
                    continue
                filtered = True
                actors.append(("plane:%s:%d" % (src.name or "plane", d.id),
                               src.name or "Plane Section", src.id,
                               plane_section_actor(sim, src, b, line_width=max(lw, 2.0))))
                continue
            if not (want_surface or want_outline or want_mesh):
                continue
            part_obj = owning_mesh_part(sim, src)
            if part_obj is None:
                continue
            p = mesh_bundle_for_part(sim, part_obj, source_id, tess_by_id=by_id)
            if p is None:
                continue
            cad = part_surface_cad_ids(sim, src)
            got = poly_for_source(p, cad)
            if not got:
                continue
            pd, ntri = got
            filtered = True
            color = colors[color_i % len(colors)]
            if d.dict.get("UseDisplayerColor") and dc:
                color = tuple(float(x) for x in dc[:3])
            # 透明度覆盖：只作用在计算域外壁（极少三角的 CAD 面），模型本体保持不透明
            opacity = float(d.dict.get("Opacity", 1.0))
            if trans_override == 1 and want_surface and ntri <= 32:
                opacity = min(opacity, 0.38)
            gpu_edges = mc if (want_mesh and ntri > LARGE_EDGE_EXTRACT) else None
            label = src.name or p.get("name") or part_obj.name
            if want_surface or gpu_edges:
                actors.append(("part:%s:%d" % (label, d.id), label, src.id,
                               _actor(pd, color, opacity=opacity, edge_color=gpu_edges,
                                      line_width=lw)))
            if want_outline or want_mesh:
                if ntri <= LARGE_EDGE_EXTRACT:
                    ea = edges_actor(pd, color=mc, line_width=lw, mode=emode,
                                     feature_angle=float(d.dict.get("FeatureAngle", 15.0)))
                    if ea is not None:
                        edges.append(("edges:%s:%d" % (label, d.id), label, src.id, ea))
            color_i += 1

    if not filtered:
        for idx, p in enumerate(parts):
            pd = mesh_polydata(p["vertices"], p["faces"], one_based=p["one_based"])
            color = colors[idx % len(colors)]
            actors.append(("part:%s" % p["name"], p["name"], p["id"],
                           _actor(pd, color)))
    return actors + edges, scene_camera(sim, scene_obj)


def bounds_of(actors):
    """actors 合并包围盒 → (xmin,xmax,ymin,ymax,zmin,zmax)。"""
    xs, ys, zs = [], [], []
    for _k, _n, _pid, actor in actors:
        b = actor.GetBounds()
        if b and b[1] >= b[0]:
            xs += [b[0], b[1]]
            ys += [b[2], b[3]]
            zs += [b[4], b[5]]
    if not xs:
        return None
    return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))


def render_offscreen_png(actors, out_path, size=(1200, 900), background=None):
    """离屏渲染 actors → PNG（测试/验证）。"""
    import vtk
    if background is None:
        background = STARCCM_BG_BOTTOM
    ren = vtk.vtkRenderer()
    apply_starccm_background(ren)
    if background != STARCCM_BG_BOTTOM:
        ren.SetBackground(*background)
    for _k, _n, _pid, actor in actors:
        ren.AddActor(actor)
    b = bounds_of(actors)
    if b:
        ren.ResetCamera()
    ren.GetActiveCamera().Zoom(1.0)
    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    rw.SetSize(*size)
    rw.AddRenderer(ren)
    rw.Render()
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(rw)
    w2i.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(out_path)
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()
    return out_path

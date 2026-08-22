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
    """缺省 Part 调色板（STAR-CCM+ 风格常用色）。"""
    return [
        (0.30, 0.60, 0.95), (0.95, 0.45, 0.25), (0.25, 0.80, 0.45),
        (0.90, 0.80, 0.20), (0.75, 0.35, 0.85), (0.30, 0.85, 0.85),
        (0.90, 0.50, 0.70), (0.55, 0.55, 0.60),
    ]


def mesh_polydata(vertices, faces, one_based=True):
    """顶点/面 → vtkPolyData（三角形面片，1 基索引自动校正）。"""
    import vtk
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if f.size == 0 or v.size == 0:
        raise ValueError("空网格")
    if one_based and f.min() >= 1:
        f = f - 1
    f = np.clip(f, 0, v.shape[0] - 1)
    pd = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.SetData(_numpy_support().numpy_to_vtk(v, deep=True))
    pd.SetPoints(pts)
    cells = vtk.vtkCellArray()
    for tri in f:
        cells.InsertNextCell(3, [int(tri[0]), int(tri[1]), int(tri[2])])
    pd.SetPolys(cells)
    return pd


def part_meshes(sim):
    """按 Part 抽取网格：返回 [(name, vertices, faces, one_based), ...]。

    顶点共用 extract_mesh() 的顶点表；每个 Part 按 TriangleCount×3 匹配各自
    的面索引数组（复用改进②的 part: 匹配逻辑）。
    """
    from sim_parser import SimFile  # noqa: F401
    m = sim.extract_mesh()
    if m["vertices"] is None:
        return []
    verts = m["vertices"]
    one_based = bool(m["faces"] is not None and m["faces"].size and m["faces"].min() >= 1)
    parts = []
    seen = set()
    for o in sim.objects:
        t = o.dict.get("TriangleCount")
        if not isinstance(t, int) or t <= 0:
            continue
        if o.name in seen:
            continue
        seen.add(o.name)
        faces = None
        for i, a in enumerate(sim.arrays):
            if a["type"] in ("Unsigned4", "Integer4") and a["count"] == t * 3:
                faces = sim.array_data(i).reshape(-1, 3)
                break
        if faces is None:
            continue
        parts.append({"name": o.name or "part%d" % o.id,
                      "id": o.id, "triangles": t, "vertices": verts,
                      "faces": np.asarray(faces, dtype=np.int64),
                      "one_based": bool(faces.min() >= 1)})
    return parts


def _actor(pd, color, opacity=1.0, wireframe=False, line_width=1.0, user_data=None):
    import vtk
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(pd)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(opacity)
    if wireframe:
        actor.GetProperty().SetRepresentationToWireframe()
    actor.GetProperty().SetLineWidth(line_width)
    if user_data is not None:
        actor.SetProperty("part_id", user_data) if hasattr(actor, "SetProperty") else None
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


def edges_actor(pd, color=(0.15, 0.15, 0.18), line_width=1.0):
    """网格边线 actor（vtkExtractEdges）。"""
    import vtk
    edges = vtk.vtkExtractEdges()
    edges.SetInputData(pd)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(edges.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetRepresentationToWireframe()
    actor.GetProperty().SetLineWidth(line_width)
    return actor


def axes_actor(length=1.0):
    """全局坐标轴（RGB = XYZ）。"""
    import vtk
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(length, length, length)
    axes.SetShaftType(0)
    return axes


def orientation_marker_widget(interactor, size_frac=0.16):
    """左上角方向指示器（对齐 cab_vtk.orientation_marker_widget）。"""
    import vtk
    marker = vtk.vtkOrientationMarkerWidget()
    marker.SetOrientationMarker(axes_actor(1.0))
    marker.SetInteractor(interactor)
    marker.SetViewport(0.0, 0.85, size_frac, 1.0)
    marker.SetEnabled(1)
    marker.InteractOff()
    return marker


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
    bg = (0.08, 0.09, 0.12)
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


def build_scene_actors(sim, scene_obj, fallback_palette=True):
    """按场景构建 actors：PartDisplayer 颜色/透明度 + 边线。"""
    parts = part_meshes(sim)
    if not parts:
        return [], None
    om = sim.objmap
    colors = _colors()
    actors = []
    edges = []
    disp = scene_displayers(sim, scene_obj)
    for idx, p in enumerate(parts):
        color = colors[idx % len(colors)]
        opacity = 1.0
        use_disp = False
        if disp:
            d0 = disp[0]
            dc = d0.dict.get("DisplayerColor")
            if d0.dict.get("UseDisplayerColor") and dc:
                color = tuple(float(x) for x in dc[:3])
                use_disp = True
            opacity = float(d0.dict.get("Opacity", 1.0))
        pd = mesh_polydata(p["vertices"], p["faces"], one_based=p["one_based"])
        actors.append(("part:%s" % p["name"], p["name"], p["id"], _actor(pd, color, opacity=opacity)))
        if disp and disp[0].dict.get("Mesh"):
            mc = disp[0].dict.get("MeshColor") or (0.0, 0.0, 0.0)
            edges.append(("edges:%s" % p["name"], p["name"], p["id"],
                          edges_actor(pd, color=tuple(float(x) for x in mc[:3]),
                                      line_width=float(disp[0].dict.get("LineWidth", 1.0)))))
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


def render_offscreen_png(actors, out_path, size=(1200, 900), background=(0.08, 0.09, 0.12)):
    """离屏渲染 actors → PNG（测试/验证）。"""
    import vtk
    ren = vtk.vtkRenderer()
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

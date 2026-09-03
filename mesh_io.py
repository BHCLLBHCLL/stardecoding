# -*- coding: utf-8 -*-
"""STL/OBJ 读写与顶点变换（不依赖 CAD 内核）。"""

import os
import struct


def read_surface(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        return read_obj(path)
    if ext in (".ccm", ".ccmg"):
        mesh = read_ccm(path)
        return mesh["vertices"], mesh["faces"]
    return read_stl(path)


def read_ccm(path):
    """CCM / CCMG → 顶点 + 边界三角面（以及体网格摘要）。"""
    from ccm_io import read_ccm as _read
    return _read(path)


def read_obj(path):
    vertices = []
    faces = []
    with open(path, "r", encoding="ascii", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                idx = []
                for p in parts[1:4]:
                    idx.append(int(p.split("/")[0]) - 1)
                faces.append(idx)
    return vertices, faces


def read_stl(path):
    """返回 (vertices[N][3], faces[M][3])，0 基索引，重合顶点已合并。"""
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) >= 84 and not raw[:5].lower().startswith(b"solid"):
        return _read_binary_stl(raw)
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        return _read_binary_stl(raw)
    if "facet" in text.lower():
        return _read_ascii_stl(text)
    return _read_binary_stl(raw)


def _read_ascii_stl(text):
    verts = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 4 and parts[0].lower() == "vertex":
            verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return _weld(verts)


def _read_binary_stl(raw):
    if len(raw) < 84:
        raise ValueError("STL 太短")
    n = struct.unpack_from("<I", raw, 80)[0]
    need = 84 + n * 50
    if len(raw) < need:
        raise ValueError("STL 面数与文件长度不符")
    verts = []
    off = 84
    for _ in range(n):
        rec = struct.unpack_from("<12fH", raw, off)
        verts.append(rec[3:6])
        verts.append(rec[6:9])
        verts.append(rec[9:12])
        off += 50
    return _weld(verts)


def _weld(flat_verts, ndigits=7):
    key_to_i = {}
    vertices = []
    faces = []
    tri = []
    for p in flat_verts:
        key = (round(p[0], ndigits), round(p[1], ndigits), round(p[2], ndigits))
        if key not in key_to_i:
            key_to_i[key] = len(vertices)
            vertices.append([float(p[0]), float(p[1]), float(p[2])])
        tri.append(key_to_i[key])
        if len(tri) == 3:
            faces.append(tri)
            tri = []
    return vertices, faces


def write_ascii_stl(path, vertices, faces, name="sim"):
    with open(path, "w", encoding="ascii") as fh:
        fh.write("solid %s\n" % name)
        for tri in faces:
            a, b, c = (vertices[i] for i in tri)
            fh.write("  facet normal 0 0 0\n    outer loop\n")
            for p in (a, b, c):
                fh.write("      vertex %.9g %.9g %.9g\n" % (p[0], p[1], p[2]))
            fh.write("    endloop\n  endfacet\n")
        fh.write("endsolid %s\n" % name)
    return path


def write_binary_stl(path, vertices, faces, name="sim"):
    """二进制 STL 写出（C6 格式族；与 read_stl 二进制分支对称）。"""
    n = len(faces)
    header = b"solid " + bytes(name, "ascii", "replace")[:70] + b"\x00" * 4
    header = (header + b"\x00" * 80)[:80]
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(struct.pack("<I", n))
        for tri in faces:
            a, b, c = (vertices[i] for i in tri)
            fh.write(struct.pack(
                "<12fH",
                0.0, 0.0, 0.0,
                a[0], a[1], a[2],
                b[0], b[1], b[2],
                c[0], c[1], c[2],
                0))
    return path


def write_obj(path, vertices, faces, name="sim"):
    """OBJ 写出（C6 格式族；与 read_obj 对称，仅三角面）。"""
    with open(path, "w", encoding="ascii") as fh:
        fh.write("# %s\n" % name)
        fh.write("o %s\n" % name)
        for p in vertices:
            fh.write("v %.9g %.9g %.9g\n" % (p[0], p[1], p[2]))
        for tri in faces:
            fh.write("f %d %d %d\n" % (tri[0] + 1, tri[1] + 1, tri[2] + 1))
    return path


def write_surface(path, vertices, faces, name="sim", binary_stl=False):
    """按扩展名分派写出（.stl→ascii/binary、.obj→OBJ），与 read_surface 对称。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        return write_obj(path, vertices, faces, name)
    if ext in (".stl", ".stla", ".stlb"):
        if binary_stl or ext == ".stlb":
            return write_binary_stl(path, vertices, faces, name)
        return write_ascii_stl(path, vertices, faces, name)
    raise ValueError("不支持的写出格式: %s（支持 .stl/.obj）" % ext)


def cube_mesh(size=1.0):
    s = float(size)
    v = [
        [0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0],
        [0, 0, s], [s, 0, s], [s, s, s], [0, s, s],
    ]
    f = [
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1],
        [2, 6, 7], [2, 7, 3],
        [0, 3, 7], [0, 7, 4],
        [1, 5, 6], [1, 6, 2],
    ]
    return v, f


def _subset_part_triangles(p, cad_ids=None):
    faces = [list(map(int, row)) for row in p["faces"]]
    verts = [list(map(float, row)) for row in p["vertices"]]
    ft = p.get("face_types")
    if cad_ids and ft is not None and len(ft) == len(faces):
        want = set(int(x) for x in cad_ids)
        faces = [tri for tri, t in zip(faces, ft) if int(t) in want]
    return verts, faces


def export_part_stl(sim, part_id, path):
    """按 Part 导出 STL；找不到分块网格则失败。"""
    from star_gui_vtk import part_meshes
    for p in part_meshes(sim):
        if p.get("id") == part_id:
            verts, faces = _subset_part_triangles(p)
            return write_ascii_stl(path, verts, faces, p.get("name") or "part")
    raise RuntimeError("没有 id=%s 的分块网格" % part_id)


def export_scene_stl(sim, scene, path):
    """按场景显示器 Parts / FaceTypes 子集导出合并 STL。"""
    from star_gui_model import collector_sources, owning_mesh_part, part_surface_cad_ids
    from star_gui_vtk import mesh_bundle_for_part, part_meshes, representation_source_id, scene_displayers
    parts = {p["id"]: p for p in part_meshes(sim)}
    all_v = []
    all_f = []
    base = 0
    for d in scene_displayers(sim, scene):
        src_id = representation_source_id(sim, d)
        for src in collector_sources(sim, d):
            part = owning_mesh_part(sim, src)
            if part is None:
                continue
            p = mesh_bundle_for_part(sim, part, src_id, tess_by_id=parts) or parts.get(part.id)
            if p is None:
                continue
            cad = part_surface_cad_ids(sim, src)
            verts, faces = _subset_part_triangles(p, cad or None)
            if not faces:
                continue
            all_v.extend(verts)
            all_f.extend([[base + a, base + b, base + c] for a, b, c in faces])
            base += len(verts)
    if not all_f:
        raise RuntimeError("场景没有可导出的表面")
    return write_ascii_stl(path, all_v, all_f, (scene.name if scene is not None else None) or "scene")


def transform_vertices(vertices, translate=(0, 0, 0), scale=(1, 1, 1)):
    tx, ty, tz = translate
    sx, sy, sz = scale
    out = []
    for p in vertices:
        out.append([p[0] * sx + tx, p[1] * sy + ty, p[2] * sz + tz])
    return out

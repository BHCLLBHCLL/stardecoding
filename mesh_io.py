# -*- coding: utf-8 -*-
"""STL/OBJ 读写与顶点变换（不依赖 CAD 内核）。"""

import os
import struct


def read_surface(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        return read_obj(path)
    return read_stl(path)


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


def export_part_stl(sim, part_id, path):
    """按 Part 导出 STL；找不到分块网格则失败。"""
    from star_gui_vtk import part_meshes
    for p in part_meshes(sim):
        if p.get("id") == part_id:
            verts = [list(map(float, row)) for row in p["vertices"]]
            faces = [list(map(int, row)) for row in p["faces"]]
            return write_ascii_stl(path, verts, faces, p.get("name") or "part")
    raise RuntimeError("没有 id=%s 的分块网格" % part_id)


def transform_vertices(vertices, translate=(0, 0, 0), scale=(1, 1, 1)):
    tx, ty, tz = translate
    sx, sy, sz = scale
    out = []
    for p in vertices:
        out.append([p[0] * sx + tx, p[1] * sy + ty, p[2] * sz + tz])
    return out

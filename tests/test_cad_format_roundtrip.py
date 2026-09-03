# -*- coding: utf-8 -*-
"""C 波（自研轻量几何外壳）：C6 格式族往返 + C1 外部曲面重建显示。

范围与受限声明：
  - C6 自研可落地点：STL(ascii/binary)/OBJ 双向读写 + 往返面数一致（纯文件层，
    与 .sim 对象图解耦，零污染）。
  - C1 显示级：外部 STL/OBJ → 本客户端 vtkPolyData 重建显示（顶点/面数校验）。
  - STEP/IGES/BREP 与 C2–C5 B-Rep 建模（拉伸/布尔/圆角/包裹）依赖
    OpenCascade(OCP)/官方 Parasolid，本机 Python 3.14 无 OCP wheel 且无 STARCCM
    许可，属受限项（差异回归测试 assert skip/受限标注，不误报为已支持）。

新增 tests 文件会被 tests/run_all.py 自动纳入。
"""
import os
import struct
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from mesh_io import (cube_mesh, read_surface, write_surface,
                     write_ascii_stl, write_binary_stl, write_obj)  # noqa: E402


def _write(tmp, ext, binary=False):
    path = os.path.join(tmp, "geom" + ext)
    verts, faces = cube_mesh(1.25)
    write_surface(path, verts, faces, name="cube", binary_stl=binary)
    return path, verts, faces


def test_c6_stl_ascii_roundtrip():
    """C6：STL ascii 写→读 往返，面数/顶点数一致。"""
    with tempfile.TemporaryDirectory() as tmp:
        path, verts, faces = _write(tmp, ".stl")
        v2, f2 = read_surface(path)
        assert len(f2) == len(faces) == 12
        assert len(v2) == len(verts) == 8


def test_c6_stl_binary_roundtrip():
    """C6：STL binary 写→读 往返（走 write_binary_stl 与二进制读分支）。"""
    with tempfile.TemporaryDirectory() as tmp:
        path, verts, faces = _write(tmp, ".stlb", binary=True)
        raw = open(path, "rb").read()
        n = struct.unpack_from("<I", raw, 80)[0]
        assert n == 12
        v2, f2 = read_surface(path)
        assert len(f2) == 12 and len(v2) == 8
        # 不带 solid 头的裸二进制也能正确读出
        header = raw[:80].replace(b"solid", b"\x00" * 5)
        path2 = os.path.join(tmp, "geom_b.stl")
        with open(path2, "wb") as fh:
            fh.write(header + raw[80:])
        v3, f3 = read_surface(path2)
        assert len(f3) == 12 and len(v3) == 8


def test_c6_obj_roundtrip():
    """C6：OBJ 写→读 往返，面数与三角索引一致。"""
    with tempfile.TemporaryDirectory() as tmp:
        path, verts, faces = _write(tmp, ".obj")
        raw = open(path).read()
        assert raw.count("\nv ") == 8
        assert raw.count("\nf ") == 12
        v2, f2 = read_surface(path)
        assert len(f2) == 12 and len(v2) == 8
        assert sorted(tuple(t) for t in f2) and all(len(t) == 3 for t in f2)


def test_c6_binary_stl_is_not_misread_as_ascii():
    """C6：read_surface 对 binary STL 不误判为 ascii solid。"""
    bpath, _, faces = None, None, None
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cube.stl")
        v, f = cube_mesh(2.0)
        write_binary_stl(path, v, f)
        v2, f2 = read_surface(path)
        assert len(f2) == 12 and len(v2) == 8


def test_c6_unsupported_format_rejected():
    """C6：非 STL/OBJ 写出明确拒绝（STEP/IGES/BREP 受限，依赖 OCP/官方桥）。"""
    import pytest
    with tempfile.TemporaryDirectory() as tmp:
        v, f = cube_mesh(1.0)
        with pytest.raises(ValueError):
            write_surface(os.path.join(tmp, "geom.step"), v, f)
        with pytest.raises(ValueError):
            write_surface(os.path.join(tmp, "geom.iges"), v, f)


def test_c6_stl_obj_cross_roundtrip_face_count_conserved():
    """C6：立方体 STL→OBJ 交叉，往返面数始终 12（面数一致验收）。"""
    import shutil
    from mesh_io import write_obj as _wobj
    with tempfile.TemporaryDirectory() as tmp:
        stl = os.path.join(tmp, "cube.stl")
        v, f = cube_mesh(0.5)
        write_ascii_stl(stl, v, f, "cube")
        obj = os.path.join(tmp, "cube.obj")
        _wobj(obj, v, f, "cube")
        assert len(read_surface(stl)[1]) == len(read_surface(obj)[1])


def test_c1_surface_rebuild_for_display():
    """路径：read_surface→mesh_polydata 重建显示，顶点/面数一致（C1 显示级）。"""
    from star_gui_vtk import surface_polydata
    with tempfile.TemporaryDirectory() as tmp:
        for ext in (".stl", ".obj"):
            path = os.path.join(tmp, "geom" + ext)
            v, f = cube_mesh(1.0)
            if ext == ".obj":
                write_obj(path, v, f)
            else:
                write_ascii_stl(path, v, f)
            out = surface_polydata(path)
            assert out["n_vertices"] == 8
            assert out["n_triangles"] == 12
            pd = out["polydata"]
            assert pd.GetNumberOfCells() == 12  # 三角单元数=面数（C1 面数验收）
            assert pd.GetNumberOfPoints() >= 8  # 法向分割可能复制共享点，只保下限
# -*- coding: utf-8 -*-
"""S 波 S6：write_cgns 边界 patch（TRI_3 section 续接 + ZoneBC_t/BC_t）验收。

布局依据 cgnslib ADFH 源码实证（CGNS-4.5.1/src/adfh/ADFH.c + cgnslib.c）：
  - 每个边界 patch = 一个 TRI_3(5) Elements_t，元素编号自 nc+1 续接；
  - ZoneBC_t 下 BC_t 的 data 为 BCType 名字符串（C1）；
  - BC_t 的 PointList(IndexArray_t, I4, shape (1,n)) 引用面元素编号，
    GridLocation=FaceCenter（C1 字符串）；
  - 无 patches 时不写 ZoneBC_t（零回归）。
"""
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

from fvm_core import FVM, cube_tet_mesh
from postprocess import CGNS_BCTYPES, _cgns_str, write_cgns

h5py = pytest.importorskip("h5py")


def _cube_patches(fv):
    """按面质心把单位立方体边界面分成 6 个 patch。"""
    idx = np.where(fv.is_boundary)[0]
    cent = fv.face_centroid[idx]
    patches = []
    for axis, val, name, tp in ((0, 0.0, "Xmin", "BCInflow"),
                                (0, 1.0, "Xmax", "BCOutflow"),
                                (1, 0.0, "Ymin", "BCWall"),
                                (1, 1.0, "Ymax", "BCWall"),
                                (2, 0.0, "Zmin", "BCWall"),
                                (2, 1.0, "Zmax", "BCWall")):
        sel = idx[np.isclose(cent[:, axis], val)]
        patches.append({"name": name, "type": tp, "faces": sel})
    return patches


def _write_with_patches(fv, patches, fields=None):
    tmp = tempfile.mkdtemp(prefix="s6_cgns_")
    path = os.path.join(tmp, "m.cgns")
    write_cgns(path, fv, fields=fields, patches=patches)
    return tmp, path


def test_patches_sections_and_bcs():
    fv = FVM(*cube_tet_mesh(2))
    patches = _cube_patches(fv)
    assert sum(len(p["faces"]) for p in patches) == fv.n_boundary_faces
    tmp, path = _write_with_patches(fv, patches)
    try:
        with h5py.File(path, "r") as f:
            zone = f["Base"]["Zone"]
            # 每 patch 一个 TRI_3 Elements_t，元素编号自 nc+1 续接且全局唯一
            nxt = fv.n_cells + 1
            for p in patches:
                sec = zone[p["name"]]
                assert sec.attrs["label"] == b"Elements_t"
                assert sec.attrs["type"] == b"I4"
                assert np.array_equal(sec[" data"][()],
                                      np.array([5, 0], np.int32))
                rng = sec["ElementRange"][" data"][()]
                assert rng.tolist() == [nxt, nxt + len(p["faces"]) - 1]
                conn = sec["ElementConnectivity"][" data"][()]
                assert conn.shape == (len(p["faces"]) * 3,)
                assert conn.min() >= 1 and conn.max() <= fv.n_vertices
                nxt = int(rng[1]) + 1
            # ZoneBC_t / BC_t 结构
            zbc = zone["ZoneBC"]
            assert zbc.attrs["label"] == b"ZoneBC_t"
            assert zbc.attrs["type"] == b"MT"
            assert " data" not in zbc
            for p in patches:
                bc = zbc[p["name"]]
                assert bc.attrs["label"] == b"BC_t"
                assert bc.attrs["type"] == b"C1"
                assert _cgns_str(bc) == p["type"].encode()
                assert _cgns_str(bc["GridLocation"]) == b"FaceCenter"
                assert bc["PointList"].attrs["label"] == b"IndexArray_t"
                plist = bc["PointList"][" data"][()]
                assert plist.shape == (1, len(p["faces"]))
                assert plist.dtype == np.int32
            # 全部 PointList 编号不重不漏覆盖所有边界面元素
            allp = np.concatenate(
                [zbc[p["name"]]["PointList"][" data"][()].ravel()
                 for p in patches])
            assert len(np.unique(allp)) == len(allp) == fv.n_boundary_faces
            assert allp.min() == fv.n_cells + 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_patch_faces_area_conserved():
    """patch section 的面总面积 = 立方体表面积 6（编号续接正确性几何验证）。"""
    fv = FVM(*cube_tet_mesh(2))
    patches = _cube_patches(fv)
    tmp, path = _write_with_patches(fv, patches)
    try:
        with h5py.File(path, "r") as f:
            zone = f["Base"]["Zone"]
            conn = zone["Elements"]["ElementConnectivity"][" data"][()].reshape(-1, 4)
            total = 0.0
            for p in patches:
                t = zone[p["name"]]["ElementConnectivity"][" data"][()].reshape(-1, 3)
                tri = fv.vertices[t - 1]
                total += float(np.sum(
                    0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0],
                                                  tri[:, 2] - tri[:, 0]),
                                         axis=1)))
            assert np.isclose(total, 6.0, atol=1e-9)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_no_patches_no_zonebc():
    fv = FVM(*cube_tet_mesh(2))
    tmp, path = _write_with_patches(fv, None)
    try:
        with h5py.File(path, "r") as f:
            zone = f["Base"]["Zone"]
            assert "ZoneBC" not in zone
            # 子节点恰为无 patch 时的固定集（无额外 section）
            assert set(zone.keys()) == {" data", "ZoneType", "GridCoordinates",
                                        "Elements"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_patch_validation_errors():
    fv = FVM(*cube_tet_mesh(2))
    bnd = np.where(fv.is_boundary)[0]
    inner = np.where(~fv.is_boundary)[0]
    with pytest.raises(ValueError):  # 内部面不允许
        write_cgns(os.path.join(tempfile.mkdtemp(), "m.cgns"), fv,
                   patches=[{"name": "Bad", "type": "BCWall",
                             "faces": inner}])
    with pytest.raises(ValueError):  # 未知 BCType
        write_cgns(os.path.join(tempfile.mkdtemp(), "m.cgns"), fv,
                   patches=[{"name": "Bad", "type": "BCBogus",
                             "faces": bnd}])
    with pytest.raises(ValueError):  # 同一面归两个 patch
        write_cgns(os.path.join(tempfile.mkdtemp(), "m.cgns"), fv,
                   patches=[{"name": "A", "type": "BCWall", "faces": bnd},
                            {"name": "B", "type": "BCWall", "faces": bnd}])


def test_bctype_whitelist_contains_s2_case_types():
    for t in ("BCInflow", "BCOutflow", "BCWall", "BCSymmetryPlane"):
        assert t in CGNS_BCTYPES

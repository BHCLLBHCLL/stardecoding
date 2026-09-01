# -*- coding: utf-8 -*-
"""读取 legacy CCM（.ccm / .ccmg）：ctypes 绑定 ccmio.dll，对齐 gph2ccm。

不依赖 STAR-CCM+ Java API。动态库查找顺序：
1. GPH2CCM_CCMIO_DLL / CCMIO_DLL
2. STARCCM_HOME 下 star/lib/.../ccmio.dll
3. 本机 Siemens / D:\\training\\starccm 安装树
4. gph2ccm.ccmio.find_ccmio_library()
"""

import glob
import os
import sys


GPH2CCM_ROOTS = (
    os.environ.get("GPH2CCM_ROOT") or "",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "cgns", "gph2ccm"),
    r"D:\training\cgns\gph2ccm",
)


def triangulate_face_stream(stream, one_based=True):
    """CCM 面流 [n, v1..vn, ...] → 0 基三角形列表。"""
    faces = []
    data = [int(x) for x in stream]
    i, n = 0, len(data)
    shift = 1 if one_based else 0
    while i < n:
        nv = data[i]
        i += 1
        if nv < 0 or i + nv > n:
            break
        vids = [data[i + k] - shift for k in range(nv)]
        i += nv
        if nv < 3:
            continue
        for k in range(1, nv - 1):
            faces.append([vids[0], vids[k], vids[k + 1]])
    return faces


def _gph2ccm_root():
    for root in GPH2CCM_ROOTS:
        if not root:
            continue
        root = os.path.abspath(root)
        if os.path.isdir(os.path.join(root, "gph2ccm")):
            return root
    return None


def _glob_ccmio():
    patterns = []
    home = os.environ.get("STARCCM_HOME") or ""
    if home:
        patterns.append(os.path.join(home, "star", "lib", "win64", "*", "lib", "ccmio.dll"))
        patterns.append(os.path.join(home, "star", "lib", "*", "*", "lib", "ccmio.dll"))
    for base in (r"D:\training\starccm", r"C:\Program Files\Siemens",
                 r"D:\Program Files\Siemens", r"C:\Siemens"):
        patterns.append(os.path.join(base, "*", "STAR-CCM+*", "star", "lib", "win64", "*", "lib", "ccmio.dll"))
        patterns.append(os.path.join(base, "STAR-CCM+*", "star", "lib", "win64", "*", "lib", "ccmio.dll"))
    hits = []
    for pat in patterns:
        hits.extend(glob.glob(pat))
    hits = [p for p in hits if os.path.isfile(p)]
    hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return hits[0] if hits else None


def _ensure_dll_env():
    for key in ("GPH2CCM_CCMIO_DLL", "CCMIO_DLL"):
        p = os.environ.get(key)
        if p and os.path.isfile(p):
            os.environ["GPH2CCM_CCMIO_DLL"] = p
            return p
    hit = _glob_ccmio()
    if hit:
        os.environ["GPH2CCM_CCMIO_DLL"] = hit
        parent = os.path.dirname(hit)
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(parent)
            except (OSError, FileNotFoundError):
                pass
        return hit
    return None


def _import_ccmio():
    _ensure_dll_env()
    root = _gph2ccm_root()
    if root and root not in sys.path:
        sys.path.insert(0, root)
    from gph2ccm.ccmio import (  # noqa: WPS433
        CCMIO, CCMIOError,
        K_CCMIO_BOUNDARY_FACES, K_CCMIO_BOUNDARY_REGION, K_CCMIO_CELL_TYPE,
        K_CCMIO_CELLS, K_CCMIO_INTERNAL_FACES, K_CCMIO_PROCESSOR,
        K_CCMIO_TOPOLOGY, K_CCMIO_VERTICES,
    )
    return {
        "CCMIO": CCMIO, "CCMIOError": CCMIOError,
        "K_CCMIO_BOUNDARY_FACES": K_CCMIO_BOUNDARY_FACES,
        "K_CCMIO_BOUNDARY_REGION": K_CCMIO_BOUNDARY_REGION,
        "K_CCMIO_CELL_TYPE": K_CCMIO_CELL_TYPE,
        "K_CCMIO_CELLS": K_CCMIO_CELLS,
        "K_CCMIO_INTERNAL_FACES": K_CCMIO_INTERNAL_FACES,
        "K_CCMIO_PROCESSOR": K_CCMIO_PROCESSOR,
        "K_CCMIO_TOPOLOGY": K_CCMIO_TOPOLOGY,
        "K_CCMIO_VERTICES": K_CCMIO_VERTICES,
    }


def ccmio_available():
    """gph2ccm + ccmio.dll 是否可用。"""
    try:
        mods = _import_ccmio()
        mods["CCMIO"]()
        return True
    except Exception:
        return False


def ccmio_unavailable_reason():
    if not _gph2ccm_root():
        return "找不到 gph2ccm（设置 GPH2CCM_ROOT 或使用 D:\\training\\cgns\\gph2ccm）"
    if not (_ensure_dll_env() or os.environ.get("GPH2CCM_CCMIO_DLL")):
        return "找不到 ccmio.dll（设置 GPH2CCM_CCMIO_DLL 或安装 STAR-CCM+）"
    try:
        _import_ccmio()["CCMIO"]()
    except Exception as exc:
        return str(exc)
    return ""


def _first_entity(ccmio, parent, etype):
    for node in ccmio.iter_entities(parent, etype):
        return node
    return None


def _optstr(ccmio, node, name):
    try:
        return ccmio.read_optstr(node, name)
    except Exception:
        return ""


def read_ccm(path):
    """读 CCM 网格：顶点（已乘 scale）+ 边界三角面（0 基）。"""
    if not os.path.isfile(path):
        raise IOError("CCM 文件不存在: %s" % path)
    mods = _import_ccmio()
    CCMIO = mods["CCMIO"]
    K_BF = mods["K_CCMIO_BOUNDARY_FACES"]
    K_BR = mods["K_CCMIO_BOUNDARY_REGION"]
    K_CT = mods["K_CCMIO_CELL_TYPE"]
    K_CELLS = mods["K_CCMIO_CELLS"]
    K_IF = mods["K_CCMIO_INTERNAL_FACES"]
    K_PROC = mods["K_CCMIO_PROCESSOR"]
    K_TOPO = mods["K_CCMIO_TOPOLOGY"]
    K_VERT = mods["K_CCMIO_VERTICES"]

    ccmio = CCMIO()
    root = ccmio.open_file_readonly(path)
    try:
        version = ccmio.get_version(root)
        verts_node = _first_entity(ccmio, root, K_VERT)
        topo = _first_entity(ccmio, root, K_TOPO)
        try:
            state, problem = ccmio.get_state(root)
            proc = ccmio.next_entity(state, K_PROC, 0)
            if proc is not None:
                pv, ptopo, _, _ = ccmio.read_processor(proc)
                verts_node, topo = pv, ptopo
        except Exception:
            problem = None

        if verts_node is None:
            raise RuntimeError("CCM 没有 Vertices")
        if topo is None:
            raise RuntimeError("CCM 没有 Topology")

        dims, scale, _map, coords = ccmio.read_vertices(verts_node)
        scale = float(scale) if scale else 1.0
        verts = [[float(p[0]) * scale, float(p[1]) * scale, float(p[2]) * scale]
                 for p in coords]

        n_cells = 0
        cells_node = _first_entity(ccmio, topo, K_CELLS)
        if cells_node is not None:
            _cmap, cell_types = ccmio.read_cells(cells_node)
            n_cells = int(cell_types.size)

        n_if = 0
        iface = _first_entity(ccmio, topo, K_IF)
        if iface is not None:
            n_if, _ = ccmio.entity_size(iface)
            n_if = int(n_if)

        faces = []
        regions = []
        for node in ccmio.iter_entities(topo, K_BF):
            rid = ccmio.entity_index(node)
            n_bf, _ = ccmio.entity_size(node)
            n_bf = int(n_bf)
            label = ""
            btype = ""
            if problem is not None:
                try:
                    br = ccmio.get_entity(problem, K_BR, rid)
                    label = _optstr(ccmio, br, "Label")
                    btype = _optstr(ccmio, br, "BoundaryType")
                except Exception:
                    pass
            if n_bf:
                _bmap, bstream = ccmio.read_faces(node, K_BF)
                faces.extend(triangulate_face_stream(bstream, one_based=True))
            regions.append({"id": int(rid), "label": label or ("Boundary-%d" % rid),
                            "type": btype, "n_faces": n_bf})

        cell_types_info = {}
        if problem is not None:
            for node in ccmio.iter_entities(problem, K_CT):
                cid = ccmio.entity_index(node)
                cell_types_info[int(cid)] = {
                    "label": _optstr(ccmio, node, "Label"),
                    "material": _optstr(ccmio, node, "MaterialType"),
                }

        if not faces and iface is not None and n_if:
            _imap, istream = ccmio.read_faces(iface, K_IF)
            faces = triangulate_face_stream(istream, one_based=True)

        return {
            "path": os.path.abspath(path),
            "version": int(version),
            "scale": scale,
            "dims": int(dims),
            "vertices": verts,
            "faces": faces,
            "n_vertices": len(verts),
            "n_cells": n_cells,
            "n_internal_faces": n_if,
            "n_boundary_faces": sum(r["n_faces"] for r in regions),
            "regions": regions,
            "cell_types": cell_types_info,
        }
    finally:
        ccmio.close_file(root)

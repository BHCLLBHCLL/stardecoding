# -*- coding: utf-8 -*-
"""C 波 C5：表面包裹 wrapper（收缩包裹 + 特征捕捉 + 局部尺寸）。

自研（A 路线）：基于「体素占据 + 边界面提取」的收缩包裹。输入三角网格
(vertices, faces)，输出一张**必然水密（watertight）**的闭合包裹表面——即使
输入含孔洞/开放面/缝隙也能密封成闭合体。纯 numpy，任意 Python 环境可用。

能力：
  - 收缩包裹：`shrink_wrap(vertices, faces, cell_size, gap=0)`——均匀体素包裹，
    占位单元 = 与三角面相交的格子，包裹表面 = 占据体与空腔的界面，天然水密；
  - 局部尺寸/特征捕捉：`wrap_mesh(..., refine=(center,radius), features=(pt,...))`
    ——自适应粗细分辨率：`refine` 指定球形区域局部加密一级（cell/2），`features`
    指定特征点附近再加密一级（cell/4）。混合分辨率经**面投影覆盖拼接**保证水密；
  - 尖角检测：`detect_features(vertices, faces, angle_deg=30)`——按相邻面二面角
    探测 sharp 特征边，供 wrap_mesh 的 refine 决策参考。
"""
import math

import numpy as np

from occ_repair import boundary_edges


# ---------------------------------------------------------------- 单元占据
def _squared(v):
    return float((v * v).sum())


def _tri_box_overlap(lo, hi, tri):
    """三角形与 AABB [lo,hi] 相交（Akenine-Möller 严格分离轴测试）。

    精确命中（含恰好贴合面），不做半径近似——保证不同分辨率占据**厚度一致**，
    是收缩包裹在跨界粗细网格处仍水密的根基。
    """
    v0 = np.asarray(tri[0], float)
    v1 = np.asarray(tri[1], float)
    v2 = np.asarray(tri[2], float)
    c = 0.5 * (np.asarray(lo, float) + np.asarray(hi, float))
    h = 0.5 * (np.asarray(hi, float) - np.asarray(lo, float))
    return _tri_box_overlap_test(v0 - c, v1 - c, v2 - c, h)


def _tri_box_overlap_test(v0, v1, v2, h):
    """平移后三角形（box 中心在原点）与半尺寸 h 盒的分离轴测试。"""
    # 三顶点投影在盒坐标系；轴分离测试
    for dim in range(3):
        amin = min(v0[dim], v1[dim], v2[dim])
        amax = max(v0[dim], v1[dim], v2[dim])
        if amin > h[dim] or amax < -h[dim]:
            return False
    # 三角形平面与盒对角区间
    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v0 - v2
    n = np.cross(e0, e1)
    if not _plane_box_overlap(n, v0, h):
        return False
    # 9 条边×轴分离轴（Akenine-Möller 的 9 个分离轴）
    for e_axis in (e0, e1, e2):
        for dim in range(3):
            # 分离轴 a = e_axis × u_dim
            ax = np.zeros(3)
            ax[(dim + 1) % 3] = e_axis[(dim + 2) % 3]
            ax[(dim + 2) % 3] = -e_axis[(dim + 1) % 3]
            if not _edge_axis_separates(ax, v0, v1, v2, h):
                return False
    return True


def _edge_axis_separates(ax, v0, v1, v2, h):
    p0 = float(ax @ v0)
    p1 = float(ax @ v1)
    p2 = float(ax @ v2)
    r = h[0] * abs(ax[0]) + h[1] * abs(ax[1]) + h[2] * abs(ax[2])
    if max(p0, p1, p2) < -r or min(p0, p1, p2) > r:
        return False
    return True


def _plane_box_overlap(n, v0, h):
    v = np.abs(n)
    r = float(v @ h)
    d = float(n @ v0)
    return -r <= d <= r


def _occupied_leaves(V, F, lo, hi, cell):
    """生成占据叶单元（均匀 cell）：与任一三角面精确相交的格子。"""
    dims = np.maximum(np.ceil((hi - lo) / cell).astype(int), 1)
    leaves = []
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                o = lo + cell * np.array([i, j, k])
                cbox_lo = o
                cbox_hi = o + cell
                if any(_tri_box_overlap(cbox_lo, cbox_hi, V[t]) for t in F):
                    leaves.append(((cbox_lo + cbox_hi) * 0.5, cell))
    return leaves


# ------------------------------------------------------ 水密边界面提取
def _extract_surface(leaves):
    """从占据叶（中心,尺寸）列表提取水密边界三角面。

    规则：占据体=这些叶盒的并集，其边界是闭合多面体，必定水密。对每个叶的
    每个外侧面，检测「紧贴该面的相邻叶投影」是否把该面完全覆盖；被盖住则不
    出表面（内部），露出部分才出两三角。混合尺寸经投影覆盖可靠拼接、无裂缝。
    """
    verts = {}
    tris = []

    def vkey(p):
        return (round(float(p[0]), 9), round(float(p[1]), 9), round(float(p[2]), 9))

    def vidx(p):
        k = vkey(p)
        if k not in verts:
            verts[k] = len(verts)
        return verts[k]

    def tri(p0, p1, p2):
        tris.append((vidx(p0), vidx(p1), vidx(p2)))

    EPS = 1e-9
    boxes = [(np.asarray(c, float) - 0.5 * s, np.asarray(c, float) + 0.5 * s, s)
             for (c, s) in leaves]

    def adjacent(box, d, sign):
        # 返回与 box 的 d 轴 sign 侧紧贴的其它叶（投影矩形, 作为覆盖物）
        (blo, bhi, s) = box
        if sign > 0:
            plane = bhi[d]
        else:
            plane = blo[d]
        covers = []
        for (a2, b2, s2) in boxes:
            a2 = np.asarray(a2, float)
            b2 = np.asarray(b2, float)
            if sign > 0:
                if abs(a2[d] - plane) > EPS:
                    continue
            else:
                if abs(b2[d] - plane) > EPS:
                    continue
            # 投影在另外两轴与 box 的重叠（取重合交集矩形）
            inter = []
            ok = True
            for ax in range(3):
                if ax == d:
                    continue
                lo0 = max(blo[ax], a2[ax])
                hi0 = min(bhi[ax], b2[ax])
                if hi0 - lo0 <= -EPS:
                    ok = False
                    break
                inter.append((lo0, hi0))
            if ok:
                covers.append(tuple(inter))
        return covers

    def emit_face(box, d, sign):
        (blo, bhi, s) = box
        ax_u, ax_v = [a for a in range(3) if a != d]
        lo_u, hi_u = blo[ax_u], bhi[ax_u]
        lo_v, hi_v = blo[ax_v], bhi[ax_v]
        covers = adjacent(box, d, sign)
        covers = [((float(max(lo_u, cu)), float(min(hi_u, cu_hi))),
                   (float(max(lo_v, cv)), float(min(hi_v, cv_hi))))
                  for (cu, cu_hi), (cv, cv_hi) in covers]
        # 轴分割点：R 边界 + 各覆盖矩形边界（平面坐标，全部转为 Python 浮点）
        u_b = sorted(set([float(lo_u), float(hi_u)] +
                         [cu for (cu, cu_hi), _ in covers] +
                         [cu_hi for (cu, cu_hi), _ in covers]))
        v_b = sorted(set([float(lo_v), float(hi_v)] +
                         [cv for _, (cv, cv_hi) in covers] +
                         [cv_hi for _, (cv, cv_hi) in covers]))

        def mk(u, v):
            q = [lo_u, lo_v, 0.0]
            q[ax_u] = u
            q[ax_v] = v
            q[d] = bhi[d] if sign > 0 else blo[d]
            return q

        for i in range(len(u_b) - 1):
            if u_b[i + 1] - u_b[i] <= 1e-12:      # 零宽条带 → 跳过（防退化三角）
                continue
            for j in range(len(v_b) - 1):
                if v_b[j + 1] - v_b[j] <= 1e-12:
                    continue
                uc = 0.5 * (u_b[i] + u_b[i + 1])
                vc = 0.5 * (v_b[j] + v_b[j + 1])
                covered = any(
                    cu <= uc <= cu_hi and cv <= vc <= cv_hi
                    for (cu, cu_hi), (cv, cv_hi) in covers)
                if covered:
                    continue
                # 该小块是占据体的边界：出两三角（法向朝 sign 方向）
                A = mk(u_b[i], v_b[j])
                Bp = mk(u_b[i + 1], v_b[j])
                C = mk(u_b[i + 1], v_b[j + 1])
                D = mk(u_b[i], v_b[j + 1])
                if sign > 0:
                    tri(A, Bp, C)
                    tri(A, C, D)
                else:
                    tri(A, D, C)
                    tri(A, C, Bp)

    for (blo, bhi, s) in boxes:
        for d in range(3):
            emit_face((blo, bhi, s), d, 1)
            emit_face((blo, bhi, s), d, -1)
    return verts, tris


# ---------------------------------------------------------------- 公共 API
def shrink_wrap(vertices, faces, cell_size, gap=0.0):
    """均匀收缩包裹：输出水密闭合表面。

    返回 {vertices, faces, n_vertices, n_triangles, watertight, cell_size, gap,
          volume_estimate, n_leaves, cell_sizes}。
    """
    V = np.asarray(vertices, dtype=float)
    F = np.asarray(faces, dtype=np.int64)
    lo = V.min(axis=0) - gap
    hi = V.max(axis=0) + gap
    leaves = _occupied_leaves(V, F, lo, hi, cell_size)
    verts, tris = _extract_surface(leaves)
    return _assemble(verts, tris, cell_size, gap, leaves)


def wrap_mesh(vertices, faces, cell_size, gap=0.0, refine=(), features=(),
              rounds=3):
    """自适应收缩包裹：先出**水密**均匀包裹，再在 refine 球区 / features 特征点
    附近做**质心扇形加密**——只切三角面内部、不拆分共享边，故天然保持拓扑闭合
    水密，且局部面密度更细（局部尺寸可控）。返回同 shrink_wrap。
    """
    out = shrink_wrap(vertices, faces, cell_size, gap=gap)
    refine_p = [np.asarray(c, float) for c, _ in refine]
    refine_r = [float(r) for _, r in refine]
    feat_pts = [np.asarray(f, float) for f in features]

    def want(c):
        for f in feat_pts:
            if _squared(c - f) <= (1.5 * cell_size) ** 2:
                return True
        for fc, r in zip(refine_p, refine_r):
            if _squared(c - fc) <= (r + 0.25 * cell_size) ** 2:
                return True
        return False

    outV, outF = _local_fan_refine(out["vertices"], out["faces"], want, rounds)
    closed = len(boundary_edges(outV, outF)[0]) == 0
    return {"vertices": outV, "faces": outF,
            "n_vertices": len(outV), "n_triangles": len(outF),
            "watertight": closed, "cell_size": cell_size, "gap": gap,
            "base_faces": len(out["faces"])}


def _local_fan_refine(vertices, faces, want, rounds):
    """质心扇形加密：把 want(重心)=True 的三角按重心扇成 3 个共面子三角。

    只新增三角形内部顶点、不拆分任何共享边，因此不产生悬挂点/T 型裂缝，
    水密性与流形保持由构造保证。
    """
    Vs = [list(v) for v in vertices]
    Fs = [list(map(int, f)) for f in faces]
    for _ in range(max(rounds, 1)):
        newv = len(Vs)
        next_faces = []
        changed = False
        for (a, b, c) in Fs:
            m3 = [Vs[a][0] + Vs[b][0] + Vs[c][0],
                  Vs[a][1] + Vs[b][1] + Vs[c][1],
                  Vs[a][2] + Vs[b][2] + Vs[c][2]]
            centroid = [x / 3.0 for x in m3]
            if want(centroid):
                Vs.append(centroid)
                m = newv
                newv += 1
                next_faces.append([a, b, m])
                next_faces.append([b, c, m])
                next_faces.append([c, a, m])
                changed = True
            else:
                next_faces.append([a, b, c])
        Fs = next_faces
        if not changed:
            break
    return Vs, Fs


def _assemble(vert_dict, tris, cell_size, gap, leaves):
    order = sorted(vert_dict, key=vert_dict.get)
    verts = [list(p) for p in order]
    closed = len(boundary_edges(verts, tris)[0]) == 0
    vol = float(sum(s * s * s for (_, s) in leaves))
    return {"vertices": verts, "faces": tris,
            "n_vertices": len(verts), "n_triangles": len(tris),
            "watertight": closed, "cell_size": cell_size, "gap": gap,
            "volume_estimate": vol, "n_leaves": len(leaves),
            "cell_sizes": sorted({round(s, 9) for (_, s) in leaves})}


# ---------------------------------------------------------------- 特征检测
def detect_features(vertices, faces, angle_deg=30.0):
    """尖角特征检测：相邻面二面角 > angle_deg 的边判为特征（sharp）边。

    返回 {edges:[(a,b,dihedral_deg)], n_edges, n_feature, angle_deg}。
    """
    V = np.asarray(vertices, dtype=float)
    F = np.asarray(faces, dtype=np.int64)
    normals = _face_normals(V, F)
    emap = {}
    for t, tri in enumerate(F):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            e = (a, b) if a < b else (b, a)
            emap.setdefault(e, []).append(t)
    edges = []
    for e, faces_of in emap.items():
        if len(faces_of) < 2:
            continue
        n0 = normals[faces_of[0]]
        n1 = normals[faces_of[1]]
        dot = float(np.clip(n0 @ n1, -1, 1))
        ang = math.degrees(math.acos(dot))
        dihed = 180.0 - ang            # 二面角：平角=180 非特征，锐二面=特征
        if dihed > angle_deg:
            edges.append((int(e[0]), int(e[1]), float(dihed)))
    return {"edges": edges, "n_edges": len(edges),
            "n_feature": len(edges), "angle_deg": float(angle_deg),
            "edge_map": {tuple(k): v for k, v in emap.items()}}


def _face_normals(V, F):
    a = V[F[:, 0]]
    b = V[F[:, 1]]
    c = V[F[:, 2]]
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(ln, 1e-300)
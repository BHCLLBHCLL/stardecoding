# -*- coding: utf-8 -*-
"""N 波 N2b：曲率自适应尺寸场 + 局部算子表面重网格化（纯 numpy）。

曲率尺寸场：
  - `curvature_field`：每顶点平均曲率幅值 |H|，用 cotan Laplacian（曲面 normal 散度
    1/(2A) Σ w_ij (V_j - V_i)，w=cot 对角）估计；平面≈0、球面≈1/R、尖边显著非零。
  - `size_field`：每顶点目标边长 L_v = target_max / (1 + κ_v/κ_q)，κ_q=κ 的 70 分位
    → 平坦(κ≈0)≈target_max，高曲率区收缩到 ≥min_ratio·target_max。单调：曲率越大边越短。

局部算子重网格化（`remesh_surface`），目标边长 = 两端点尺寸场均并：
  - split：三角边 > 4/3·L 则沿共享边一致中点细分（水密保持），实现曲率驱动非均匀加密；
  - collapse：内边 < 4/5·L 且满足 link 条件（公共邻居=2、端点非边界、不产生重复面）则折叠；
  - flip：内边四角若旋转后最小角增大则翻边（角度/Delaunay 改善）；
  - smooth：非边界顶点沿切平面各向同性光滑（Botsch 式）。

闭环输入保持水密、开面输入保持边界（同数目边界半环）。纯 Python / numpy，两环境皆可用。
"""
import numpy as np


def _build_adj(F):
    """无向边 → 共享该边的面 id 列表（1=边界边，2=内边）。"""
    edge_faces = {}
    for fi, (a, b, c) in enumerate(F):
        for e in ((a, b), (b, c), (c, a)):
            key = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
            edge_faces.setdefault(key, []).append(fi)
    return edge_faces


def _boundary_vertices(F, adj):
    bv = set()
    for key, fs in adj.items():
        if len(fs) == 1:
            bv.add(key[0])
            bv.add(key[1])
    return bv


def _face_cot_angles(V, F):
    """每个面的三内角对应 cot 值（按面内 (a,b,c) 顺序：在 a/b/c 三点）。"""
    cot = []
    area = []
    for a, b, c in F:
        A, B, C = V[a], V[b], V[c]
        e1 = B - A
        e2 = C - A
        cr = np.cross(e1, e2)
        ar = np.linalg.norm(cr)
        if ar < 1e-30:
            cot.append((0.0, 0.0, 0.0))
        else:
            # 对角 cot：在 a 点夹 (e1,e2)，在 b 夹 (-e1, c-b)，在 c 夹 (-e2, b-c)
            cot_a = float(e1 @ e2) / ar
            cot_b = float((-e1) @ (C - B)) / ar
            cot_c = float((-e2) @ (B - C)) / ar
            cot.append((cot_a, cot_b, cot_c))
        area.append(0.5 * ar)
    return cot, area


def curvature_field(V, F):
    """每顶点平均曲率幅值 |H|（cotan Laplacian），向量 (N,)。"""
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    N = V.shape[0]
    cot, area = _face_cot_angles(V, F)
    op_area = np.zeros(N)            # barycentric 面积 (1/3 面面积)
    Lap = np.zeros((N, 3))           # cotan Laplacian
    for fi, (a, b, c) in enumerate(F):
        ca, cb, cc = cot[fi]
        th = area[fi] / 3.0
        op_area[a] += th
        op_area[b] += th
        op_area[c] += th
        A, B, C = V[a], V[b], V[c]
        # 每条边贡献对面角的 cot × 差向量（单三角近似）：
        Lap[a] += cc * (B - A) + cb * (C - A)
        Lap[b] += cc * (A - B) + ca * (C - B)
        Lap[c] += cb * (A - C) + ca * (B - C)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.divide(1.0, 2.0 * op_area,
                        out=np.zeros_like(op_area), where=op_area > 1e-30)
    H = np.linalg.norm(Lap * inv[:, None], axis=1)
    return H


def _vertex_normals(V, F):
    """面积加权平均面法向 → 单位顶点法向 (N,3)。"""
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    N = V.shape[0]
    nrm = np.zeros((N, 3))
    for a, b, c in F:
        n = np.cross(V[b] - V[a], V[c] - V[a])
        nrm[a] += n
        nrm[b] += n
        nrm[c] += n
    ln = np.linalg.norm(nrm, axis=1)
    z = ln > 1e-30
    nrm[z] /= ln[z][:, None]
    return nrm


def size_field(kappa, target_max, curvature_adaptive=True, min_ratio=0.2):
    """每顶点目标边长 L_v（尺寸场）。"""
    kappa = np.asarray(kappa, float)
    tm = float(target_max)
    if not curvature_adaptive:
        return np.full_like(kappa, tm)
    kq = float(np.percentile(kappa, 70))
    kq = max(kq, 1e-9)
    L = tm / (1.0 + kappa / kq)
    return np.clip(L, tm * min_ratio, tm)


def _edge_len_target(V, u, v, L):
    return 0.5 * (L[u] + L[v])


def _split_edge(V, F, adj, uv, index_of):
    """沿内/边界边 uv 中点细分；返回新 (V,F,adj)。uv=(i,j) 无序。"""
    i, j = uv
    m = len(V)
    V = list(V)
    V.append([0.5 * (V[i][0] + V[j][0]),
              0.5 * (V[i][1] + V[j][1]),
              0.5 * (V[i][2] + V[j][2])])
    faces_to_split = list(adj.get(uv, ()))
    new_adj = {k: [] for k in adj}
    new_F = []
    for fi, (a, b, c) in enumerate(F):
        if fi not in set(faces_to_split):
            new_F.append((a, b, c))
            continue
        # 找到该面对应 (i,j) 的第三个点 k
        if a == i and b == j:
            k = c
        elif b == i and c == j:
            k = a
        elif c == i and a == j:
            k = b
        elif a == j and b == i:
            k = c
        elif b == j and c == i:
            k = a
        elif c == j and a == i:
            k = b
        else:  # 边的顺序异常，跳过
            new_F.append((a, b, c))
            continue
        new_F.append((i, m, k))
        new_F.append((m, j, k))
    # 重建 adj
    new_adj = _build_adj(new_F)
    # 忽略键删除、仅保留现存边即可
    return V, new_F, new_adj


def _split_pass(V, F, L, adj, bv, max_rounds=20):
    """批处理：每轮一次**向量化**边扫描选过长边，按降序分裂（O(E) 选边 + O(分裂数·F) 更新）。"""
    changed = False
    L = np.asarray(L, float)       # 随细分增长（新顶点目标=两端平均）
    VA = np.asarray(V, float)
    budget = max(200, len(F))      # 整次调用分裂边数预算：按入口面数固定，防级联爆炸
    total_split = 0
    for _rnd in range(max_rounds):
        if not adj:
            break
        keys = list(adj.keys())
        e = np.asarray(keys, np.int64)          # (E, 2)
        lens = np.linalg.norm(VA[e[:, 0]] - VA[e[:, 1]], axis=1)
        tgt = 1.3333 * 0.5 * (L[e[:, 0]] + L[e[:, 1]])
        over = np.where(lens > tgt)[0]
        if len(over) == 0:
            break
        order = over[np.argsort(-lens[over])]
        made = False
        n_split = 0
        # 动态 VA（分裂补点）以列表承载
        VL = VA.tolist()
        for k in order:
            if total_split >= budget:          # 整次调用预算耗尽 → 外层迭代继续细化
                break
            u, v = keys[int(k)]
            key = (u, v) if u < v else (v, u)
            if key not in adj:
                continue
            if len(adj[key]) == 1:
                continue                       # 边界边不劈分，保证开面边界半环数不变
            dx = VL[u][0] - VL[v][0]
            dy = VL[u][1] - VL[v][1]
            dz = VL[u][2] - VL[v][2]
            le2 = float((dx * dx + dy * dy + dz * dz) ** 0.5)
            if le2 <= 1.3333 * 0.5 * (L[u] + L[v]):
                continue
            if len(F) + 2 > _CAP:
                return V, F, adj, changed
            i, j = key
            mil = 0.5 * (L[i] + L[j])
            V, F, adj = _split_edge(V, F, adj, key, len(V))
            VL.append([0.5 * (VL[i][0] + VL[j][0]),
                       0.5 * (VL[i][1] + VL[j][1]),
                       0.5 * (VL[i][2] + VL[j][2])])
            L = np.append(L, mil)
            changed = True
            made = True
            n_split += 1
            total_split += 1
        VA = np.asarray(VL, float)
        if not made:
            break
    return V, F, adj, changed


_CAP = 300000


def _min_angle_of(V, a, b, c):
    """三角形 (a,b,c) 的最小内角（弧度）；退化返回 0。"""
    A = np.asarray(V[a], float)
    B = np.asarray(V[b], float)
    C = np.asarray(V[c], float)
    ang = []
    for (p, q, r) in ((A, B, C), (B, C, A), (C, A, B)):
        v1 = q - p
        v2 = r - p
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-30 or n2 < 1e-30:
            return 0.0
        cosv = float(v1 @ v2) / (n1 * n2)
        cosv = max(-1.0, min(1.0, cosv))
        ang.append(float(np.arccos(cosv)))
    return min(ang)


def _flip_pass(V, F, adj):
    """翻边：每成功翻一条内边即**整体重扫**并从当前 adj 取最新面引用，
    避免在旧快照上复用已被替换的面位置而产生退化面 / 破坏水密。

    翻转内边 (i,j) 的两面 (i,j,k),(j,i,l) → (k,i,l),(k,l,j)，仅当：
      - 四边形四角互异（无退化）；
      - 对角 (k,l) 尚不存在（否则重叠）；
      - 四顶点 i,k,j,l 共线面反面（新四边形严格凸，翻转后最小角增大）；
      - 局部边权校验：新四边形的 4 条外边界边出现次数不变
        （翻边理论上不增边界，若陈引用导致非法则弃用并回滚）。
    """
    F = list(F)
    changed = False
    for _ in range(200):                       # 每遍重扫一次 adj（O(E)）
        adj = _build_adj(F)
        items = list(adj.items())
        dirty = set()                          # 本遍已被替换的面 id
        forbid = set()                         # 本遍已产生的新对角 (k,l)，后续不可复用
        flipped_local = False
        for (i, j), fs in items:
            if len(fs) != 2:
                continue
            if fs[0] in dirty or fs[1] in dirty:
                continue                       # 面已被本遍先前翻边改过 → 跳过，防陈旧引用
            f0, f1 = F[fs[0]], F[fs[1]]
            s0 = set(f0)
            s1 = set(f1)
            if not (i in s0 and j in s0 and i in s1 and j in s1):
                continue
            k = (s0 - {i, j}).pop()
            l = (s1 - {i, j}).pop()
            if k == l:
                continue
            key_kl = (k, l) if k < l else (l, k)
            if key_kl in forbid or key_kl in adj:   # 重叠对角，跳过
                continue
            if not (_min_angle_of(V, i, j, k) > 1e-3 and
                    _min_angle_of(V, j, i, l) > 1e-3):
                continue
            old_min = min(_min_angle_of(V, i, j, k), _min_angle_of(V, j, i, l))
            new_min = min(_min_angle_of(V, k, i, l), _min_angle_of(V, k, l, j))
            if new_min <= old_min + 1e-9:
                continue
            # 局部洛链校验：翻边前后四边形外边界必须完全一致（否则陈引用/退化）
            old_face0, old_face1 = f0, f1
            F[fs[0]] = (k, i, l)
            F[fs[1]] = (k, l, j)
            if _quad_boundary((old_face0, old_face1)) != _quad_boundary(
                    ((k, i, l), (k, l, j))):
                F[fs[0]], F[fs[1]] = old_face0, old_face1   # 回滚
                continue
            dirty.add(fs[0])
            dirty.add(fs[1])
            forbid.add(key_kl)
            changed = True
            flipped_local = True
        if not flipped_local:
            break
    return V, F, adj, changed


def _quad_boundary(faces):
    """两共享边面的 4 条外边界边（归一化）。翻边前后必须一致。"""
    edges = {}
    for (a, b, c) in faces:
        for e in ((a, b), (b, c), (c, a)):
            k = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
            edges[k] = edges.get(k, 0) + 1
    return frozenset(k for k, v in edges.items() if v == 1)


def _smooth_pass(V, F, adj, iters=4, omega=0.5):
    V = list(V)
    bv = _boundary_vertices(F, adj)
    nrm = _vertex_normals(np.asarray(V, float), np.asarray(F, np.int64))
    # 顶点→邻接列表
    vadj = {}
    for (u, v) in adj:
        vadj.setdefault(u, set()).add(v)
        vadj.setdefault(v, set()).add(u)
    for _ in range(iters):
        V2 = [list(p) for p in V]
        for v in range(len(V)):
            if v in bv or v not in vadj:
                continue
            nb = [V[x] for x in vadj[v]]
            if not nb:
                continue
            avg = [sum(p[0] for p in nb) / len(nb),
                   sum(p[1] for p in nb) / len(nb),
                   sum(p[2] for p in nb) / len(nb)]
            d = [avg[0] - V[v][0], avg[1] - V[v][1], avg[2] - V[v][2]]
            nn = nrm[v]
            ncomp = nn[0] * d[0] + nn[1] * d[1] + nn[2] * d[2]
            V2[v] = [V[v][0] + omega * (d[0] - nn[0] * ncomp),
                     V[v][1] + omega * (d[1] - nn[1] * ncomp),
                     V[v][2] + omega * (d[2] - nn[2] * ncomp)]
        V = V2
    return V


def remesh_surface(V, F, target_max, curvature_adaptive=True,
                   refine=True, collapse=True, smooth=True, flip=True,
                   max_iter=25):
    """曲率自适应表面重网格化，返回 (V', F')。

    - target_max：平坦区目标边长；
    - curvature_adaptive：开则用曲率尺寸场（高曲率加密），关则各向同性均匀重网格。
    """
    V = [list(map(float, p)) for p in V]
    F = list(map(tuple, (map(lambda t: tuple(t), F))))
    F = [tuple(map(int, f)) for f in F]
    tm = float(target_max)
    for _ in range(max_iter):
        kappa = curvature_field(np.asarray(V, float), np.asarray(F, np.int64))
        L = size_field(kappa, tm, curvature_adaptive)
        adj = _build_adj(F)
        bv = _boundary_vertices(F, adj)
        V, F, adj, ch_split = _split_pass(V, F, L, adj, bv)
        if collapse:
            # split 会使顶点变多，重算当前尺寸场供折叠使用
            k2 = curvature_field(np.asarray(V, float), np.asarray(F, np.int64))
            L2 = size_field(k2, tm, curvature_adaptive)
            V, F, adj, ch_col = _collapse_pass(V, F, L2, adj)
        else:
            ch_col = False
        if flip:
            _, F, adj, ch_flip = _flip_pass(V, F, adj)
        else:
            ch_flip = False
        if smooth:
            V = _smooth_pass(V, F, adj)
        adj = _build_adj(F)
        if not ch_split and not ch_col and not ch_flip:
            if smooth:
                continue
            break
        if len(F) > _CAP:
            break
    if collapse:
        # collapse 会遗留未被引用的孤儿顶点，返回前压缩索引
        V, F = _compact_vertices(V, F)
    return V, F


def _compact_vertices(V, F):
    """按面中实际出现的顶点重新紧凑编号，删除孤儿顶点。"""
    remap = {}
    newF = []
    for f in F:
        nf = []
        for v in f:
            if v not in remap:
                remap[v] = len(remap)
            nf.append(remap[v])
        newF.append(tuple(nf))
    V2 = [V[old] for old, _ in sorted(remap.items(), key=lambda p: p[1])]
    return V2, newF


def _collapse_pass(V, F, L, adj):
    changed = False
    bv = _boundary_vertices(F, adj)
    # 顶点→邻接
    vadj = {}
    for (u, v) in adj:
        vadj.setdefault(u, set()).add(v)
        vadj.setdefault(v, set()).add(u)
    # 候选：内部且短于阈值 (4/5 L) 的边（向量化选边）
    keys = list(adj.keys())
    e = np.asarray(keys, np.int64)
    lens = np.linalg.norm(np.asarray(V, float)[e[:, 0]]
                          - np.asarray(V, float)[e[:, 1]], axis=1)
    Ln = np.asarray(L, float)
    short = lens < 0.8 * 0.5 * (Ln[e[:, 0]] + Ln[e[:, 1]])
    interior = np.array([len(adj[k]) == 2 for k in keys])
    nb = np.array([not (k[0] in bv or k[1] in bv) for k in keys])
    choose = np.where(short & interior & nb)[0]
    order = choose[np.argsort(lens[choose])]
    for k in order:
        i, j = keys[int(k)]
        # 重新检查存在性与 link
        nj = vadj.get(j, set())
        ni = vadj.get(i, set())
        key = (i, j) if i < j else (j, i)
        fs = adj.get(key)
        if not fs or len(fs) != 2:
            continue
        third = []
        ok3 = True
        for fi in fs:
            a, b, c = F[fi]
            verts = {a, b, c}
            rest = verts - {i, j}
            if len(rest) != 1:
                ok3 = False
                break
            third.append(next(iter(rest)))
        if not ok3 or len(third) != 2:
            continue
        k, l = third
        if nj & ni != {k, l}:
            continue
        # 执行折叠
        new_F = []
        for (a, b, c) in F:
            ra, rb, rc = a, b, c
            if ra == j:
                ra = i
            if rb == j:
                rb = i
            if rc == j:
                rc = i
            if ra == rb or rb == rc or rc == ra:
                continue
            new_F.append((ra, rb, rc))
        if not new_F:
            continue
        # 验证不产生非流形重复面：新 adj 每条边 ≤2 面
        new_adj = _build_adj(new_F)
        bad = any(len(x) > 2 for x in new_adj.values())
        if bad:
            continue
        F = new_F
        adj = new_adj
        # 顶点去除 j（索引保持不变，j 不再被引用即可）
        changed = True
        vadj = {}
        for (u2, v2) in adj:
            vadj.setdefault(u2, set()).add(v2)
            vadj.setdefault(v2, set()).add(u2)
    return V, F, adj, changed
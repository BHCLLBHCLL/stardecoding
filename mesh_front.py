# -*- coding: utf-8 -*-
"""N 波 N2c：Delaunay 细化 + 前沿推进（纯 numpy，双环境可用）。

在 N2b（`mesh_remesh`）曲率自适应尺寸场之上补全 N2 的 B 路线：

Delaunay（自研）：
  - `delaunay_metric` / `delaunay_refine_surface`：内蕴准测 —— 内边 (i,j) 若共享两三角
    (i,j,k),(j,i,l) 且 k、l 处对边角 α+β ≤ π 则该边 Delaunay（平面下等价空外接圆 / max-min 角）；
  - `delaunay_flip`：逐内边翻转直到**每条内边皆 Delaunay**（脏面跳过 + 对角禁重）；
  - `bowyer_watson`：2D 点云增量 Delaunay 三角化（超三角→逐点插入→扇形重建）。

前沿推进（自研）：
  - `advancing_front`：从边界环（外环 CCW + 可选内孔 CW）向内推三角前缘，按目标边长
    优选理想雀形 / 复用前缘邻接顶点关隙，产出**正面积、覆盖域、边界保持**的三角网；
  - `remesh_wavy_advancing`：对图曲面 z=f(x,y) 取 (x,y) 域做前沿推进再提升。

纯 Python / numpy，两环境（默认 / conda occ）皆可跑。
"""
import numpy as np

from mesh_remesh import (_build_adj, _compact_vertices, _split_pass,
                         _collapse_pass, _boundary_vertices, curvature_field,
                         size_field, _smooth_pass)

_PI = float(np.pi)


# ---------------------------------------------------------------------------
# 几何小工具（3D / 2D）
# ---------------------------------------------------------------------------
def _cwx(o, a, b):
    """2D 叉积 z 分量：(a-o) × (b-o)。"""
    o = np.asarray(o, float)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _min_angle_deg(a, b, c):
    """平面三角形 (a,b,c) 最小内角（度）。退化返回 0。"""
    a, b, c = (np.asarray(x, float) for x in (a, b, c))
    ab = np.linalg.norm(a - b)
    ac = np.linalg.norm(a - c)
    bc = np.linalg.norm(b - c)
    if min(ab, ac, bc) < 1e-12:
        return 0.0
    def ang(p, q, r):
        v1 = q - p
        v2 = r - p
        cs = float(v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return float(np.degrees(np.arccos(max(-1.0, min(1.0, cs)))))
    return min(ang(a, b, c), ang(b, c, a), ang(c, a, b))


def _tri_angles_at(V, a, b, c):
    """3D 三角形 (a,b,c) 三内角（弧）。退化返回 (0,0,0)。"""
    A = np.asarray(V[a], float)
    B = np.asarray(V[b], float)
    C = np.asarray(V[c], float)
    ab = float(np.linalg.norm(A - B))
    ac = float(np.linalg.norm(A - C))
    bc = float(np.linalg.norm(B - C))
    if min(ab, ac, bc) < 1e-12:
        return 0.0, 0.0, 0.0
    def ang(pa, pb, pc):
        cs = (pb * pb + pc * pc - pa * pa) / (2.0 * pb * pc)
        return float(np.arccos(max(-1.0, min(1.0, cs))))
    return (ang(bc, ab, ac), ang(ac, ab, bc), ang(ab, ac, bc))


def _opposite_angles(V, F, i, j, k, l):
    """对边 (i,j) 的两邻三角在 k、l 处的对边角 (α, β)。"""
    _, _, atk = _tri_angles_at(V, i, k, j)     # (i,k,j) 第三角在 k
    _, _, atl = _tri_angles_at(V, i, l, j)     # (i,l,j) 第三角在 l
    return atk, atl


# ---------------------------------------------------------------------------
# Delaunay：准测、定位、一致
# ---------------------------------------------------------------------------
def delaunay_metric(V, F):
    """返回 (violations, interior_edges)：违反 α+β≤π 的内边数。"""
    V = np.asarray(V, float)
    F = [tuple(map(int, f)) for f in F]
    adj = _build_adj(F)
    vio = 0
    interior = 0
    for (i, j), fs in adj.items():
        if len(fs) != 2:
            continue
        interior += 1
        sA = set(F[fs[0]])
        sB = set(F[fs[1]])
        if not (i in sA and j in sA and i in sB and j in sB):
            continue
        k = (sA - {i, j}).pop()
        l = (sB - {i, j}).pop()
        if k == l:
            continue
        atk, atl = _opposite_angles(V, F, i, j, k, l)
        if atk + atl > _PI + 1e-9:
            vio += 1
    return vio, interior


def delaunay_flip(V, F, max_passes=96):
    """逐内边翻转直到每条内边皆 Delaunay（内蕴角准则）。返回新 F。"""
    V = np.asarray(V, float)
    F = [tuple(map(int, f)) for f in F]
    for _ in range(max_passes):
        adj = _build_adj(F)
        items = list(adj.items())
        dirty = set()
        forbid = set()
        flipped = False
        for (i, j), fs in items:
            if len(fs) != 2 or fs[0] in dirty or fs[1] in dirty:
                continue
            sA = set(F[fs[0]])
            sB = set(F[fs[1]])
            if not (i in sA and j in sA and i in sB and j in sB):
                continue
            k = (sA - {i, j}).pop()
            l = (sB - {i, j}).pop()
            if k == l:
                continue
            atk, atl = _opposite_angles(V, F, i, j, k, l)
            if atk + atl <= _PI + 1e-9:
                continue
            key_kl = (k, l) if k < l else (l, k)
            if key_kl in forbid or key_kl in adj:
                continue
            if _quad_boundary((F[fs[0]], F[fs[1]])) != _quad_boundary(
                    ((k, i, l), (k, l, j))):
                continue
            F[fs[0]] = (k, i, l)
            F[fs[1]] = (k, l, j)
            dirty.add(fs[0])
            dirty.add(fs[1])
            forbid.add(key_kl)
            flipped = True
        if not flipped:
            break
    return F


def _quad_boundary(faces):
    edges = {}
    for (a, b, c) in faces:
        for e in ((a, b), (b, c), (c, a)):
            k = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
            edges[k] = edges.get(k, 0) + 1
    return frozenset(k for k, v in edges.items() if v == 1)


def delaunay_refine_surface(V, F, target_max, curvature_adaptive=True,
                            smooth=True, max_iter=12, min_ratio=0.2):
    """曲率自适应 Delaunay 一致表面重网格化（复用 N2b split/collapse + 强制 Delaunay）。

    返回 (V', F')：顶点压缩、水密/边界保持、每条内边皆 Delaunay。
    """
    V = [list(map(float, p)) for p in V]
    F = [tuple(map(int, f)) for f in F]
    tm = float(target_max)
    for _ in range(max_iter):
        kap = curvature_field(np.asarray(V, float), np.asarray(F, np.int64))
        L = size_field(kap, tm, curvature_adaptive, min_ratio=min_ratio)
        adj = _build_adj(F)
        bv = _boundary_vertices(F, adj)
        V, F, adj, ch_split = _split_pass(V, F, L, adj, bv)
        k2 = curvature_field(np.asarray(V, float), np.asarray(F, np.int64))
        L2 = size_field(k2, tm, curvature_adaptive, min_ratio=min_ratio)
        V, F, adj, ch_col = _collapse_pass(V, F, L2, adj)
        F = delaunay_flip(V, F)
        if smooth:
            V = _smooth_pass(V, F, adj)
        if not (ch_split or ch_col):
            break
    V, F = _compact_vertices(V, F)
    return V, F


# ---------------------------------------------------------------------------
# Bowyer–Watson 2D 增量 Delaunay
# ---------------------------------------------------------------------------
def _in_circumcircle(a, b, c, p):
    """p 是否在三角形 (a,b,c) 外接圆内（a,b,c 逆时针；共圆/外部返回 False）。"""
    a, b, c, p = (np.asarray(x, float) for x in (a, b, c, p))
    ax, ay, bx, by, cx, cy, px, py = (a[0], a[1], b[0], b[1], c[0], c[1], p[0], p[1])
    adx, ady = ax - px, ay - py
    bdx, bdy = bx - px, by - py
    cdx, cdy = cx - px, cy - py
    al = adx * adx + ady * ady
    bl = bdx * bdx + bdy * bdy
    cl = cdx * cdx + cdy * cdy
    det = (al * (bdx * cdy - bdy * cdx)
           - bl * (adx * cdy - ady * cdx)
           + cl * (adx * bdy - ady * bdx))
    return det > 1e-12


def bowyer_watson(points):
    """2D 点云增量 Delaunay 三角化 → 三角索引三元组列表。"""
    pts = [tuple(np.asarray(p, float)) for p in points]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mnx, mxx, mny, mxy = min(xs), max(xs), min(ys), max(ys)
    m = max(mxx - mnx, mxy - mny) or 1.0
    cx, cy = 0.5 * (mnx + mxx), 0.5 * (mny + mxy)
    pts.extend([(cx - 8 * m, cy - m), (cx + 8 * m, cy - m), (cx, cy + 8 * m)])
    N = len(pts) - 3
    tris = [(N, N + 1, N + 2)]
    for i in range(N):
        p = pts[i]
        bad = [t for t in tris if _in_circumcircle(pts[t[0]], pts[t[1]], pts[t[2]], p)]
        ec = {}
        for t in bad:
            a, b, c = t
            for e in ((a, b), (b, c), (c, a)):
                k = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
                ec[k] = ec.get(k, 0) + 1
        tris = [t for t in tris if t not in bad]
        for (a, b), cnt in ec.items():
            if cnt == 1:
                o = _cwx(pts[a], pts[b], p)
                tris.append((a, b, i) if o > 1e-12 else (b, a, i))
    return [t for t in tris if max(t) < N]


def is_delaunay2d(pts, tris):
    """2D 点云三角网是否全部空外接圆（含共圆则视为满足）。返回违规数。"""
    P = [np.asarray(p, float) for p in pts]
    vio = 0
    for (a, b, c) in tris:
        for x in range(len(P)):
            if x in (a, b, c):
                continue
            if _in_circumcircle(P[a], P[b], P[c], P[x]):
                vio += 1
    return vio


# ---------------------------------------------------------------------------
# 前沿推进（2D 边界环 → 三角网）
# ---------------------------------------------------------------------------
def _ring_area(poly):
    pts = [np.asarray(x, float) for x in poly]
    s = 0.0
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        s += a[0] * b[1] - a[1] * b[0]   # 2D cross z 分量（兼容新旧 numpy）
    return 0.5 * s


def _point_in_poly(p, poly):
    p = np.asarray(p, float)
    x, y = p[0], p[1]
    inside = False
    pts = [np.asarray(q, float) for q in poly]
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        if (a[1] > y) != (b[1] > y):
            xc = (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]
            if x < xc:
                inside = not inside
    return inside


def advancing_front(outer, holes=(), target=0.1, max_tri=8000):
    """2D 前沿推进：由外环（CCW，随意）+ 内孔（CW，随意）向内推三角前缘。

    返回 (V, F)、V:(K,2)。前缘每步取第一条边 (a,b)：
      - 首选理想雀形点（等边、域内侧、目标边长）；
      - 若该雀形点已太近前缘邻接顶点（< 0.62·target）则改接到该既有顶点关隙；
      - 候选须使三角形正面积且质心在域内（外环内、孔外）。
    前缘合并用 `ins/outs` 邻接维护，保证正面积、覆盖、边界保持。
    """
    outer = [tuple(map(float, x[:2])) for x in outer]
    if _ring_area(outer) < 0:
        outer = outer[::-1]
    holes = [[tuple(map(float, x[:2])) for x in h] for h in holes]
    rings = [outer] + holes

    def inside(p):
        if not _point_in_poly(p, rings[0]):
            return False
        for h in rings[1:]:
            if _point_in_poly(p, h):
                return False
        return True

    # 建立统一点表（含环顶点）
    V = []
    eye = {}
    F = []

    def add(p):
        key = (round(float(p[0]), 11), round(float(p[1]), 11))
        if key in eye:
            return eye[key]
        eye[key] = len(V)
        V.append(key)
        return len(V) - 1

    h = float(target)
    # 前缘：CCW 外环、CW 内孔（内部一律左侧）；先细分到 ~h 边长，
    # 使理想雀形推进能干净地铺满域、避免粗长边过早乱连角点。
    front = []
    for ri, ring in enumerate(rings):
        pts = ring
        if _ring_area(pts) < 0:
            pts = pts[::-1]
        sub = []
        for k in range(len(pts)):
            p0, p1 = np.asarray(pts[k], float), np.asarray(pts[(k + 1) % len(pts)], float)
            seg = p1 - p0
            n = max(1, int(np.ceil(np.linalg.norm(seg) / h)))
            for m in range(n):
                sub.append(p0 + (m / n) * seg)
        idx = [add(x) for x in sub]
        n = len(idx)
        for k in range(n):
            a, b = idx[k], idx[(k + 1) % n]
            if ri == 0:
                front.append((a, b))
            else:
                front.append((b, a))     # 孔反向 → 内部左
    # 规范化偶消（重叠方向的边缘化）
    cnt = {}
    for e in front:
        cnt[e] = cnt.get(e, 0) + 1
    front = [e for e, c in cnt.items() if c % 2 == 1]

    seen = set()                              # 已生成三角形（frozenset），防重复生成

    ins = {v: set() for v in range(len(V))}   # v ← 前缘 (u,v)
    outs = {v: set() for v in range(len(V))}  # v → 前缘 (v,w)
    for (a, b) in front:
        outs[a].add(b)
        ins[b].add(a)

    def discard_edges(*edges):
        for (u, v) in edges:
            if u in outs and v in outs[u]:
                outs[u].discard(v)
                ins[v].discard(u)

    from collections import deque
    dq = deque(front)
    made = 0
    guard = 0
    # 已落盘三角形（坐标）及其包围盒 → 质心遮挡测试，杜绝重复覆盖（判定先进）。
    fill = []
    fill_bb = []

    def _pt_in_tri(p, t):                       # 含边界（质心快速遮挡）
        a, b, c = t
        d1 = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        d2 = (c[0] - b[0]) * (p[1] - b[1]) - (c[1] - b[1]) * (p[0] - b[0])
        d3 = (a[0] - c[0]) * (p[1] - c[1]) - (a[1] - c[1]) * (p[0] - c[0])
        neg = d1 < 0 or d2 < 0 or d3 < 0
        pos = d1 > 0 or d2 > 0 or d3 > 0
        return not (neg and pos)

    def occluded(p):
        px, py = float(p[0]), float(p[1])
        return any(mnx - 1e-12 <= px <= mxx + 1e-12 and
                   mny - 1e-12 <= py <= mxy + 1e-12 and _pt_in_tri((px, py), t)
                   for (t, (mnx, mny, mxx, mxy)) in zip(fill, fill_bb))

    while dq and made < max_tri and guard < max_tri * 80:
        guard += 1
        a, b = dq.popleft()
        if b not in outs.get(a, ()):
            continue                             # 该前缘边已被消费
        Va = np.asarray(V[a], float)
        Vb = np.asarray(V[b], float)
        ed = Vb - Va
        L = float(np.linalg.norm(ed))
        if L < 1e-12:
            discard_edges((a, b))
            continue
        nrm = np.array([-ed[1], ed[0]]) / L
        h = float(target)
        mid = 0.5 * (Va + Vb)
        # 雀形高度取当前边长等边高度并封顶 h：短余段不过冲、长边不超量。
        apex = mid + (np.sqrt(3.0) / 2.0) * min(L, h) * nrm

        # 候选：a·b 的前缘邻接顶点（局部关隙）+ 理想雀形点。
        # 正确优先级：优先在邻近产生理想雀形点；仅当某既有前缘顶点已非常接近
        # 雀形（≤0.62·h，即关隙）时复用该既有顶点，避免过早乱连边界角点。
        cand = set()
        cand |= {c for c in ins.get(a, ()) if c != b}
        cand |= {c for c in outs.get(b, ()) if c != a}

        def quality(c):
            ac = np.asarray(V[c], float)
            if _cwx(Va, Vb, ac) < 1e-12:        # 三角形须域内侧
                return -1.0
            cen = (Va + Vb + ac) / 3.0
            if not inside(cen):
                return -1.0
            if occluded(cen):                    # 质心已落入既有三角形 → 拒
                return -1.0
            return _min_angle_deg(Va, Vb, ac)

        # 所有合法既有候选按“距雀形最近”排序，用于关隙/兜底
        valid = [(c, float(np.linalg.norm(np.asarray(V[c], float) - apex)))
                 for c in cand if quality(c) > 0.0]
        close = [c for c, d in valid if d <= 0.62 * h]

        c = None
        if close:                                # 既有关隙顶点且紧邻 → 复用
            c = min(close, key=lambda x: float(np.linalg.norm(
                np.asarray(V[x], float) - apex)))
        elif inside(apex):                       # 无紧邻既有顶点 → 生成理想雀形
            cnew = add(apex)
            ins.setdefault(cnew, set())
            outs.setdefault(cnew, set())
            if quality(cnew) > 0.0:
                c = cnew
        if c is None:
            # 该边既无合法雀形点也无既有候选（其推进侧已被覆盖）→ 消费该前缘边关隙
            discard_edges((a, b))
            continue
        key3 = frozenset((a, b, c))
        if key3 in seen:                         # 已生成过 → 消费并跳过（该处已填）
            discard_edges((a, b))
            continue

        # —— 提交三角形 (a,b,c)，合并前缘 ——
        # 挖掉旧 a->b，新前缘应为 a->c->b（内部仍左侧，未填区向里缩）：
        #   边 a->c：若反向 c->a 已存在则抵消（双侧都填满），否则新增 a->c；
        #   边 c->b：若反向 b->c 已存在则抵消，否则新增 c->b。
        discard_edges((a, b))                    # 删旧 (a,b)
        if c in outs and a in outs.get(c, ()):   # 反向 c->a 存在 → 抵消
            discard_edges((c, a))
        else:
            outs.setdefault(a, set()).add(c)
            ins.setdefault(c, set()).add(a)
        if b in outs and c in outs.get(b, ()):   # 反向 b->c 存在 → 抵消
            discard_edges((b, c))
        else:
            outs.setdefault(c, set()).add(b)
            ins.setdefault(b, set()).add(c)
        F.append((a, b, c))
        seen.add(key3)
        _cen = ((Va[0] + Vb[0] + float(V[c][0])) / 3.0,
                (Va[1] + Vb[1] + float(V[c][1])) / 3.0)
        fill.append((tuple(Va), tuple(Vb), tuple(np.asarray(V[c], float))))
        fill_bb.append((min(Va[0], Vb[0], V[c][0]), min(Va[1], Vb[1], V[c][1]),
                        max(Va[0], Vb[0], V[c][0]), max(Va[1], Vb[1], V[c][1])))
        made += 1
        # 重新入队新前缘出边
        for x in (a, b, c):
            for w in tuple(outs.get(x, ())):
                dq.append((x, w))

    V2, F2 = [], []
    remap = {}
    for (a, b, c) in F:
        vv = []
        for x in (a, b, c):
            if x not in remap:
                remap[x] = len(remap)
            vv.append(remap[x])
        F2.append(tuple(vv))
    for old in sorted(remap, key=remap.get):
        V2.append(V[old])
    return V2, F2


def remesh_wavy_advancing(outer_xy, z_fn, target, max_tri=8000):
    """对图曲面 z=z_fn(x,y)：域边界环 outer_xy → 前沿推进 → 提升 z。

    返回 (V3, F)，V3:(K,3)。
    """
    V2, F2 = advancing_front(outer_xy, (), target, max_tri)
    V3 = []
    for (x, y) in V2:
        V3.append((x, y, float(z_fn(x, y))))
    return V3, F2
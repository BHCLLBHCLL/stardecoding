# -*- coding: utf-8 -*-
"""S2 第 3 步 ③(a)：方形 O 型环带 + 张量积外围的**协调混合网格**。

背景（S2 贴体瞬态阻断的穷举结论，见 s_wave_plan.md）：经典 O 型网格（射线直达
矩形边界）非正交角 median 56.6° / 偏斜 p95 0.97 / max 1.25，稳态配合非正交修正
可解（残差 5.4e-6），但瞬态在对流格式 × 修正开关 × pc_inner × n_inner 的全组合下
都发散 → 阻断点是网格本身。已单独验证：极坐标环带（r0 → r1 = 10·r0，几何径向
加密）对求解器友好（稳态残差 3.4e-5，|u|max 0.048 ≈ 物理量级）。

本模块把「求解器友好的环带」与「近笛卡尔外围」**协调**拼接：

  · 环带外边界取**正方形**（边长 2a，与圆柱同心），四条边落在张量网格线上；
  · 环带为方形 O 型网格：4m 条射线（θ = 45° + i·Δ，Δ = 90°/m —— 四条对角线方向
    恰为射线 → 正方形四角是环带边界顶点），沿射线从圆柱面**直线**连到正方形边；
  · 正方形每边 m 段细分，细分偏移 off_j = a·tan(−45° + jΔ)。环带射线在边上的落点
    与细分点满足解析恒等式 a·cot(45°+iΔ) = a·tan(−45°+(m−i)Δ)，实现时**直接用
    同一数组取值**（端点 off_0/off_m 显式置 ±a），零浮点失配；
  · 外围 9 块张量积块（左右带、上下带、四角块）共用这些线数组 → 全域协调。

三维：2D 四边形按「短对角线」切成 2 三角形（对角线只在四边形内部、不跨单元，
不影响协调性）→ 沿 z 拉伸成三棱柱 → 按全局顶点编号规范化的 3-tet 分解
（channel_tet_mesh 的 S4 修复同款：侧面四边形对角线取「编号较小的底面顶点」所
连的对角线 —— 共享该面的两个单元算出**同一条**对角线 → 协调）。

解析核对：体积 = (L·H − π·r0²)·T（方形与圆之间 = (2a)² − π·r0²，方形外各块恰好
补成 L·H − (2a)²）。若环带外边界偏离正方形、或任何拼接失配，体积/协调性检查
都会立刻暴露。
"""
import math

import numpy as np


def _graded_line(span, h_near, far_ratio, n_min=2):
    """单调渐变线：靠近起点间距 ≈ h_near，几何增长，最远间距 ≤ h_near·far_ratio。

    用于尾迹加密：近场（正方形边）细、远场（出口/壁）粗，单元数远少于均匀加密。
    返回精确覆盖 [0, span] 的含端点数组（末端重标定保证几何闭合）。
    """
    span = float(span)
    h_near = float(h_near)
    far_ratio = float(far_ratio)
    if span <= 0.0 or h_near <= 0.0 or far_ratio <= 1.0:
        return np.linspace(0.0, span, max(int(math.ceil(span / h_near)), 1) + 1)
    for n in range(max(n_min, 2), 500):
        # 解 r：h_near·(r^n − 1)/(r − 1) = span（n 段几何和恰为 span）
        lo, hi = 1.0 + 1e-9, 20.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            s = (h_near * (mid ** n - 1.0) / (mid - 1.0)
                 if mid > 1.0 + 1e-12 else h_near * n)
            if s < span:
                lo = mid
            else:
                hi = mid
        r = 0.5 * (lo + hi)
        if r <= 1.0 + 1e-6:
            break                              # 均匀已足够（span/ h_near 段）
        last = h_near * r ** (n - 1)
        if last <= h_near * far_ratio * 1.05:
            steps = h_near * r ** np.arange(n)
            pos = np.concatenate(([0.0], np.cumsum(steps)))
            return pos * (span / pos[-1])
    return np.linspace(0.0, span, max(int(math.ceil(span / h_near)), 1) + 1)


def hybrid_channel_mesh(D, length_D=16.0, height_D=8.0, thickness_D=0.25,
                        center_x_D=4.0, h_factor=4.0, m=24, n_r=16,
                        a_D=3.0, stretch=1.3, n_layers=1, stretch_far=None):
    """混合网格：方形 O 型环带（贴体圆柱）+ 张量积外围（通道）。

    默认参数 = S2 实测工作点（19,020 tet，非正交 median 32.6°/p95 69.0°、偏斜
    p95 0.35、d_n/|d|<0.15 占比 0%，稳态 nonorth corr=0.33 收敛 4.9e-5）：
    thickness_D=0.25（z 厚 ≈ 外围 h → tet 分解长细比≈1，这是质量杠杆，
    thickness_D=0.5 时 p95 会恶化到 ~83°）、m=24、n_r=16、stretch=1.3。

    参数：D 圆柱直径；length_D/height_D 通道长高（×D）；thickness_D 展向厚（×D）；
    center_x_D 圆心 x（×D）；h_factor 外围分辨率（h = D/h_factor）；m 每边细分段数
    （角向射线总数 = 4m）；n_r 径向层数；a_D 正方形半边长（×D，须 < 距壁最小距离）；
    stretch 径向加密幂次（ts = linspace(0,1,n_r+1)**stretch）；n_layers z 向层数。
    返回 dict（vertices/cells 全四面体，与 channel_tet_mesh 同构，可直接喂
    PressureSolver / run_case(mesh=...)）。
    """
    D = float(D)
    L, H = float(length_D) * D, float(height_D) * D
    T = float(thickness_D) * D
    h = D / float(h_factor)
    cx, cy = float(center_x_D) * D, 0.5 * H
    r0 = 0.5 * D
    a = float(a_D) * D
    m = int(m)
    n_r = int(n_r)
    n_layers = int(n_layers)
    if m < 4:
        raise ValueError("m 必须 ≥ 4（每边至少 4 段）")
    if n_r < 2 or n_layers < 1:
        raise ValueError("n_r ≥ 2 且 n_layers ≥ 1")
    if not (cx - a > 0.0 and cx + a < L and cy - a > 0.0 and cy + a < H):
        raise ValueError("正方形半边长 a=%.4g 超出通道：需 cx±a ∈ (0, L)、cy±a ∈ (0, H)"
                         % a)
    dth = math.pi / (2.0 * m)                     # Δ = 90°/m
    # 正方形每边的细分偏移（m+1 个，从 −a 到 +a）；端点显式置精确值（tan(±45°) 的
    # 浮点值是 0.9999999999999999，会让角点与 linspace 端点差 1 ulp → 顶点失配）
    off = [a * math.tan(-math.pi / 4.0 + j * dth) for j in range(m + 1)]
    off[0], off[m] = -a, a
    XT = [cx + v for v in off]                    # 上/下边的 x 细分线
    YT = [cy + v for v in off]                    # 左/右边的 y 细分线
    # 外围线：stretch_far=None → 均匀 h；给定（如 4.0）→ 尾迹几何渐变加密
    # （细端必须在**靠正方形**一侧：xL/yB 的细端在 span 终点，xR/yT 的细端在起点）
    if stretch_far:
        xL = (cx - a) - _graded_line(cx - a, h, stretch_far)[::-1]
        xR = (cx + a) + _graded_line(L - cx - a, h, stretch_far)
        yB = (cy - a) - _graded_line(cy - a, h, stretch_far)[::-1]
        yT = (cy + a) + _graded_line(H - cy - a, h, stretch_far)
    else:
        xL = np.linspace(0.0, cx - a, max(int(math.ceil((cx - a) / h)), 1) + 1)
        xR = np.linspace(cx + a, L, max(int(math.ceil((L - cx - a) / h)), 1) + 1)
        yB = np.linspace(0.0, cy - a, max(int(math.ceil((cy - a) / h)), 1) + 1)
        yT = np.linspace(cy + a, H, max(int(math.ceil((H - cy - a) / h)), 1) + 1)

    pts2 = []
    vmap = {}

    def vid(x, y):
        # 顶点去重：坐标舍入到 1e-9（真实最小间距 ~1e-3，安全余量 6 个量级）。
        # 环带外边界点直接取自 XT/YT 数组（t=1 特判）→ 与外围块的键**逐位相同**。
        k = (round(float(x), 9), round(float(y), 9))
        i = vmap.get(k)
        if i is None:
            i = len(pts2)
            vmap[k] = i
            pts2.append((float(x), float(y)))
        return i

    quads = []

    def emit_block(xs, ys):
        xs = np.asarray(xs, float)
        ys = np.asarray(ys, float)
        for i in range(xs.size - 1):
            for j in range(ys.size - 1):
                quads.append((vid(xs[i], ys[j]), vid(xs[i + 1], ys[j]),
                              vid(xs[i + 1], ys[j + 1]), vid(xs[i], ys[j + 1])))

    # ① 方形 O 型环带：每条射线从圆柱面（r0）直线连到正方形边（落点 = 边细分点）
    n_th = 4 * m
    ts = np.linspace(0.0, 1.0, n_r + 1) ** float(stretch)
    P = [[0] * (n_r + 1) for _ in range(n_th)]
    for i in range(n_th):
        th = math.pi / 4.0 + i * dth
        ix, iy = cx + r0 * math.cos(th), cy + r0 * math.sin(th)
        if i % m == 0:                            # 角点射线（θ = 45°+k·90°）
            c = i // m
            ox, oy = ((XT[m], YT[m]), (XT[0], YT[m]),
                      (XT[0], YT[0]), (XT[m], YT[0]))[c]
        else:
            b = i // m
            if b == 0:                            # 上边：x = XT[m−i]（i=1 在右上角旁）
                ox, oy = XT[m - i], YT[m]
            elif b == 1:                          # 左边：y = YT[2m−i]
                ox, oy = XT[0], YT[2 * m - i]
            elif b == 2:                          # 下边：x = XT[i−2m]
                ox, oy = XT[i - 2 * m], YT[0]
            else:                                 # 右边：y = YT[i−3m]
                ox, oy = XT[m], YT[i - 3 * m]
        for k in range(n_r + 1):
            if k == 0:
                P[i][k] = vid(ix, iy)
            elif k == n_r:
                P[i][k] = vid(ox, oy)             # 外边界点 = 数组原值（零失配）
            else:
                t = ts[k]
                P[i][k] = vid(ix + t * (ox - ix), iy + t * (oy - iy))
    for i in range(n_th):
        i2 = (i + 1) % n_th
        for k in range(n_r):
            quads.append((P[i][k], P[i][k + 1], P[i2][k + 1], P[i2][k]))

    # ② 外围 9 块张量积（与环带正方形边界逐点共线 → 协调拼接）
    emit_block(xL, YT)                            # 左带
    emit_block(xR, YT)                            # 右带（尾迹）
    emit_block(XT, yT)                            # 上带
    emit_block(XT, yB)                            # 下带
    emit_block(xL, yB)                            # 左下角
    emit_block(xR, yB)                            # 右下角
    emit_block(xL, yT)                            # 左上角
    emit_block(xR, yT)                            # 右上角

    # ③ 2D 三角化：短对角线（对角线只在本四边形内部 → 与邻块无关，不破坏协调）
    V2 = np.asarray(pts2, float)
    tris = []
    for (v0, v1, v2, v3) in quads:
        d02 = float(np.sum((V2[v0] - V2[v2]) ** 2))
        d13 = float(np.sum((V2[v1] - V2[v3]) ** 2))
        if d02 <= d13:
            tris.append((v0, v1, v2))
            tris.append((v0, v2, v3))
        else:
            tris.append((v1, v2, v3))
            tris.append((v1, v3, v0))

    # ④ z 向拉伸（n_layers 层）→ 三棱柱 → 全局编号规范化 3-tet 分解
    nz = n_layers + 1
    zs = np.linspace(0.0, T, nz)
    n2 = len(pts2)
    V = np.zeros((n2 * nz, 3), float)
    for kk in range(nz):
        V[kk * n2:(kk + 1) * n2, :2] = V2
        V[kk * n2:(kk + 1) * n2, 2] = zs[kk]
    cells = []
    for kk in range(n_layers):
        bb, tt = kk * n2, (kk + 1) * n2
        for (a0, a1, a2) in tris:
            pairs = sorted(((a0 + bb, a0 + tt), (a1 + bb, a1 + tt),
                            (a2 + bb, a2 + tt)), key=lambda t: t[0])
            (p0, q0), (p1, q1), (p2, q2) = pairs
            # 侧面四边形对角线 = 编号较小的底面顶点所连 —— 共享面两侧一致 → 协调
            cells.append((p0, p1, p2, q2))
            cells.append((p0, p1, q2, q1))
            cells.append((p0, q1, q2, q0))
    C = np.asarray(cells, np.int64)
    from mesh_tet import _tet_volumes
    vol = _tet_volumes(V, C)
    n_flip = int((vol < 0).sum())
    if n_flip:
        C = C.copy()
        C[vol < 0] = C[vol < 0][:, [0, 1, 3, 2]]
        vol = _tet_volumes(V, C)
    analytic = (L * H - math.pi * r0 ** 2) * T
    return {"ok": True, "mesher": "hybrid", "vertices": V, "cells": C,
            "n_cells": int(C.shape[0]), "n_points": int(V.shape[0]),
            "n_quads_2d": len(quads), "n_prisms": len(tris) * n_layers,
            "volume": float(vol.sum()), "volume_exact": analytic,
            "volume_rel_err": abs(float(vol.sum()) - analytic) / analytic,
            "n_negative": int((vol <= 0.0).sum()), "n_flipped": n_flip,
            "hole_center": (cx, cy), "hole_r": r0, "D": D,
            "thickness": T, "length": L, "height": H, "h": h,
            "a": a, "m": m, "n_theta": n_th, "n_r": n_r,
            "stretch": float(stretch), "n_layers": n_layers,
            "stretch_far": (float(stretch_far) if stretch_far else None),
            "square": (cx - a, cx + a, cy - a, cy + a),
            "quality_proxy": {"min_vol": float(vol.min()),
                              "max_vol": float(vol.max()),
                              "mean_vol": float(vol.mean())}}


def hybrid_mesh_report(mesh):
    """混合网格自检：协调性 + 非正交/偏斜 + 体积（供 CLI/测试/文档复用）。"""
    from mesh_amr import mesh_conformity
    from mesh_quality import orthogonality_report
    V = np.asarray(mesh["vertices"], float)
    C = np.asarray(mesh["cells"], np.int64)
    conf = mesh_conformity(V, C)
    orth = orthogonality_report(V, C, kind="tet")
    return {"conformity": conf, "orthogonality": orth,
            "volume_rel_err": mesh.get("volume_rel_err"),
            "n_negative": mesh.get("n_negative")}

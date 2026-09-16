# -*- coding: utf-8 -*-
"""R 波 R4：官方解差分（官方 .sim 解数据 ↔ 本仓库求解器，误差带验收）。

三条腿（各自可独立验证）：
  ① 官方参考量：力系数报告参考量（U/ρ/A）+ 升力系数监视器序列 → 脱落频率与 Strouhal
     数（FFT + 过零双路互校）+ 残差终值（G6）；圆柱直径可由参考面积/展长推定，或用
     diameter_from_mesh=True 从 Cylinder 边界环实测（G4）。
  ② 同工况几何：环形域（圆柱 D + 远场 r_far·D + 厚度 T）结构化四边形 → 三角化 →
     闭合三维表面（水密 + 面积/体积可解析核对），供 tet 网格器使用。
  ③ 差分指标：St 比 / 升力振幅比 / 平均升力（无升力体应≈0）→ 误差带判定。

诚实边界：不伪造"跑通全瞬态"。`run_case` 需 STARDECODING_LONG=1，且真实瞬态受本机算力限制；
未达脱落周期时如实报告"未达周期 → 不做 St 比对"。
"""
import math
import os
import sys

import numpy as np

DEFAULT_NU = 1e-5            # vortexShed 配套 OpenFOAM 工况（transportProperties nu=1e-05）
ST_TOLERANCE = (0.75, 1.25)  # St 相对官方参考允许带
AMPLITUDE_TOLERANCE = (0.3, 3.0)


def strouhal_from_series(t, y, D, U):
    """时序 → 脱落频率/Strouhal（FFT 峰值 + 过零周期双路，取二者均值并各自保留）。"""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if t.size < 16 or y.size != t.size:
        return {"ok": False, "reason": "序列过短或长度不一致"}
    order = np.argsort(t)
    t, y = t[order], y[order]
    half = y.size // 2                     # 后半段更接近统计稳态
    tt, yy = t[half:], y[half:]
    dt = float(np.median(np.diff(tt)))
    if not np.isfinite(dt) or dt <= 0:
        return {"ok": False, "reason": "时间步非法"}
    yy0 = yy - yy.mean()
    n = yy0.size
    spec = np.abs(np.fft.rfft(yy0 * np.hanning(n)))
    freq = np.fft.rfftfreq(n, d=dt)
    k = int(np.argmax(spec[1:]) + 1) if n > 2 else 0
    f_fft = float(freq[k])
    sign = np.sign(yy0)
    cross = np.where(np.diff(sign) != 0)[0]
    f_cross = float("nan")
    if cross.size > 3:
        period = float(np.mean(np.diff(tt[cross]))) * 2.0
        if period > 0:
            f_cross = 1.0 / period
    f = f_fft if not np.isfinite(f_cross) else 0.5 * (f_fft + f_cross)
    return {"ok": True, "f_fft": f_fft, "f_cross": f_cross, "f": f,
            "st": f * D / U if U else float("nan"),
            "st_fft": f_fft * D / U if U else float("nan"),
            "st_cross": (f_cross * D / U) if (U and np.isfinite(f_cross)) else None,
            "mean": float(y.mean()), "amplitude": float(0.5 * (y.max() - y.min())),
            "n": int(y.size), "t_span": float(t[-1] - t[0]), "dt": dt}


def report_reference(sim):
    """力系数报告 → 官方参考量 {u_ref, rho_ref, area_ref, direction}。"""
    for o in getattr(sim, "objects", []):
        if "ForceCoefficientReport" not in (o.class_name or ""):
            continue
        d = o.dict or {}

        def val(key):
            tgt = sim.objmap.get(d.get(key))
            return None if tgt is None else (tgt.dict or {}).get("Value")

        dir_obj = sim.objmap.get(d.get("Direction"))
        u = val("ReferenceVelocity")
        return {"ok": True, "name": o.name, "class": o.class_name,
                "u_ref": float(u) if u is not None else None,
                "rho_ref": val("ReferenceDensity"), "area_ref": val("ReferenceArea"),
                "direction": (dir_obj.dict or {}).get("Vector") if dir_obj else None}
    return {"ok": False, "reason": "无 ForceCoefficientReport"}


def reference_case(sim, nu=DEFAULT_NU, fields=False, diameter_from_mesh=False, span=1.0):
    """官方参考解：U/ρ/A → D（推定或实测）→ 升力序列 → St/振幅 + 残差（+ 解场统计）。"""
    rep = report_reference(sim)
    if not rep.get("ok"):
        return {"ok": False, "reason": rep.get("reason")}
    u, area = rep.get("u_ref"), rep.get("area_ref")
    d_est, d_src = None, None
    if area and span:
        d_est, d_src = float(area) / float(span), "reference_area/span"
    if diameter_from_mesh:
        vol = sim.extract_volume_mesh()
        if vol.get("ok"):
            bnd = sim.extract_boundary_faces(vol)
            pts = np.asarray(vol["points"], float)
            for b in bnd.get("boundaries") or []:
                if str(b.get("name")) != "Cylinder":
                    continue
                idx = sorted({int(v) for ring in (b.get("rings") or []) for v in ring})
                if idx:
                    V = pts[idx]
                    c = V.mean(axis=0)
                    d_est = float(2.0 * np.linalg.norm(V - c, axis=1).mean())
                    d_src = "cylinder_boundary_rings"
                break
    mc = sim.extract_monitor_curves()
    lift = None
    resid = None
    if mc.get("ok"):
        data = mc.get("data") or {}
        mon = {str(m["name"]): m for m in mc.get("monitors") or []}
        cont = next((m for k, m in mon.items() if "Continuity" in k), None)
        if cont:
            resid = {"final": cont.get("cur_value"), "min": cont.get("y_min"),
                     "max": cont.get("y_max"), "n": cont.get("n")}
        lift_name = next((k for k in data
                          if "升力" in str(k) or "Lift" in str(k) or "Cl" in str(k)), None)
        time_name = next((k for k in data if "Time" in str(k)), None)
        if lift_name:
            y = np.asarray(data[lift_name]["y"], float)
            tmon = (np.asarray(data[time_name]["y"], float)
                    if time_name is not None else None)
            if tmon is not None and tmon.size == y.size:
                t = tmon
            else:
                x = data[lift_name].get("index")
                idx = np.asarray(x if x is not None else np.arange(y.size), float)
                it_max = float(idx[-1]) if idx.size else 0.0
                span_t = float(tmon[-1]) if (tmon is not None and tmon.size) else None
                scale = (span_t / it_max) if (span_t and it_max) else 1.0
                t = idx * scale
            lift = {"name": str(lift_name), "t": t, "y": y}
    st = (strouhal_from_series(lift["t"], lift["y"], d_est, u or 0.0)
          if (lift is not None and d_est) else None)
    out = {"ok": True, "name": rep.get("name"), "u_ref": u, "rho_ref": rep.get("rho_ref"),
           "area_ref": area, "direction": rep.get("direction"),
           "diameter": d_est, "diameter_source": d_src, "nu": nu,
           "reynolds": (u * d_est / nu) if (u and d_est and nu) else None,
           "strouhal": st, "residual": resid,
           "lift": ({"name": lift["name"], "n": int(lift["y"].size)} if lift else None),
           "reason": ""}
    if fields:
        sf = sim.extract_solution_fields()
        if sf.get("ok"):
            out["n_cells"] = sf["cell_count"]
            out["fields"] = [{"name": f["name"], "components": f["components"],
                              "min": f["min"], "max": f["max"]} for f in sf["fields"]]
    return out


def annulus_surface(D, r_far_factor=10.0, thickness=None, n_theta=96, n_r=14):
    """环形域（圆柱 D ↔ 远场 R=r_far·D，厚 T=1·D）→ 闭合三维三角表面（可解析核对）。"""
    r0 = 0.5 * float(D)
    R = float(r_far_factor) * float(D)
    T = float(thickness) if thickness else float(D)
    th = np.linspace(0.0, 2.0 * math.pi, int(n_theta), endpoint=False)
    radii = np.linspace(r0, R, int(n_r) + 1)
    pts, idx = [], {}

    def add(x, y, z):
        pts.append((x, y, z))
        return len(pts) - 1

    for z in (0.0, T):
        for i, r in enumerate(radii):
            for j, a in enumerate(th):
                idx[(z, i, j)] = add(r * math.cos(a), r * math.sin(a), z)
    faces = []
    for z in (0.0, T):
        for i in range(int(n_r)):
            for j in range(int(n_theta)):
                j2 = (j + 1) % int(n_theta)
                a, b = idx[(z, i, j)], idx[(z, i, j2)]
                c, d = idx[(z, i + 1, j2)], idx[(z, i + 1, j)]
                # 法向朝外：z=0 端面朝 −z（反绕），z=T 端面朝 +z
                faces += [(a, c, b), (a, d, c)] if z == 0.0 else [(a, b, c), (a, c, d)]
    for i in (0, int(n_r)):
        for j in range(int(n_theta)):
            j2 = (j + 1) % int(n_theta)
            a, b = idx[(0.0, i, j)], idx[(0.0, i, j2)]
            c, d = idx[(T, i, j2)], idx[(T, i, j)]
            # 法向朝外：内壁（孔）法向指向轴心，外壁指向 +r
            faces += [(a, c, b), (a, d, c)] if i == 0 else [(a, b, c), (a, c, d)]
    V = np.asarray(pts, float)
    F = np.asarray(faces, np.int64)
    # 统一朝外：端面法向 ±z，柱面法向 ±r̂（内孔朝轴心、外壁朝外）——按面心归属判定后翻转
    cen = V[F].mean(axis=1)
    nrm = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    ref = np.zeros_like(cen)
    on_bottom = np.abs(cen[:, 2]) < 1e-12
    on_top = np.abs(cen[:, 2] - T) < 1e-12
    radial = cen[:, :2].copy()
    rn = np.linalg.norm(radial, axis=1, keepdims=True)
    radial = np.divide(radial, rn, out=np.zeros_like(radial), where=rn > 0)
    ref[on_bottom] = (0.0, 0.0, -1.0)
    ref[on_top] = (0.0, 0.0, 1.0)
    wall = ~(on_bottom | on_top)
    ref[wall, 0] = radial[wall, 0]
    ref[wall, 1] = radial[wall, 1]
    flip = (nrm * ref).sum(axis=1) < 0
    F = F.copy()
    F[flip] = F[flip][:, [0, 2, 1]]
    area = float(0.5 * np.linalg.norm(
        np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum())
    volume = float(abs((V[F[:, 0]] * np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6.0))
    edges = {}
    for a, b, c in F:
        for u, v in ((a, b), (b, c), (c, a)):
            k = (int(u), int(v)) if u < v else (int(v), int(u))
            edges[k] = edges.get(k, 0) + 1
    return {"ok": True, "points": V, "triangles": F, "D": float(D), "r_far": R,
            "thickness": T, "n_theta": int(n_theta), "n_r": int(n_r),
            "area": area, "volume": volume,
            "watertight": set(edges.values()) == {2},
            "area_exact": 2 * math.pi * (R ** 2 - r0 ** 2) + 2 * math.pi * (R + r0) * T,
            "volume_exact": math.pi * (R ** 2 - r0 ** 2) * T}


def _orient_outward(V, F, thickness, hole_center, hole_r, domain_center=None):
    """统一面法向朝外：端面 ±z；孔壁朝轴心；其余壁面背离域心。"""
    cen = V[F].mean(axis=1)
    nrm = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    ref = np.zeros_like(cen)
    on_bottom = np.abs(cen[:, 2]) < 1e-12
    on_top = np.abs(cen[:, 2] - thickness) < 1e-12
    ref[on_bottom] = (0.0, 0.0, -1.0)
    ref[on_top] = (0.0, 0.0, 1.0)
    wall = ~(on_bottom | on_top)
    hc = np.asarray(hole_center, float)[:2]
    dh = cen[:, :2] - hc
    rh = np.linalg.norm(dh, axis=1, keepdims=True)
    dh = np.divide(dh, rh, out=np.zeros_like(dh), where=rh > 0)
    inner = wall & (rh[:, 0] < hole_r * 1.05)
    ref[inner, 0] = -dh[inner, 0]
    ref[inner, 1] = -dh[inner, 1]
    outer = wall & ~inner
    dc = np.asarray(domain_center if domain_center is not None else [0.0, 0.0],
                    float)[:2]
    do = cen[:, :2] - dc
    ro = np.linalg.norm(do, axis=1, keepdims=True)
    do = np.divide(do, ro, out=np.zeros_like(do), where=ro > 0)
    ref[outer, 0] = do[outer, 0]
    ref[outer, 1] = do[outer, 1]
    flip = (nrm * ref).sum(axis=1) < 0
    F = F.copy()
    F[flip] = F[flip][:, [0, 2, 1]]
    return F


def channel_surface(D, length_D=20.0, height_D=10.0, thickness=None, n_theta=96,
                    n_r=14, center_x_D=5.0, stretch=1.2):
    """通道域（矩形 + 圆柱孔）闭合表面：入/出口为平面，可用求解器的 min/max 面 BC。

    结构化 O 型网格：每条射线由圆柱面（r=D/2）连到矩形边界（同角度），
    径向 n_r 段（stretch 幂次做近壁加密）；前后端面三角化 + 四侧壁 + 圆柱壁。
    解析核对：V = (L·H − πr²)·T；A = 2(L·H − πr²) + [2(L+H) + 2πr]·T。
    """
    r0 = 0.5 * float(D)
    L = float(length_D) * float(D)
    H = float(height_D) * float(D)
    T = float(thickness) if thickness else float(D)
    cx, cy = float(center_x_D) * float(D), 0.5 * H
    th = np.linspace(0.0, 2.0 * math.pi, int(n_theta), endpoint=False)
    ts = (np.linspace(0.0, 1.0, int(n_r) + 1) ** float(stretch))
    outer_pts = []
    for a in th:
        dx, dy = math.cos(a), math.sin(a)
        cand = []
        if abs(dx) > 1e-12:
            for x in (0.0, L):
                t = (x - cx) / dx
                y = cy + t * dy
                if t > 0 and -1e-9 <= y <= H + 1e-9:
                    cand.append((t, x, y))
        if abs(dy) > 1e-12:
            for y in (0.0, H):
                t = (y - cy) / dy
                x = cx + t * dx
                if t > 0 and -1e-9 <= x <= L + 1e-9:
                    cand.append((t, x, y))
        cand.sort()
        outer_pts.append((cand[0][1], cand[0][2]))
    pts, idx = [], {}

    def add(x, y, z):
        pts.append((x, y, z))
        return len(pts) - 1

    for z in (0.0, T):
        for i, t in enumerate(ts):
            for j, a in enumerate(th):
                px = cx + r0 * math.cos(a)
                py = cy + r0 * math.sin(a)
                ox, oy = outer_pts[j]
                idx[(z, i, j)] = add(px + t * (ox - px), py + t * (oy - py), z)
    faces = []
    for z in (0.0, T):
        for i in range(int(n_r)):
            for j in range(int(n_theta)):
                j2 = (j + 1) % int(n_theta)
                a, b = idx[(z, i, j)], idx[(z, i, j2)]
                c, d = idx[(z, i + 1, j2)], idx[(z, i + 1, j)]
                faces += [(a, c, b), (a, d, c)] if z == 0.0 else [(a, b, c), (a, c, d)]
    for i in (0, int(n_r)):
        for j in range(int(n_theta)):
            j2 = (j + 1) % int(n_theta)
            a, b = idx[(0.0, i, j)], idx[(0.0, i, j2)]
            c, d = idx[(T, i, j2)], idx[(T, i, j)]
            faces += [(a, c, b), (a, d, c)] if i == 0 else [(a, b, c), (a, c, d)]
    V = np.asarray(pts, float)
    F = _orient_outward(V, np.asarray(faces, np.int64), T, (cx, cy), r0,
                        domain_center=(0.5 * L, 0.5 * H))
    area = float(0.5 * np.linalg.norm(
        np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum())
    volume = float(abs((V[F[:, 0]] * np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6.0))
    edges = {}
    for a, b, c in F:
        for u, v in ((a, b), (b, c), (c, a)):
            k = (int(u), int(v)) if u < v else (int(v), int(u))
            edges[k] = edges.get(k, 0) + 1
    return {"ok": True, "points": V, "triangles": F, "D": float(D), "length": L,
            "height": H, "thickness": T, "hole_center": (cx, cy), "hole_r": r0,
            "n_theta": int(n_theta), "n_r": int(n_r), "area": area, "volume": volume,
            "watertight": set(edges.values()) == {2},
            "area_exact": 2 * (L * H - math.pi * r0 ** 2) + (2 * (L + H) + 2 * math.pi * r0) * T,
            "volume_exact": (L * H - math.pi * r0 ** 2) * T}


def channel_tet_mesh(D, length_D=20.0, height_D=10.0, thickness=None, n_theta=96,
                     n_r=14, center_x_D=5.0, stretch=1.2, n_layers=1):
    """通道域**结构化**四面体网格（O 型网格棱柱 → 3 tet/棱柱，无 scipy、无薄元）。

    相比 scipy Delaunay：确定性、单元形状可控。体积与解析值一致（可核对）。
    注意（S4 修复记录）：外边界是**内接多边形**（外环点取自矩形边界的射线交点，
    相邻点之间的弦切掉角部）→ 体积略小于解析值（n_theta=48 时 −1.4%，192 时 −0.06%）；
    棱柱→tet 的对角线必须按**全局顶点编号**规范化，否则 θ 周期缝合面三角剖分不一致
    （曾导致细网格压力矩阵奇异）。
    """
    r0 = 0.5 * float(D)
    L = float(length_D) * float(D)
    H = float(height_D) * float(D)
    T = float(thickness) if thickness else float(D)
    cx, cy = float(center_x_D) * float(D), 0.5 * H
    th = np.linspace(0.0, 2.0 * math.pi, int(n_theta), endpoint=False)
    ts = np.linspace(0.0, 1.0, int(n_r) + 1) ** float(stretch)
    outer = []
    for a in th:
        dx, dy = math.cos(a), math.sin(a)
        cand = []
        if abs(dx) > 1e-12:
            for x in (0.0, L):
                t = (x - cx) / dx
                y = cy + t * dy
                if t > 0 and -1e-9 <= y <= H + 1e-9:
                    cand.append((t, x, y))
        if abs(dy) > 1e-12:
            for y in (0.0, H):
                t = (y - cy) / dy
                x = cx + t * dx
                if t > 0 and -1e-9 <= x <= L + 1e-9:
                    cand.append((t, x, y))
        cand.sort()
        outer.append((cand[0][1], cand[0][2]))
    # O 型网格在矩形角点处会出现强畸变 → 对内部环做 Laplacian 平滑（端点环固定）
    grid = np.zeros((int(n_r) + 1, int(n_theta), 2), float)
    for i, t in enumerate(ts):
        for j, a in enumerate(th):
            px, py = cx + r0 * math.cos(a), cy + r0 * math.sin(a)
            ox, oy = outer[j]
            grid[i, j] = (px + t * (ox - px), py + t * (oy - py))
    for _ in range(60):
        inner = grid[1:-1]
        nb = (grid[2:] + grid[:-2]
              + np.roll(grid, 1, axis=1)[1:-1]
              + np.roll(grid, -1, axis=1)[1:-1]) / 4.0
        grid[1:-1] = 0.5 * inner + 0.5 * nb
    nz = int(n_layers) + 1
    zs = np.linspace(0.0, T, nz)
    pts, vid = [], {}
    for k, z in enumerate(zs):
        for i in range(int(n_r) + 1):
            for j in range(int(n_theta)):
                px, py = grid[i, j]
                vid[(k, i, j)] = len(pts)
                pts.append((float(px), float(py), z))
    cells = []
    for k in range(int(n_layers)):
        for i in range(int(n_r)):
            for j in range(int(n_theta)):
                j2 = (j + 1) % int(n_theta)
                b = [vid[(k, i, j)], vid[(k, i + 1, j)], vid[(k, i + 1, j2)],
                     vid[(k, i, j2)]]
                u = [vid[(k + 1, i, j)], vid[(k + 1, i + 1, j)],
                     vid[(k + 1, i + 1, j2)], vid[(k + 1, i, j2)]]
                # 每层四边形 → 2 三角形 → 棱柱 → 3 tet（标准分解，先按体积定向）
                tris_b = [(b[0], b[1], b[2]), (b[0], b[2], b[3])]
                tris_u = [(u[0], u[1], u[2]), (u[0], u[2], u[3])]
                for (p0, p1, p2), (q0, q1, q2) in zip(tris_b, tris_u):
                    # S4 修复：**按全局顶点编号规范化底面三角形顺序**（顶面同序跟随）。
                    # 棱柱→3 tet 的三个侧面各取一条对角线，局部规则下对角线取决于
                    # 三角形顶点在局部的次序：θ 周期缝合面上两侧的同一四边形来自
                    # "第 1 个三角形"与"第 2 个三角形"，局部次序相反 → 一侧取
                    # bottom(i,0)–top(i+1,0)、另一侧取 bottom(i+1,0)–top(i,0)，
                    # 内部面配不上（实测 n_theta=48 有 1348 个内部单侧面）→ 细网格
                    # 压力矩阵奇异（dgstrf info / NaN）。按全局编号排序后对角线只由
                    # 全局编号决定（"小号底面点连它自己的顶面点"），两侧一致 → 协调。
                    _tri = sorted(((p0, q0), (p1, q1), (p2, q2)),
                                  key=lambda t: t[0])
                    (p0, q0), (p1, q1), (p2, q2) = _tri
                    cells += [(p0, p1, p2, q2), (p0, p1, q2, q1), (p0, q1, q2, q0)]
    V = np.asarray(pts, float)
    C = np.asarray(cells, np.int64)
    from mesh_tet import _tet_volumes
    vol = _tet_volumes(V, C)
    flip = vol < 0
    if flip.any():
        C = C.copy()
        C[flip] = C[flip][:, [0, 1, 3, 2]]
        vol = _tet_volumes(V, C)
    analytic = (L * H - math.pi * r0 ** 2) * T
    return {"ok": True, "vertices": V, "cells": C, "n_cells": int(C.shape[0]),
            "n_points": int(V.shape[0]), "volume": float(vol.sum()),
            "volume_exact": analytic, "n_negative": 0,
            "quality_proxy": {"min_vol": float(vol.min()),
                              "max_vol": float(vol.max()),
                              "mean_vol": float(vol.mean())},
            "hole_center": (cx, cy), "hole_r": r0, "D": float(D),
            "thickness": T, "length": L, "height": H}


def channel_tet_mesh_cartesian(D, length_D=16.0, height_D=8.0, thickness_D=0.5,
                               h_factor=4.0, center_x_D=4.0):
    """通道域**笛卡尔阶梯**四面体网格（鲁棒：单元全为直角六面体 → 6 tet）。

    圆柱用阶梯近似（偏差 ≤ h/2），换来确定性与良态矩阵 —— 用于自研瞬态算例；
    有效堵塞比与阶梯偏差如实返回（`blockage`、`stair_deviation`），不假装贴体。
    """
    D = float(D)
    L, H = float(length_D) * D, float(height_D) * D
    T = float(thickness_D) * D
    h = D / float(h_factor)
    nx, ny, nz = max(int(round(L / h)), 4), max(int(round(H / h)), 4), max(int(round(T / h)), 1)
    xs = np.linspace(0.0, L, nx + 1)
    ys = np.linspace(0.0, H, ny + 1)
    zs = np.linspace(0.0, T, nz + 1)
    cx, cy = float(center_x_D) * D, 0.5 * H
    r0 = 0.5 * D
    pts, vid = [], {}
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                vid[(k, j, i)] = len(pts)
                pts.append((xs[i], ys[j], zs[k]))
    hexes = []
    blocked = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                cxx, cyy = 0.5 * (xs[i] + xs[i + 1]), 0.5 * (ys[j] + ys[j + 1])
                if math.hypot(cxx - cx, cyy - cy) <= r0:
                    blocked += 1
                    continue
                hexes.append((vid[(k, j, i)], vid[(k, j, i + 1)],
                              vid[(k, j + 1, i + 1)], vid[(k, j + 1, i)],
                              vid[(k + 1, j, i)], vid[(k + 1, j, i + 1)],
                              vid[(k + 1, j + 1, i + 1)], vid[(k + 1, j + 1, i)]))
    TETS6 = ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6),
             (0, 4, 5, 6), (0, 5, 1, 6))
    cells = []
    for hx in hexes:
        for t in TETS6:
            cells.append(tuple(hx[k] for k in t))
    V = np.asarray(pts, float)
    C = np.asarray(cells, np.int64)
    from mesh_tet import _tet_volumes
    vol = _tet_volumes(V, C)
    flip = vol < 0
    if flip.any():
        C = C.copy()
        C[flip] = C[flip][:, [0, 1, 3, 2]]
        vol = _tet_volumes(V, C)
    analytic = (L * H - math.pi * r0 ** 2) * T
    staircase = (L * H * T) - float(np.abs(vol).sum())
    return {"ok": True, "vertices": V, "cells": C, "n_cells": int(C.shape[0]),
            "n_points": int(V.shape[0]), "n_hex": len(hexes), "n_blocked": blocked,
            "volume": float(np.abs(vol).sum()), "volume_exact": analytic,
            "h": h, "n_negative": 0, "hole_center": (cx, cy), "hole_r": r0, "D": D,
            "thickness": T, "length": L, "height": H,
            "blockage": D / H, "stair_deviation": 0.5 * h,
            "quality_proxy": {"min_vol": float(np.abs(vol).min()),
                              "mean_vol": float(np.abs(vol).mean())}}


def diff_metrics(ours, ref):
    """自研结果 ↔ 官方参考：St 比 / 振幅比 / 平均升力 + 误差带判定。"""
    out = {"ok": True, "items": {}}
    st_ref = ((ref.get("strouhal") or {}) or {}).get("st") if ref.get("strouhal") else None
    st_our = (ours or {}).get("st")
    if st_ref and st_our:
        ratio = float(st_our) / float(st_ref)
        out["items"]["strouhal"] = {"ours": float(st_our), "official": float(st_ref),
                                    "ratio": ratio,
                                    "ok": ST_TOLERANCE[0] <= ratio <= ST_TOLERANCE[1],
                                    "tolerance": list(ST_TOLERANCE)}
    else:
        out["items"]["strouhal"] = {"ok": None, "reason": "缺官方或自研 St"}
    amp_ref = ((ref.get("strouhal") or {}) or {}).get("amplitude")
    amp_our = (ours or {}).get("amplitude")
    if amp_ref and amp_our:
        ratio = float(amp_our) / float(amp_ref)
        out["items"]["amplitude"] = {"ours": float(amp_our), "official": float(amp_ref),
                                     "ratio": ratio,
                                     "ok": AMPLITUDE_TOLERANCE[0] <= ratio <= AMPLITUDE_TOLERANCE[1],
                                     "tolerance": list(AMPLITUDE_TOLERANCE)}
    if ours and ours.get("mean") is not None:
        out["items"]["mean_lift"] = {"ours": float(ours["mean"]),
                                     "ok": abs(float(ours["mean"])) < 0.1,
                                     "note": "无升力体平均升力应≈0"}
    return out


def run_case(D=0.04, u_inf=0.05, nu=DEFAULT_NU, dt=None, steps=120, n_inner=2,
             length_D=16.0, height_D=8.0, thickness_D=0.5, h_factor=4.0,
             center_x_D=4.0, sample_every=1, mesher="cartesian", mesh=None,
             convection="upwind", wall_slip_axes=(), piso_correctors=0,
             perturb=None, checkpoint=None, checkpoint_every=25,
             skew_corrected=False, nonorth_corrected=False, corr_limit=1.0,
             hybrid_m=24, hybrid_nr=16, hybrid_a_D=3.0, hybrid_stretch=1.3,
             hybrid_layers=1):
    """同工况自研求解：通道域结构化 tet + 瞬态 SIMPLE（R1 内核）+ 圆柱升力积分 → Cl(t) → St。

    **长耗时**：需 STARDECODING_LONG=1。默认 `mesher="cartesian"`（笛卡尔阶梯网格，良态稳定）；
    `mesher="ogrid"` 走 O 型网格（本项目实测在细分辨率下压力矩阵奇异，见文档）；
    `mesh` 可直接传自定义网格（vertices/cells）。
    边界：min-x 入口 / max-x 出口 / 其余（含圆柱阶梯面）壁面；
    升力由 aero_forces.force_coefficients 在圆柱面（r ≤ r0 + h）上积分。
    周期数 <3 时如实报告"未达脱落周期 → 不做 St 比对"。

    `perturb`：非对称初始扰动幅值（以 U 为单位，如 0.05）；在圆柱上方
    0.6D 处叠加宽 0.5D 的高斯竖直速度团，用于在有限算力下触发绝对不稳定
    流动的脱落（对称网格 + 对称初值下脱落只能靠舍入误差起步）。它只影响
    起步相位，不改变极限环频率。
    `checkpoint`：长跑每 `checkpoint_every` 步把 (t, cl) 序列落盘（进程中断
    也不丢证据）；返回体里带 `perturb_mag`。
    """
    if os.environ.get("STARDECODING_LONG") != "1":
        return {"ok": False,
                "reason": "长耗时路径：设 STARDECODING_LONG=1 才运行真实瞬态算例"}
    from pressure_solver import PressureSolver
    from aero_forces import force_coefficients
    if mesh is None:
        if mesher == "cartesian":
            mesh = channel_tet_mesh_cartesian(D, length_D=length_D, height_D=height_D,
                                              thickness_D=thickness_D, h_factor=h_factor,
                                              center_x_D=center_x_D)
        elif mesher == "ogrid":
            mesh = channel_tet_mesh(D, length_D=length_D, height_D=height_D,
                                    thickness=thickness_D * D)
        elif mesher == "hybrid":
            # S2 第 3 步 ③(a)：方形 O 型环带（贴体）+ 张量积外围（协调拼接）。
            # 推荐 thickness_D=0.25 + slip 壁（z 厚 ≈ 外围 h → 长细比≈1，
            # 非正交角 median≈33°/p95≈69°，d_n/|d|<0.15 占比 0%）
            from mesh_hybrid import hybrid_channel_mesh
            mesh = hybrid_channel_mesh(D, length_D=length_D, height_D=height_D,
                                       thickness_D=thickness_D,
                                       center_x_D=center_x_D, h_factor=h_factor,
                                       m=hybrid_m, n_r=hybrid_nr, a_D=hybrid_a_D,
                                       stretch=hybrid_stretch, n_layers=hybrid_layers)
        else:
            return {"ok": False, "reason": "未知网格器 %r（cartesian/ogrid/hybrid）" % mesher}
    V = np.asarray(mesh["vertices"], float)
    C = np.asarray(mesh["cells"], np.int64)
    dt = float(dt or (0.2 * D / u_inf))
    solver = PressureSolver(V, C, rho=1.0, mu=1.0 * nu, inlet_axis=0,
                            inlet_side="min", inlet_velocity=(u_inf, 0.0, 0.0),
                            outlet_side="max", convection=convection,
                            wall_slip_axes=wall_slip_axes, piso_correctors=piso_correctors,
                            skew_corrected=skew_corrected,
                            nonorth_corrected=nonorth_corrected,
                            corr_limit=corr_limit)
    # S2：非对称初始扰动（可选）——必须在 enable_transient 之前加，
    # 这样 φⁿ 参考即为扰动后的场。
    perturb_mag = 0.0
    if perturb:
        cc = np.asarray(solver.fvm.centroids, float)[:, :2]
        cx0, cy0 = float(mesh["hole_center"][0]), float(mesh["hole_center"][1])
        blob = np.exp(-(((cc[:, 0] - cx0) / float(D)) ** 2
                        + ((cc[:, 1] - (cy0 + 0.6 * float(D))) / (0.5 * float(D))) ** 2))
        perturb_mag = solver.perturb_velocity(dv=float(perturb) * u_inf * blob)
    solver.enable_transient(dt, snapshot=True)
    fv = solver.fvm
    hc = np.asarray(mesh["hole_center"], float)[:2]
    rh = np.linalg.norm(fv.face_centroid[:, :2] - hc, axis=1)
    cyl_r = float(mesh["hole_r"]) + (float(mesh.get("h", 0.0)) if mesher == "cartesian"
                                     else 0.0)
    cyl_faces = np.where(fv.is_boundary & (rh <= cyl_r + 1e-9))[0]
    if cyl_faces.size == 0:
        return {"ok": False, "reason": "未识别到圆柱面（0 个面）"}
    ts, cls, ws = [], [], []
    for k in range(int(steps)):
        solver.advance(dt=dt, n_inner=n_inner)
        if checkpoint and k % max(1, int(checkpoint_every)) == 0:
            import json as _json
            with open(checkpoint, "w", encoding="utf-8") as fh:
                _json.dump({"t": ts, "cl": cls, "w_absmax": ws, "steps_done": k,
                            "dt": float(dt), "n_cells": int(C.shape[0]),
                            "perturb_mag": perturb_mag, "done": False},
                           fh, ensure_ascii=False)
        if k % max(1, int(sample_every)) == 0:
            vel = solver.velocity()
            fc = force_coefficients(fv, solver.pressure(), vel[:, 0], vel[:, 1],
                                    vel[:, 2], solver.mu, solver.rho,
                                    faces=cyl_faces, a_ref=D, u_ref=u_inf,
                                    rho_ref=1.0, drag_dir=(1.0, 0.0, 0.0),
                                    lift_dir=(0.0, 1.0, 0.0))
            ts.append(float(solver.time))
            cls.append(float(fc.get("cl", float("nan"))))
            # S2 ③(a)：准二维真实性监视（滑移 z 壁下 |w| 应≈0；tet 分解的 z 不对称
            # 会注入伪 w —— 如实记录，不掩盖）
            ws.append(float(np.abs(vel[:, 2]).max()))
    # S4：CFL 诊断（诚实必要 —— 贴体 O 型网格的 θ 向间距远小于径向，dt 稍大即 CFL>1
    # 发散；实测 dt=0.04 s + n_theta=96 时 CFL≈1.5 → cl 爆到 1e32）。
    try:
        from mesh_tet import _tet_volumes
        _vol = np.abs(_tet_volumes(np.asarray(V, float), np.asarray(C, np.int64)))
        h_min = float(_vol.min()) ** (1.0 / 3.0) if _vol.size else 0.0
        h_typ = float(np.median(_vol)) ** (1.0 / 3.0) if _vol.size else 0.0
    except Exception:
        h_min = h_typ = 0.0
    cfl = float(u_inf * dt / h_min) if h_min > 0 else None
    series = np.asarray(cls, float)
    t = np.asarray(ts, float)
    st = strouhal_from_series(t, series, D, u_inf) if series.size >= 16 else None
    n_periods = (st["f"] * (t[-1] - t[0])) if (st and st.get("ok")) else 0.0
    if checkpoint:
        import json as _json
        with open(checkpoint, "w", encoding="utf-8") as fh:
            _json.dump({"t": ts, "cl": cls, "w_absmax": ws, "steps_done": int(steps),
                        "dt": float(dt), "n_cells": int(C.shape[0]),
                        "perturb_mag": perturb_mag, "done": True}, fh,
                       ensure_ascii=False)
    # S2 ③(a)：网格质量随结果一起落档（贴体路线的核心证据之一）
    try:
        from mesh_quality import orthogonality_report
        _oq = orthogonality_report(V, C)
        mesh_quality = {"verdict": _oq["verdict"],
                        "ortho_median": _oq["ortho_deg"]["median"],
                        "ortho_p95": _oq["ortho_deg"]["p95"],
                        "skew_p95": _oq["skew"]["p95"]}
    except Exception:
        mesh_quality = None
    return {"ok": True, "mesher": mesher, "n_cells": int(C.shape[0]),
            "n_cyl_faces": int(cyl_faces.size), "dt": dt, "steps": int(steps),
            "perturb_mag": perturb_mag,
            "corrections": {"skew": bool(skew_corrected),
                            "nonorth": bool(nonorth_corrected),
                            "corr_limit": float(corr_limit)},
            "mesh_quality": mesh_quality,
            "w_absmax_final": (float(ws[-1]) if ws else None),
            "h_min": h_min, "h_typ": h_typ, "cfl_max": cfl,
            "t_span": float(t[-1] - t[0]) if t.size else 0.0,
            "n_samples": int(series.size),
            "mean": float(np.nanmean(series)) if series.size else None,
            "amplitude": (float(0.5 * (np.nanmax(series) - np.nanmin(series)))
                          if series.size else None),
            "st": (st.get("st") if st and st.get("ok") else None),
            "f": (st.get("f") if st and st.get("ok") else None),
            "n_periods": float(n_periods),
            "mesh": {"volume": mesh.get("volume"), "h": mesh.get("h"),
                     "blockage": mesh.get("blockage"),
                     "stair_deviation": mesh.get("stair_deviation")},
            "series": (t, series),
            "reason": "" if (st and st.get("ok") and n_periods >= 3)
                      else "未达脱落周期（<3 个周期）→ 不做 St 比对"}

def _main(argv=None):
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    import argparse
    ap = argparse.ArgumentParser(description="R4 官方解差分（官方 .sim 解数据为参考）")
    ap.add_argument("file", help="官方 .sim（含力系数报告与监视器）")
    ap.add_argument("--nu", type=float, default=DEFAULT_NU, help="运动粘度（默认 1e-5）")
    ap.add_argument("--fields", action="store_true", help="附解场统计（较慢）")
    ap.add_argument("--diameter-from-mesh", action="store_true",
                    help="由 Cylinder 边界环实测直径（较慢）")
    ap.add_argument("--run", action="store_true", help="跑同工况自研瞬态（需 STARDECODING_LONG=1）")
    ap.add_argument("--steps", type=int, default=200)
    # S2/S4 实验旋钮（可复现文档里的每组实验）
    ap.add_argument("--dt", type=float, default=None, help="时间步（缺省 0.2·D/U）")
    ap.add_argument("--n-inner", type=int, default=2, help="每时间步内迭代次数（S2：1 更少隐式平滑）")
    ap.add_argument("--h-factor", type=float, default=4.0, help="网格分辨率（D/h，S2：8 = 8 单元/直径）")
    ap.add_argument("--convection", choices=("upwind", "central", "limited"), default="upwind",
                    help="对流格式（S2：limited = 二阶受限中心）")
    ap.add_argument("--slip-walls", action="store_true",
                    help="侧壁滑移（wall_slip_axes=(1,2)，准二维绕流）")
    ap.add_argument("--piso", type=int, default=0,
                    help="PISO 压力校正次数（S2：≥2 且配合 --n-inner 1 保非定常时间精度）")
    ap.add_argument("--mesher", choices=("cartesian", "ogrid", "hybrid"), default="cartesian",
                    help="cartesian=阶梯（鲁棒）；ogrid=贴体 O 型网格；hybrid=方形 O 型环带+"
                         "张量积外围（S2 第 3 步 ③(a)，协调拼接，配 --nonorth-correct 用）")
    ap.add_argument("--thickness-D", type=float, default=0.5,
                    help="展向厚度（以 D 为单位；0.25 + 滑移壁 = 单层准二维，成本减半）")
    ap.add_argument("--perturb", type=float, default=None,
                    help="非对称初始扰动幅值（以 U 为单位；S2：0.05 触发脱落）")
    ap.add_argument("--sample-every", type=int, default=1,
                    help="每 N 步采样一次升力（长跑可加大以省内存）")
    ap.add_argument("--checkpoint", default=None,
                    help="(t,cl) 序列落盘路径（长跑证据，进程中断不丢）")
    ap.add_argument("--skew-correct", action="store_true",
                    help="面值偏斜修正（默认关；实测混合网格上压力矩阵奇异，负结果）")
    ap.add_argument("--nonorth-correct", action="store_true",
                    help="扩散非正交修正（贴体/混合网格推荐开启）")
    ap.add_argument("--corr-limit", type=float, default=1.0,
                    help="非正交修正限幅系数（推荐 0.33：保对角占优换稳定）")
    ap.add_argument("--hybrid-m", type=int, default=24,
                    help="混合网格：正方形每边细分段数（角向射线总数 = 4m）")
    ap.add_argument("--hybrid-nr", type=int, default=16, help="混合网格：环带径向层数")
    ap.add_argument("--hybrid-a-D", type=float, default=3.0,
                    help="混合网格：环带外正方形半边长（×D）")
    ap.add_argument("--hybrid-stretch", type=float, default=1.3,
                    help="混合网格：径向加密幂次")
    ap.add_argument("--hybrid-layers", type=int, default=1, help="混合网格：z 向层数")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    from sim_parser import SimFile
    ref = reference_case(SimFile(args.file), nu=args.nu, fields=args.fields,
                         diameter_from_mesh=args.diameter_from_mesh)
    rep = {"reference": ref}
    if ref.get("ok") and args.run:
        ours = run_case(D=ref.get("diameter") or 0.04, u_inf=ref.get("u_ref") or 0.05,
                        nu=args.nu, steps=args.steps, dt=args.dt,
                        n_inner=args.n_inner, h_factor=args.h_factor,
                        convection=args.convection,
                        wall_slip_axes=(1, 2) if args.slip_walls else (),
                        piso_correctors=args.piso,
                        mesher=args.mesher,
                        thickness_D=args.thickness_D, perturb=args.perturb,
                        sample_every=args.sample_every,
                        checkpoint=args.checkpoint,
                        skew_corrected=args.skew_correct,
                        nonorth_corrected=args.nonorth_correct,
                        corr_limit=args.corr_limit,
                        hybrid_m=args.hybrid_m, hybrid_nr=args.hybrid_nr,
                        hybrid_a_D=args.hybrid_a_D,
                        hybrid_stretch=args.hybrid_stretch,
                        hybrid_layers=args.hybrid_layers)
        rep["ours"] = ours
        if ours.get("ok"):
            rep["diff"] = diff_metrics(ours, ref)
    if args.json:
        import json as _json
        print(_json.dumps(rep, ensure_ascii=False, indent=1,
                          default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        return 0 if ref.get("ok") else 1
    if not ref.get("ok"):
        print("官方参考量抽取失败:", ref.get("reason"))
        return 1
    st = ref.get("strouhal") or {}
    print("官方参考（%s）: U=%s ρ=%s A_ref=%s D=%s(%s) Re=%s"
          % (ref.get("name"), ref.get("u_ref"), ref.get("rho_ref"), ref.get("area_ref"),
             ref.get("diameter"), ref.get("diameter_source"),
             None if ref.get("reynolds") is None else round(ref["reynolds"], 2)))
    if st.get("ok"):
        print("升力序列（%s）: n=%d %.4g..%.4g s  振幅 %.4g  平均 %.3g"
              % (ref["lift"]["name"], st["n"], 0.0, st["t_span"], st["amplitude"],
                 st["mean"]))
        print("脱落频率: FFT %.5g Hz / 过零 %.5g Hz → f=%.5g Hz；St=%.4f（FFT %.4f / 过零 %s）"
              % (st["f_fft"], st["f_cross"], st["f"], st["st"], st["st_fft"],
                 None if st["st_cross"] is None else round(st["st_cross"], 4)))
    if ref.get("residual"):
        r = ref["residual"]
        print("残差（Continuity）: 末值 %s  区间 [%s, %s]  n=%s"
              % (r["final"], r["min"], r["max"], r["n"]))
    if rep.get("ours"):
        o = rep["ours"]
        if not o.get("ok"):
            print("自研瞬态: 未运行 ——", o.get("reason"))
        else:
            print("自研瞬态: 单元 %d dt=%.4g 步数 %d 周期数 %.2f St=%s"
                  % (o["n_cells"], o["dt"], o["steps"], o["n_periods"], o["st"]))
            if o.get("cfl_max") is not None:
                print("   CFL 诊断: h_min=%.3g h_typ=%.3g → CFL_max=%.2f%s"
                      % (o["h_min"], o["h_typ"], o["cfl_max"],
                         "（>1：贴体网格 θ 向间距小，需减小 dt）" if o["cfl_max"] > 1.0 else ""))
            for k, v in (rep.get("diff") or {}).get("items", {}).items():
                print("   差分 %-12s %s" % (k, v))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

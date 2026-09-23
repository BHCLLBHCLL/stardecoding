# -*- coding: utf-8 -*-
"""S6：同网格官方对照驱动（我方 CGNS → 官方 STAR-CCM+ 同网格稳态 → 场级对比）。

三段（可独立重跑，产物都在 s6_samemesh/）：
  prepare  建网格（cartesian 阶梯，run_case 同参）→ 我方稳态 SIMPLE 解 → Cd/Δp →
           写 CGNS（ZoneBC 边界 patch）→ ours.npz 落盘
  official official_mesh_case：官方外壳内 importFile → 建区/物理/边界 →
           稳态求解（轮询到目标步）→ Save As → official.sim + official_result.json
  compare  解析官方 .sim 解场（FvRepresentation）→ cKDTree 质心匹配 →
           场级对比（u/v/w/p：L2/中位/最大）+ 双方同码算 Cd/Δp → samemesh_report.json

诚实边界：官方侧未跑成（license/CGNS 拒收/宏失败）时 compare 如实拒绝并给原因，
不伪造对比；解算残差/迭代数双向落档。
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(ROOT, "s6_samemesh")
SHELL = r"D:\training\starccm\startutorialsdata\mesh\data\directedMeshCAD.sim"

# 工况（与 official_diff.run_case 同源）：通道域圆柱绕流，Re=200
D, U_INF, NU, RHO = 0.04, 0.05, 1e-5, 1.0
MESH_KW = dict(length_D=16.0, height_D=8.0, thickness_D=0.5, h_factor=4.0,
               center_x_D=4.0)
STEADY_MAX_ITERS = 600     # 我方稳态 SIMPLE 外迭代上限
STEADY_TOL = 1e-6          # 我方稳态收敛阈（连续性残差比）
OFFICIAL_ITERS = 600       # STAR 稳态目标迭代步
MATCH_TOL_FACTOR = 0.25    # 质心匹配容差 = h × factor（同网格应 ~0）


def _mesh_and_solver():
    from official_diff import channel_tet_mesh_cartesian
    from pressure_solver import PressureSolver
    mesh = channel_tet_mesh_cartesian(D, **MESH_KW)
    V = np.asarray(mesh["vertices"], float)
    C = np.asarray(mesh["cells"], np.int64)
    solver = PressureSolver(V, C, rho=RHO, mu=NU, inlet_axis=0,
                            inlet_side="min", inlet_velocity=(U_INF, 0.0, 0.0),
                            outlet_side="max", convection="upwind")
    return mesh, solver


def _cyl_faces(fv, mesh):
    """圆柱面识别（与 run_case 同口径）：r ≤ hole_r + h（阶梯 pad）。"""
    hc = np.asarray(mesh["hole_center"], float)[:2]
    rh = np.linalg.norm(np.asarray(fv.face_centroid, float)[:, :2] - hc, axis=1)
    cyl_r = float(mesh["hole_r"]) + float(mesh.get("h", 0.0) or 0.0)
    return np.where(np.asarray(fv.is_boundary, bool) & (rh <= cyl_r + 1e-9))[0]


def _dp(centroids_x, p, h):
    """入口/出口段（各 4h）单元平均压差 —— 双方同码。"""
    x = np.asarray(centroids_x, float)
    p = np.asarray(p, float)
    seg = 4.0 * h
    return float(p[x <= x.min() + seg].mean() - p[x >= x.max() - seg].mean())


def do_prepare():
    from aero_forces import force_coefficients
    from official_diff import channel_boundary_patches
    from postprocess import write_cgns
    mesh, solver = _mesh_and_solver()
    hist = []
    for k in range(STEADY_MAX_ITERS):
        r = solver.step()
        hist.append(float(r["residual"]))
        if r["residual"] <= STEADY_TOL:
            break
    fv = solver.fvm
    h = float(mesh["h"])
    patches = channel_boundary_patches(fv, mesh)
    vel = solver.velocity()
    p = solver.pressure()
    cen = np.asarray(fv.centroids, float)
    fc = force_coefficients(fv, p, vel[:, 0], vel[:, 1], vel[:, 2],
                            solver.mu, solver.rho, faces=_cyl_faces(fv, mesh),
                            a_ref=D, u_ref=U_INF, rho_ref=RHO,
                            drag_dir=(1.0, 0.0, 0.0), lift_dir=(0.0, 1.0, 0.0))
    dp = _dp(cen[:, 0], p, h)
    os.makedirs(WORK, exist_ok=True)
    cgns = os.path.join(WORK, "same_mesh.cgns")
    write_cgns(cgns, fv, patches=patches)
    np.savez(os.path.join(WORK, "ours.npz"), centroids=cen,
             u=vel[:, 0], v=vel[:, 1], w=vel[:, 2], p=p,
             n_cells=int(fv.n_cells), h=h,
             iterations=len(hist),
             residual_final=float(hist[-1]) if hist else float("nan"),
             residual_hist=np.asarray(hist, float),
             cd=float(fc.get("cd", float("nan"))),
             cl=float(fc.get("cl", float("nan"))), dp=dp)
    print("prepare: cells=%d patches=%s" % (fv.n_cells,
                                            [(q["name"], len(q["faces"])) for q in patches]))
    print("prepare: steady iters=%d residual=%.3e cd=%.4f cl=%.4f dp=%.4f"
          % (len(hist), hist[-1], fc.get("cd", float("nan")),
             fc.get("cl", float("nan")), dp))
    print("prepare: cgns=%s (%.1f KB)" % (cgns, os.path.getsize(cgns) / 1024.0))
    return 0


def do_official(timeout=7200):
    from star_bridge import official_mesh_case
    cgns = os.path.join(WORK, "same_mesh.cgns")
    out = os.path.join(WORK, "official.sim")
    res = official_mesh_case(SHELL, cgns, out, rho=RHO, mu=NU, u_in=U_INF,
                             max_iterations=OFFICIAL_ITERS, timeout=timeout)
    with open(os.path.join(WORK, "official_log.txt"), "w", encoding="utf-8") as f:
        f.write(res.get("log") or "")
    summary = {k: v for k, v in res.items() if k != "log"}
    with open(os.path.join(WORK, "official_result.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=str)
    print("official: ok=%s fail_reason=%s" % (res.get("ok"), res.get("fail_reason")))
    print("official: parts_new=%s region=%s boundaries=%s iter=%s"
          % (res.get("parts_new"), res.get("region"), res.get("boundaries"),
             res.get("iterations")))
    if not res.get("ok") and not res.get("skipped"):
        tail = (res.get("log") or "")[-1500:]
        if tail:
            print("--- log tail ---\n%s" % tail)
    return 0 if (res.get("ok") or res.get("skipped")) else 1


def _l2(a):
    """BLAS-free L2 范数（展平 Frobenius 同 np.linalg.norm(a)）。

    实测 occ 环境 numpy 2.5.2 的 dot/BLAS 路径在大数组（≥~1.5 万元素）上
    触发 delay-load 异常 0xC06D007F（KERNELBASE RaiseException，事件日志
    APPCRASH 实证）——norm(axis=None) 走 x.dot(x)，故用平方和归约替代。
    """
    a = np.asarray(a, float)
    return float(np.sqrt(np.square(a).sum()))


def _pick_field(sf, comps, hints):
    """按提示名优先挑字段，找不到再退而求其次（同名取第一个）。"""
    names = [f["name"] for f in sf["fields"] if f["components"] == comps]
    for h in hints:
        for n in names:
            if h.lower() in str(n).lower():
                return n
    return names[0] if names else None


def _velocity_field(sf):
    """组装官方速度场 (N,3)：优先 3 分量矢量字段，否则 U/V/W 标量三元组。

    官方 Save 的 FvRepresentation 常把速度拆成 U_Velocity/V_Velocity/
    W_Velocity 三个 1 分量场——若此时回退取首个 3 分量场（如
    ApparentPressureGradient）会把压力梯度当速度（实测坑），故显式拒绝。
    """
    for f in sf["fields"]:
        if f["components"] == 3:
            n = str(f["name"]).lower()
            if "velocity" in n or "vel" in n:
                return f["name"], np.asarray(sf["data"][f["name"]], float)

    def _norm(s):
        return str(s).lower().replace("_", "").replace(" ", "")

    comp = {}
    for f in sf["fields"]:
        if f["components"] == 1:
            key = _norm(f["name"])
            for ax, tag in (("u", "uvelocity"), ("v", "vvelocity"),
                            ("w", "wvelocity")):
                if key == tag and ax not in comp:
                    comp[ax] = f["name"]
    if len(comp) == 3:
        arr = np.stack([np.asarray(sf["data"][comp[ax]], float).ravel()
                        for ax in "uvw"], axis=1)
        return "+".join(comp[ax] for ax in "uvw"), arr
    return None, None


def do_compare():
    from sim_parser import SimFile
    from scipy.spatial import cKDTree

    ours = np.load(os.path.join(WORK, "ours.npz"))
    sim = SimFile(os.path.join(WORK, "official.sim"))
    vol = sim.extract_volume_mesh()
    sf = sim.extract_solution_fields()
    if not vol.get("ok"):
        return _fail("官方体网格抽取失败：%s" % vol.get("reason"))
    if not sf.get("ok"):
        return _fail("官方解场抽取失败：%s" % sf.get("reason"))
    fields = sorted(f["name"] for f in sf["fields"])
    u_name, uvw = _velocity_field(sf)
    p_name = _pick_field(sf, 1, ("pressure", "press"))
    if uvw is None or p_name is None:
        return _fail("官方字段不足（velocity=%r pressure=%r，全部=%s）"
                    % (u_name, p_name, fields))
    official_meta = {}
    try:
        with open(os.path.join(WORK, "official_result.json"), encoding="utf-8") as f:
            official_meta = json.load(f)
    except Exception:  # noqa: BLE001
        pass
    h = float(ours["h"])
    # 官方单元质心（顶点均值；四面体=质心精确）
    pts = np.asarray(vol["points"], float)
    cen_off = np.array([pts[c].mean(axis=0) for c in vol["cells"]], float)
    p_off = np.asarray(sf["data"][p_name], float).ravel()
    # 质心匹配（同网格 → 距离应≈0；容差 h×0.25 防错配）
    cen_our = np.asarray(ours["centroids"], float)
    dist, idx = cKDTree(cen_our).query(cen_off, k=1)
    tol = MATCH_TOL_FACTOR * h
    matched = dist <= tol
    if matched.sum() < cen_off.shape[0] * 0.999:
        return _fail("质心匹配率 %.2f%%（容差 %.2e）—— 疑非同一网格"
                     % (100.0 * matched.mean(), tol))
    i = idx[matched]
    uo = np.stack([ours["u"][i], ours["v"][i], ours["w"][i]], axis=1)
    vo = uvw[matched]
    du = vo - uo
    vel_scale = float(np.sqrt(np.square(uo).sum(axis=1)).mean())
    uo_c, vo_c = ours["u"][i], vo[:, 0]
    # 压力基准对齐（双方出口零表压的离散实现不同 → 均值对齐后比波动）
    p_ours = np.asarray(ours["p"], float)[i]
    p_off_m = p_off[matched]
    dp_off = _dp(cen_off[matched][:, 0], p_off_m, h)
    report = {
        "ok": True,
        "n_cells_ours": int(ours["n_cells"]), "n_cells_official": int(cen_off.shape[0]),
        "match_rate": float(matched.mean()), "match_max_dist": float(dist.max()),
        "official_fields": fields, "velocity_field": u_name, "pressure_field": p_name,
        "velocity": {
            "rel_l2": _l2(du) / max(_l2(uo), 1e-12),
            "mean_abs": float(np.abs(du).mean()),
            "median_abs": float(np.median(np.abs(du))),
            "max_abs": float(np.abs(du).max()),
            "scale_mean": vel_scale,
            "component_rel_l2": {
                "u": _l2(vo[:, 0] - uo_c) / max(_l2(uo_c), 1e-12),
                "v": _l2(vo[:, 1] - ours["v"][i])
                     / max(_l2(ours["v"][i]), 1e-12),
                "w": _l2(vo[:, 2] - ours["w"][i])
                     / max(_l2(ours["w"][i]), 1e-12)},
        },
        "pressure": {
            "raw_mean_diff": float((p_off_m - p_ours).mean()),
            "aligned_rel_l2": _l2((p_off_m - p_off_m.mean())
                                  - (p_ours - p_ours.mean()))
                             / max(_l2(p_ours - p_ours.mean()), 1e-12),
            "aligned_median_abs": float(np.median(np.abs(
                (p_off_m - p_off_m.mean()) - (p_ours - p_ours.mean())))),
        },
        "cd_ours": float(ours["cd"]), "cl_ours": float(ours["cl"]),
        "dp_ours": float(ours["dp"]), "dp_official": dp_off,
        "ours_iterations": int(ours["iterations"]),
        "ours_residual_final": float(ours["residual_final"]),
        "official_iterations": official_meta.get("iterations"),
        "official_note": ("官方稳态在 Maximum Steps 停止（残差平台 ~1e-3，"
                          "未到我方的 1e-6 判据），对比含双方收敛差"),
    }
    # 官方 Cd：官方单元若为 4 节点 tet → 建我方 FVM 用**同一 force_coefficients**
    cd_off = _official_cd(vol, sf, p_name, uvw, h)
    report.update(cd_off if isinstance(cd_off, dict) else {"cd_official": cd_off,
                                                           "cd_official_note": None})
    with open(os.path.join(WORK, "samemesh_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    v = report["velocity"]
    pr = report["pressure"]
    print("compare: cells ours=%d official=%d match=%.4f%% maxdist=%.2e"
          % (report["n_cells_ours"], report["n_cells_official"],
             100.0 * report["match_rate"], report["match_max_dist"]))
    print("compare: velocity rel_l2=%.3f mean_abs=%.3e max_abs=%.3e (|u|mean=%.3f)"
          % (v["rel_l2"], v["mean_abs"], v["max_abs"], v["scale_mean"]))
    print("compare: pressure aligned rel_l2=%.3f median_abs=%.3e raw_mean=%.3e"
          % (pr["aligned_rel_l2"], pr["aligned_median_abs"], pr["raw_mean_diff"]))
    print("compare: cd ours=%.4f official=%s | dp ours=%.4f official=%.4f"
          % (report["cd_ours"], report.get("cd_official"),
             report["dp_ours"], report["dp_official"]))
    return 0


def _official_cd(vol, sf, p_name, uvw, h):
    """官方 Cd：4 节点 tet → FVM → 圆柱面 force_coefficients（与我方同码）。

    官方场是单元中心量，FVM 面心的压力/速度由相邻单元均值充当（一阶，与我方
    边界面取法同阶）；非 4 节点单元（多面体）时诚实返回 None + 说明。
    uvw 为 _velocity_field 组装好的 (N,3) 速度（U/V/W 标量三元组或矢量场）。
    """
    from aero_forces import force_coefficients
    from fvm_core import FVM
    cells = vol["cells"]
    if not all(len(c) == 4 for c in cells):
        return {"cd_official": None,
                "cd_official_note": "官方单元非四面体（%d 节点分布不一）"
                                    % len(cells[0])}
    pts = np.asarray(vol["points"], float)
    C = np.asarray(cells, np.int64)
    try:
        fv = FVM(pts, C)
    except Exception as exc:  # noqa: BLE001
        return {"cd_official": None, "cd_official_note": "FVM 构建失败：%r" % exc}
    p = np.asarray(sf["data"][p_name], float).ravel()
    if p.size != fv.n_cells or uvw.shape[0] != fv.n_cells:
        return {"cd_official": None,
                "cd_official_note": "官方场与单元数不对齐（%s/%s vs %d）"
                                    % (p.size, uvw.shape[0], fv.n_cells)}
    # 圆柱面：边界面 + 距孔心 r ≤ hole_r + h（与我方同口径，孔位取域几何）
    cen_b = np.asarray(fv.face_centroid, float)
    bnd = np.asarray(fv.is_boundary, bool)
    if not bnd.any():
        return {"cd_official": None, "cd_official_note": "官方无边界面对象"}
    cb = cen_b[bnd]
    hc = np.array([MESH_KW["center_x_D"] * D, 0.5 * MESH_KW["height_D"] * D])
    rh = np.linalg.norm(cb[:, :2] - hc, axis=1)
    cyl_r = 0.5 * D + h
    faces = np.where(bnd)[0][rh <= cyl_r + 1e-9]
    if faces.size == 0:
        return {"cd_official": None, "cd_official_note": "未识别到官方圆柱面"}
    f = force_coefficients(fv, p, uvw[:, 0], uvw[:, 1], uvw[:, 2], NU, RHO,
                           faces=faces, a_ref=D, u_ref=U_INF, rho_ref=RHO,
                           drag_dir=(1.0, 0.0, 0.0), lift_dir=(0.0, 1.0, 0.0))
    return {"cd_official": float(f.get("cd", float("nan"))),
            "cd_official_note": "单元中心场由 force_coefficients 按面插值（与我方同码）"}


def _fail(reason):
    print("compare: FAIL %s" % reason)
    with open(os.path.join(WORK, "samemesh_report.json"), "w", encoding="utf-8") as f:
        json.dump({"ok": False, "reason": reason}, f, ensure_ascii=False, indent=1)
    return 1



# --------------------------------------------------------------- S2 判别：同网格瞬态
TRANSIENT_WORK = os.path.join(WORK, "transient")
TRANSIENT_DT = 0.01            # 与我方 run_case 同时间步
TRANSIENT_TOTAL_TIME = 12.0    # 物理时间（秒）≈ 2.6 个 St=0.176 假设周期
TRANSIENT_SAMPLE_DT = 0.5      # 每 0.5 s 存一个场（24 个采样点）


def _official_forces(vol, sf, h, uvw):
    """官方解场 → FVM → 圆柱面 force_coefficients（与本仓库同码），返回 {cl, cd, note}。"""
    from aero_forces import force_coefficients
    from fvm_core import FVM
    cells = vol["cells"]
    if not all(len(c) == 4 for c in cells):
        return {"cl": None, "cd": None, "note": "官方单元非四面体"}
    pts = np.asarray(vol["points"], float)
    C = np.asarray(cells, np.int64)
    try:
        fv = FVM(pts, C)
    except Exception as exc:  # noqa: BLE001
        return {"cl": None, "cd": None, "note": "FVM 构建失败：%r" % exc}
    p_name = _pick_field(sf, 1, ("pressure", "press"))
    if p_name is None or uvw is None:
        return {"cl": None, "cd": None, "note": "缺压力/速度场"}
    p = np.asarray(sf["data"][p_name], float).ravel()
    if p.size != fv.n_cells or uvw.shape[0] != fv.n_cells:
        return {"cl": None, "cd": None, "note": "场与单元数不对齐"}
    cen_b = np.asarray(fv.face_centroid, float)
    bnd = np.asarray(fv.is_boundary, bool)
    hc = np.array([MESH_KW["center_x_D"] * D, 0.5 * MESH_KW["height_D"] * D])
    rh = np.linalg.norm(cen_b[bnd][:, :2] - hc, axis=1)
    faces = np.where(bnd)[0][rh <= 0.5 * D + h + 1e-9]
    if faces.size == 0:
        return {"cl": None, "cd": None, "note": "未识别到圆柱面"}
    f = force_coefficients(fv, p, uvw[:, 0], uvw[:, 1], uvw[:, 2], NU, RHO,
                           faces=faces, a_ref=D, u_ref=U_INF, rho_ref=RHO,
                           drag_dir=(1.0, 0.0, 0.0), lift_dir=(0.0, 1.0, 0.0))
    return {"cl": float(f.get("cl", float("nan"))),
            "cd": float(f.get("cd", float("nan"))),
            "n_cyl_faces": int(faces.size),
            "note": "单元中心场由 force_coefficients 按面插值（与我方同码）"}


def do_official_transient(total_time=None, sample_dt=None, timeout=7200):
    """S2 判别：官方求解器在**我方同一张网格**上跑瞬态（滑移外壁，与我方对齐）。"""
    from star_bridge import official_mesh_case_transient
    cgns = os.path.join(WORK, "same_mesh.cgns")
    if not os.path.isfile(cgns):
        print("official_transient: 缺 CGNS，请先 prepare")
        return 1
    os.makedirs(TRANSIENT_WORK, exist_ok=True)
    prefix = os.path.join(TRANSIENT_WORK, "official_t")
    res = official_mesh_case_transient(
        SHELL, cgns, prefix, dt=TRANSIENT_DT,
        total_time=float(total_time or TRANSIENT_TOTAL_TIME),
        sample_dt=float(sample_dt or TRANSIENT_SAMPLE_DT),
        rho=RHO, mu=NU, u_in=U_INF, timeout=timeout)
    with open(os.path.join(TRANSIENT_WORK, "official_transient_log.txt"),
              "w", encoding="utf-8") as f:
        f.write(res.get("log") or "")
    summary = {k: v for k, v in res.items() if k != "log"}
    with open(os.path.join(TRANSIENT_WORK, "official_transient_result.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=str)
    print("official_transient: ok=%s dt=%s bnd=%s saves=%d iter=%s fail=%s"
          % (res.get("ok"), res.get("dt"), res.get("bnd_kinds"),
             len(res.get("saves") or []), res.get("iterations"),
             res.get("fail_reason")))
    if not res.get("ok") and not res.get("skipped"):
        tail = (res.get("log") or "")[-1200:]
        if tail:
            print("--- log tail ---\n%s" % tail)
    return 0 if (res.get("ok") or res.get("skipped")) else 1


def do_transient_series():
    """从分段存的官方 .sim 逐个算 CL(t)/Cd(t) → official_cl_series.json。"""
    import glob
    from sim_parser import SimFile
    files = []
    for p in glob.glob(os.path.join(TRANSIENT_WORK, "official_t_*.sim")):
        try:
            n = int(os.path.basename(p).rsplit("_", 1)[1].split(".")[0])
        except Exception:  # noqa: BLE001
            continue
        files.append((n, p))
    files.sort()
    if not files:
        print("transient_series: 无分段场（先跑 official_transient）")
        return 1
    ours_npz = os.path.join(WORK, "ours.npz")
    if not os.path.isfile(ours_npz):
        print("transient_series: 缺 ours.npz（先跑 prepare）")
        return 1
    ours_h = float(np.load(ours_npz)["h"])
    out = []
    sample_dt = TRANSIENT_SAMPLE_DT
    for n, path in files:
        sim = SimFile(path)
        vol = sim.extract_volume_mesh()
        sf = sim.extract_solution_fields()
        if not (vol.get("ok") and sf.get("ok")):
            out.append({"steps": n, "error": "抽取失败 vol=%s sf=%s"
                        % (vol.get("reason"), sf.get("reason"))})
            continue
        _u, uvw = _velocity_field(sf)
        r = _official_forces(vol, sf, ours_h, uvw)
        out.append({"sample": n, "t": round((n - 1) * sample_dt, 6),
                    "cl": r.get("cl"), "cd": r.get("cd"),
                    "note": r.get("note"), "sim": path})
        print("  t=%.2fs cl=%+.5f cd=%.5f"
              % ((n - 1) * sample_dt, r.get("cl", float("nan")),
                 r.get("cd", float("nan"))))
    with open(os.path.join(TRANSIENT_WORK, "official_cl_series.json"),
              "w", encoding="utf-8") as f:
        json.dump({"dt": TRANSIENT_DT, "sample_dt": sample_dt, "series": out}, f,
                  ensure_ascii=False, indent=1)
    print("transient_series: %d 点 → official_cl_series.json" % len(out))
    return 0

def _main(argv=None):
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "all"
    os.makedirs(WORK, exist_ok=True)
    if cmd == "prepare":
        return do_prepare()
    if cmd == "official":
        return do_official()
    if cmd == "compare":
        return do_compare()
    if cmd == "transient":
        args = []
        for a in argv[1:]:
            try:
                args.append(float(a))
            except ValueError:
                pass
        return do_official_transient(*args[:2])
    if cmd == "series":
        return do_transient_series()
    if cmd == "all":
        rc = do_prepare()
        if rc:
            return rc
        rc = do_official()
        if rc:
            return rc
        return do_compare()
    print("用法: python s6_samemesh.py [prepare|official|compare|all|"
          "transient [steps] [save_every]|series]")
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())

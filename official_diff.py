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


def run_case(D=0.04, u_inf=0.05, nu=DEFAULT_NU, spacing=None, dt=None, steps=200,
             n_inner=2, r_far_factor=10.0, thickness=None, sample_every=1):
    """同工况自研求解：环形域 tet + 瞬态 SIMPLE + 圆柱升力积分 → Cl(t) → St。

    **长耗时**：需 STARDECODING_LONG=1；本函数只做"能跑就跑、跑不动如实报告"。
    """
    if os.environ.get("STARDECODING_LONG") != "1":
        return {"ok": False,
                "reason": "长耗时路径：设 STARDECODING_LONG=1 才运行真实瞬态算例"}
    from mesh_tet import tet_mesh
    from pressure_solver import PressureSolver
    from aero_forces import force_coefficients
    surf = annulus_surface(D, r_far_factor=r_far_factor, thickness=thickness)
    h = float(spacing or (D / 4.0))
    res = tet_mesh(surf["points"], surf["triangles"], spacing=h, method="scipy")
    if not res.get("ok"):
        return {"ok": False, "reason": "tet 网格未生成: %s" % res.get("reason")}
    V, C = np.asarray(res["vertices"], float), np.asarray(res["cells"], np.int64)
    dt = float(dt or (0.1 * D / u_inf))
    solver = PressureSolver(V, C, rho=1.0, mu=1.0 * nu, inlet_axis=0, inlet_side="min",
                            inlet_velocity=(u_inf, 0.0, 0.0))
    solver.enable_transient(dt, snapshot=True)
    fv = solver.fv
    cen = fv.face_centroid
    rad = np.linalg.norm(cen[:, :2], axis=1)
    cyl_faces = np.where((fv.boundary_faces) & (rad <= 0.75 * D))[0]
    ts, cls = [], []
    for k in range(int(steps)):
        solver.advance(dt=dt, n_inner=n_inner)
        if k % max(1, int(sample_every)) == 0:
            fc = force_coefficients(fv, solver.p, solver.u, solver.v, solver.w,
                                    solver.mu, solver.rho, faces=cyl_faces,
                                    a_ref=D, u_ref=u_inf, rho_ref=1.0,
                                    drag_dir=(1.0, 0.0, 0.0), lift_dir=(0.0, 1.0, 0.0))
            ts.append(solver.time if hasattr(solver, "time") else k * dt)
            cls.append(float(fc.get("cl", float("nan"))))
    series = np.asarray(cls, float)
    t = np.asarray(ts, float)
    st = strouhal_from_series(t, series, D, u_inf) if series.size >= 16 else None
    n_periods = (st["f"] * (t[-1] - t[0])) if (st and st.get("ok")) else 0.0
    out = {"ok": True, "n_cells": int(C.shape[0]), "dt": dt, "steps": int(steps),
           "spacing": h, "t_span": float(t[-1] - t[0]) if t.size else 0.0,
           "n_samples": int(series.size),
           "mean": float(series.mean()) if series.size else None,
           "amplitude": float(0.5 * (series.max() - series.min())) if series.size else None,
           "st": st.get("st") if st and st.get("ok") else None,
           "n_periods": float(n_periods),
           "reason": "" if (st and st.get("ok") and n_periods >= 3)
                     else "未达脱落周期（样本不足 3 个周期）→ 不做 St 比对"}
    return out


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
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    from sim_parser import SimFile
    ref = reference_case(SimFile(args.file), nu=args.nu, fields=args.fields,
                         diameter_from_mesh=args.diameter_from_mesh)
    rep = {"reference": ref}
    if ref.get("ok") and args.run:
        ours = run_case(D=ref.get("diameter") or 0.04, u_inf=ref.get("u_ref") or 0.05,
                        nu=args.nu, steps=args.steps)
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
            for k, v in (rep.get("diff") or {}).get("items", {}).items():
                print("   差分 %-12s %s" % (k, v))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

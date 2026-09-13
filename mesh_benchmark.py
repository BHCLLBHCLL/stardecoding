# -*- coding: utf-8 -*-
"""R 波 R5：教程尺度网格对标（官方 .sim 网格尺度 ↔ 本仓库网格内核）。

口径（可验证、可复算）：
  · 域表面：G4 边界补丁（extract_boundary_faces）去重后扇形三角化 → 水密性用
    「每条边恰被两个三角形使用」判定；面积与封闭体积（散度定理）作为几何参考量。
  · 官方尺度：官方体网格单元数/顶点数/面数（G3 抽取）。
  · 自研网格：同一域表面上用 mesh_tet.tet_mesh 重划，目标尺寸取官方特征尺寸
    h = (V_ref / n_cells_official)^(1/3)（可用 --spacing 覆盖）。
  · 对标项：单元数比（量级）、体积守恒比（严格）、质量均值；容差见 VERDICTS。

诚实边界：官方体网格在本抽取路径下是闭合流形（每条边恰两单元、无 boundary 面），
其逐单元体积不可靠（面环朝向不一致），故 R5 不拿官方逐单元体积当参考；二维（z 跨度 0）
与边界环非水密（如 methaneOnPt）的语料诚实拒绝，不硬凑对标数字。
"""
import sys

import numpy as np

try:
    from mesh_tet import tet_mesh
except Exception:  # noqa: BLE001
    tet_mesh = None
try:
    from mesh_quality import quality_report
except Exception:  # noqa: BLE001
    quality_report = None

VERDICTS = {
    "cell_ratio": (0.3, 3.0),       # 与官方单元数同量级
    "volume_ratio": (0.98, 1.02),   # 体积守恒（同一域表面）
}


def boundary_surface(sim, vol=None):
    """G4 边界补丁 → 域表面三角网（去重 + 扇形三角化 + 水密/面积/体积）。"""
    if vol is None:
        vol = sim.extract_volume_mesh()
    if not vol.get("ok"):
        return {"ok": False, "reason": vol.get("reason") or "体网格未抽取"}
    pts = np.asarray(vol["points"], float)
    bnd = sim.extract_boundary_faces(vol)
    if not bnd.get("ok"):
        return {"ok": False, "reason": bnd.get("reason") or "边界未抽取"}
    seen, tris, n_rings = set(), [], 0
    for b in bnd.get("boundaries") or []:
        key = (b.get("region_name"), b.get("index"))
        if key in seen:
            continue
        seen.add(key)
        for ring in b.get("rings") or []:
            r = [int(x) for x in ring]
            n_rings += 1
            for k in range(1, len(r) - 1):
                tris.append((r[0], r[k], r[k + 1]))
    if not tris:
        return {"ok": False, "reason": "边界补丁无面环"}
    F = np.asarray(tris, np.int64)
    V = pts
    area = float(0.5 * np.linalg.norm(
        np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum())
    volume = float(abs((V[F[:, 0]] * np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6.0))
    edges = {}
    for a, b2, c in F:
        for u, v in ((a, b2), (b2, c), (c, a)):
            k = (int(u), int(v)) if u < v else (int(v), int(u))
            edges[k] = edges.get(k, 0) + 1
    counts = {}
    for c in edges.values():
        counts[c] = counts.get(c, 0) + 1
    watertight = set(counts) == {2}
    return {"ok": True, "points": V, "triangles": F,
            "n_boundaries": len(seen), "n_rings": n_rings,
            "n_triangles": int(F.shape[0]), "n_edges": len(edges),
            "watertight": watertight, "edge_usage": {str(k): v for k, v in counts.items()},
            "area": area, "volume": volume,
            "z_span": float(V[:, 2].max() - V[:, 2].min()), "reason": ""}


def official_scale(sim, vol=None):
    """官方网格尺度：单元/顶点/面数 + 由域体积反推的特征尺寸。"""
    if vol is None:
        vol = sim.extract_volume_mesh()
    if not vol.get("ok"):
        return {"ok": False, "reason": vol.get("reason") or "体网格未抽取"}
    surf = boundary_surface(sim, vol)
    if not surf.get("ok"):
        return {"ok": False, "reason": surf["reason"]}
    n_cells = int(vol.get("count") or 0)
    face_cells = vol.get("face_cells")
    n_faces = int(np.asarray(face_cells).size // 2) if face_cells is not None else None
    spacing = (surf["volume"] / n_cells) ** (1.0 / 3.0) if n_cells else 0.0
    return {"ok": True, "n_cells": n_cells,
            "n_points": int(np.asarray(vol["points"]).shape[0]), "n_faces": n_faces,
            "domain_volume": surf["volume"], "domain_area": surf["area"],
            "watertight": surf["watertight"], "is_2d": surf["z_span"] <= 1e-12,
            "target_spacing": spacing, "reason": ""}


def _edge_stats(V, F):
    """三角网边统计：唯一边数 + 边长均值/中位数 + 边使用分布（水密判定）。"""
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    edges = {}
    for a, b, c in F:
        for u, v in ((a, b), (b, c), (c, a)):
            k = (int(u), int(v)) if u < v else (int(v), int(u))
            edges.setdefault(k, np.linalg.norm(V[k[0]] - V[k[1]]))
    L = np.array(list(edges.values()), float)
    counts = {}
    for a, b, c in F:
        for u, v in ((a, b), (b, c), (c, a)):
            k = (int(u), int(v)) if u < v else (int(v), int(u))
            counts[k] = counts.get(k, 0) + 1
    usage = {}
    for c in counts.values():
        usage[c] = usage.get(c, 0) + 1
    return {"n_edges": len(edges), "n_faces": int(F.shape[0]),
            "edge_mean": float(L.mean()) if L.size else 0.0,
            "edge_median": float(np.median(L)) if L.size else 0.0,
            "edge_min": float(L.min()) if L.size else 0.0,
            "edge_max": float(L.max()) if L.size else 0.0,
            "edge_usage": {str(k): v for k, v in usage.items()},
            "watertight": set(usage) == {2}}


def _tri_min_angles(V, F):
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    out = np.zeros(F.shape[0], float)
    for i, (P, Q, R) in enumerate(((A, B, C), (B, C, A), (C, A, B))):
        e1, e2 = Q - P, R - P
        denom = np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1)
        cos = np.divide((e1 * e2).sum(axis=1), denom,
                        out=np.ones_like(denom), where=denom > 0)
        ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        out = ang if i == 0 else np.minimum(out, ang)
    return out


def _patch_triangles(sim, vol, patch_name, patch_index):
    """单个边界补丁 → (V, F) 三角面片。"""
    bnd = sim.extract_boundary_faces(vol)
    V = np.asarray(vol["points"], float)
    tris = []
    for b in bnd.get("boundaries") or []:
        if (str(b.get("name")), b.get("index")) != (patch_name, patch_index):
            continue
        for ring in b.get("rings") or []:
            r = [int(x) for x in ring]
            for k in range(1, len(r) - 1):
                tris.append((r[0], r[k], r[k + 1]))
    return V, np.asarray(tris, np.int64)


def boundary_patch_stats(sim, vol=None):
    """每个边界补丁的离散尺度与质量（官方网格的"分片尺度"参考）。"""
    if vol is None:
        vol = sim.extract_volume_mesh()
    if not vol.get("ok"):
        return {"ok": False, "reason": vol.get("reason") or "体网格未抽取"}
    bnd = sim.extract_boundary_faces(vol)
    if not bnd.get("ok"):
        return {"ok": False, "reason": bnd.get("reason") or "边界未抽取"}
    seen, patches = set(), []
    for b in bnd.get("boundaries") or []:
        key = (str(b.get("name")), b.get("index"))
        if key in seen:
            continue
        seen.add(key)
        V, F = _patch_triangles(sim, vol, key[0], key[1])
        if not F.size:
            continue
        st = _edge_stats(V, F)
        ar = 0.5 * np.linalg.norm(
            np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
        ang = _tri_min_angles(V, F)
        patches.append({"name": key[0], "index": key[1],
                        "region": b.get("region_name"),
                        "n_faces": int(len(b.get("rings") or [])),
                        "n_triangles": int(F.shape[0]), "n_edges": st["n_edges"],
                        "area": float(ar.sum()),
                        "edge_median": st["edge_median"], "edge_mean": st["edge_mean"],
                        "min_angle_median": float(np.median(ang)) if ang.size else 0.0,
                        "min_angle_min": float(ang.min()) if ang.size else 0.0})
    return {"ok": True, "patches": patches, "n_patches": len(patches),
            "total_faces": sum(p["n_triangles"] for p in patches), "reason": ""}


def surface_benchmark(sim, patch=None, max_faces=400, max_iter=2, whiten=True):
    """R5 主路径：取一个官方边界补丁，用自研重网格化到官方边尺度并对比。

    默认取"三角形数 ≤ max_faces 的最大补丁"（保证秒级完成）；patch=(name,index) 可指定。
    对比项：三角数比、面积比、边中位数比、最小角中位数（质量）。
    """
    from mesh_remesh import remesh_surface
    vol = sim.extract_volume_mesh()
    stats = boundary_patch_stats(sim, vol)
    if not stats.get("ok"):
        return {"ok": False, "reason": stats.get("reason")}
    if not stats["patches"]:
        return {"ok": False, "reason": "无边界补丁（纯表面网格或二维文件）"}
    if patch is not None:
        cand = [p for p in stats["patches"] if (p["name"], p["index"]) == tuple(patch)]
        if not cand:
            return {"ok": False, "reason": "指定补丁不存在: %r" % (patch,)}
        pick = cand[0]
    else:
        small = [p for p in stats["patches"] if p["n_triangles"] <= max_faces]
        pick = max(small, key=lambda p: p["n_triangles"]) if small else \
            min(stats["patches"], key=lambda p: p["n_triangles"])
    V, F = _patch_triangles(sim, vol, pick["name"], pick["index"])
    if not F.size:
        return {"ok": False, "reason": "补丁无三角面片"}
    off = _edge_stats(V, F)
    h = float(pick["edge_median"])
    ar_base = float(0.5 * np.linalg.norm(
        np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum())

    def _run(**kw):
        Va, Fa = remesh_surface(V, F, h, curvature_adaptive=False, max_iter=max_iter, **kw)
        Va = np.asarray(Va, float)
        Fa = np.asarray(Fa, np.int64)
        sa = _edge_stats(Va, Fa)
        aa = float(0.5 * np.linalg.norm(
            np.cross(Va[Fa[:, 1]] - Va[Fa[:, 0]], Va[Fa[:, 2]] - Va[Fa[:, 0]]),
            axis=1).sum())
        fr = sa["n_faces"] / float(off["n_faces"]) if off["n_faces"] else float("nan")
        return {"vertices": Va, "faces": Fa, "stats": sa, "area": aa,
                "face_ratio": fr, "area_ratio": aa / ar_base if ar_base else float("nan"),
                "ops": {"collapse": kw.get("collapse", True),
                        "smooth": kw.get("smooth", True), "max_iter": max_iter}}

    attempts = [_run()]
    used_fallback = False
    a0 = attempts[0]
    if not (0.5 <= a0["face_ratio"] <= 2.0 and 0.9 <= a0["area_ratio"] <= 1.1):
        # 退化保护：激进算子集（split+collapse+smooth）在"目标≈当前边尺度"时会过粗化，
        # 保守算子集（只 split+flip）保持面积与面数 —— 两次都记录在 attempts 里，不用猜测。
        attempts.append(_run(collapse=False, smooth=False))
        used_fallback = True
        used = max(attempts, key=lambda x: (0.5 <= x["face_ratio"] <= 2.0,
                                            0.9 <= x["area_ratio"] <= 1.1,
                                            abs(x["area_ratio"] - 1.0) < 0.1))
    else:
        used = a0
    V2, F2 = used["vertices"], used["faces"]
    ours = used["stats"]
    ar_o = float(0.5 * np.linalg.norm(
        np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum())
    ar_u = float(0.5 * np.linalg.norm(
        np.cross(V2[F2[:, 1]] - V2[F2[:, 0]], V2[F2[:, 2]] - V2[F2[:, 0]]), axis=1).sum())
    ang_o = _tri_min_angles(V, F)
    ang_u = _tri_min_angles(V2, F2)
    face_ratio = ours["n_faces"] / float(off["n_faces"]) if off["n_faces"] else float("nan")
    area_ratio = ar_u / ar_o if ar_o else float("nan")
    edge_ratio = (ours["edge_median"] / off["edge_median"]
                  if off["edge_median"] else float("nan"))
    ang_o_m = float(np.median(ang_o)) if ang_o.size else 0.0
    ang_u_m = float(np.median(ang_u)) if ang_u.size else 0.0
    verdict = {
        "patch": {"name": pick["name"], "index": pick["index"],
                  "n_triangles_official": off["n_faces"]},
        "face_ratio": face_ratio, "face_ratio_ok": 0.5 <= face_ratio <= 2.0,
        "area_ratio": area_ratio, "area_ratio_ok": 0.97 <= area_ratio <= 1.03,
        "edge_median_ratio": edge_ratio, "edge_median_ok": 0.6 <= edge_ratio <= 1.6,
        "min_angle_median_official": ang_o_m,
        "min_angle_median_ours": ang_u_m,
        "min_angle_ok": ang_u_m >= min(20.0, ang_o_m - 5.0),
    }
    verdict["ok"] = bool(verdict["face_ratio_ok"] and verdict["area_ratio_ok"]
                         and verdict["edge_median_ok"] and verdict["min_angle_ok"])
    return {"ok": True, "mode": "surface-patch",
            "official": dict(off, area=ar_o, patch=pick,
                             min_angle_median=ang_o_m,
                             min_angle_min=float(ang_o.min()) if ang_o.size else 0.0),
            "ours": dict(ours, area=ar_u, n_points=int(V2.shape[0]), target_edge=h,
                         min_angle_median=ang_u_m,
                         min_angle_min=float(ang_u.min()) if ang_u.size else 0.0),
            "patches": stats["patches"],
            "attempts": [{"ops": a["ops"], "face_ratio": a["face_ratio"],
                          "area_ratio": a["area_ratio"]} for a in attempts],
            "used_fallback": used_fallback,
            "verdict": verdict, "reason": ""}


def benchmark(sim, mode="surface", **kwargs):
    """对标入口：mode=surface（默认，快速）/ tet（同域表面 tet 重划，慢，opt-in）。"""
    if mode == "surface":
        return surface_benchmark(sim, **kwargs)
    if mode == "tet":
        return benchmark_tet(sim, **kwargs)
    return {"ok": False, "reason": "未知对标模式 %r（surface/tet）" % mode}

def benchmark_tet(sim, spacing=None, coarsen=2.0, max_cells=60000):
    """官方尺度 ↔ 自研 tet：同一域表面重划 + 单元数/体积/质量对标。

    默认按官方特征尺寸**粗化 2 倍**（target = h_official × coarsen）以保证常规机器可跑；
    预估单元数超 max_cells 时自动继续粗化并如实记录（adjusted 字段）。--spacing 直接覆盖。
    """
    if tet_mesh is None:
        return {"ok": False, "reason": "mesh_tet 不可用（缺 numpy/scipy）"}
    vol = sim.extract_volume_mesh()
    off = official_scale(sim, vol)
    if not off.get("ok"):
        return {"ok": False, "official": off, "reason": off.get("reason")}
    if off["is_2d"]:
        return {"ok": False, "official": off,
                "reason": "二维域（z 跨度 0）不做 tet 尺度对标（诚实拒绝）"}
    surf = boundary_surface(sim, vol)
    if not surf.get("watertight"):
        return {"ok": False, "official": off,
                "reason": "域表面非水密（边使用分布 %s）→ 不做 tet 对标（诚实拒绝）"
                          % surf.get("edge_usage")}
    h = float(spacing) if spacing else float(off["target_spacing"]) * float(coarsen)
    adjusted = 0
    while off["domain_volume"] and (off["domain_volume"] / h ** 3) > max_cells:
        h *= 1.25
        adjusted += 1
    res = tet_mesh(surf["points"], surf["triangles"], spacing=h, method="scipy")
    if not res.get("ok"):
        return {"ok": False, "official": off,
                "reason": "自研 tet 未生成: %s" % res.get("reason")}
    V = np.asarray(res["vertices"], float)
    C = np.asarray(res["cells"], np.int64)
    from mesh_tet import _tet_volumes
    vols = np.abs(_tet_volumes(V, C))
    v_total = float(vols.sum())
    quality = None
    if quality_report is not None:
        try:
            quality = quality_report(V, C, kind="tet")
        except Exception:  # noqa: BLE001
            quality = None
    cell_ratio = C.shape[0] / float(off["n_cells"]) if off["n_cells"] else float("nan")
    vol_ratio = v_total / off["domain_volume"] if off["domain_volume"] else float("nan")
    ours = {"n_cells": int(C.shape[0]), "n_points": int(V.shape[0]),
            "volume_total": v_total,
            "cell_volume": {"mean": float(vols.mean()) if vols.size else 0.0,
                            "min": float(vols.min()) if vols.size else 0.0,
                            "max": float(vols.max()) if vols.size else 0.0},
            "char_size": (v_total / C.shape[0]) ** (1.0 / 3.0) if C.shape[0] else 0.0,
            "quality": ({"mean": quality.get("mean")} if isinstance(quality, dict)
                        and "mean" in quality else None)}
    verdict = {
        "cell_ratio": cell_ratio,
        "cell_ratio_ok": VERDICTS["cell_ratio"][0] <= cell_ratio <= VERDICTS["cell_ratio"][1],
        "volume_ratio": vol_ratio,
        "volume_ratio_ok": VERDICTS["volume_ratio"][0] <= vol_ratio <= VERDICTS["volume_ratio"][1],
        "official_cells": off["n_cells"], "our_cells": int(C.shape[0]),
        "domain_volume": off["domain_volume"],
    }
    return {"ok": True, "official": off, "ours": ours, "verdict": verdict,
            "surface": {"points": int(surf["points"].shape[0]),
                        "triangles": surf["n_triangles"], "edges": surf["n_edges"],
                        "boundaries": surf["n_boundaries"], "rings": surf["n_rings"],
                        "area": surf["area"], "volume": surf["volume"],
                        "watertight": surf["watertight"]},
            "spacing": h, "coarsen": coarsen, "coarsen_adjusted": adjusted,
            "est_cells": (off["domain_volume"] / h ** 3) if h else 0.0,
            "reason": ""}


def _main(argv=None):
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    import argparse
    ap = argparse.ArgumentParser(
        description="R5 教程尺度网格对标（官方 .sim 网格尺度为参考；默认表面补丁模式，秒级）")
    ap.add_argument("file")
    ap.add_argument("--mode", choices=("surface", "tet"), default="surface",
                    help="surface=边界补丁重网格化对标（默认，秒级）；tet=同域表面 tet 对标（慢）")
    ap.add_argument("--patch", metavar="NAME:INDEX", default=None,
                    help="指定边界补丁（如 Body 1.inlet:3）")
    ap.add_argument("--spacing", type=float, default=None, help="tet 模式目标尺寸")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args(argv)
    from sim_parser import SimFile
    sim = SimFile(args.file)
    patch = None
    if args.patch:
        name, _, idx = args.patch.rpartition(":")
        patch = (name, int(idx))
    if args.mode == "tet":
        rep = benchmark(sim, mode="tet", spacing=args.spacing)
    else:
        rep = surface_benchmark(sim, patch=patch)
    if args.json:
        import json as _json
        print(_json.dumps(rep, ensure_ascii=False, indent=1,
                          default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        return 0 if rep.get("ok") else 1
    if not rep.get("ok"):
        print("对标未完成:", rep.get("reason"))
        return 1
    if rep["mode"] == "tet":
        o, u, v = rep["official"], rep["ours"], rep["verdict"]
        print("官方网格: 单元 %d / 域体积 %.6g / 特征尺寸 %.6g"
              % (o["n_cells"], o["domain_volume"], o["target_spacing"]))
        print("自研 tet: 单元 %d / 体积 %.6g（spacing=%.6g）"
              % (u["n_cells"], u["volume_total"], rep["spacing"]))
        print("对标: 单元数比 %.3f（%s）  体积比 %.5f（%s）"
              % (v["cell_ratio"], "达标" if v["cell_ratio_ok"] else "未达标",
                 v["volume_ratio"], "达标" if v["volume_ratio_ok"] else "未达标"))
        return 0 if (v["cell_ratio_ok"] and v["volume_ratio_ok"]) else 1
    o, u, v = rep["official"], rep["ours"], rep["verdict"]
    print("边界补丁 %s(idx=%s): 三角 %d → %d  边中位数 %.6g → %.6g  面积 %.6g → %.6g"
          % (v["patch"]["name"], v["patch"]["index"], o["n_faces"], u["n_faces"],
             o["edge_median"], u["edge_median"], o["area"], u["area"]))
    print("最小角中位数: %.2f° → %.2f°（质量）" % (o["min_angle_median"],
                                                u["min_angle_median"]))
    print("对标: 面数比 %.3f（%s） 面积比 %.4f（%s） 边尺度比 %.3f（%s） 质量 %s"
          % (v["face_ratio"], "达标" if v["face_ratio_ok"] else "未达标",
             v["area_ratio"], "达标" if v["area_ratio_ok"] else "未达标",
             v["edge_median_ratio"], "达标" if v["edge_median_ok"] else "未达标",
             "达标" if v["min_angle_ok"] else "未达标"))
    for a in rep["attempts"]:
        print("   尝试 %s → 面数比 %.3f 面积比 %.4f"
              % (a["ops"], a["face_ratio"], a["area_ratio"]))
    if rep.get("used_fallback"):
        print("   （激进算子集过粗化 → 已回退保守算子集，两次记录见 attempts）")
    return 0 if v["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())


# -*- coding: utf-8 -*-
"""S 波 S6：官方参考语料库与自动对标。

管线：`benchmarks/manifest.json`（清单：算例 .sim / 参数 / 容差 / 出处）
      → `extract_reference(case)`（现场从官方 .sim 抽取参考量：St/振幅/残差，不写死）
      → `compare(case, ref, ours)`（按清单容差判定；缺结果标「未提供」不计通过）
      → `render_report` / JSON 报告。

诚实边界：官方参考量来自 .sim 内嵌监视器（G6）；自研结果由 `official_diff.run_case` 输出，
**未提供或未达脱落周期时不得判通过**（报告里显式区分「未提供 / 不可比 / 通过 / 未通过」）。
"""
import json
import os
import sys

DEFAULT_MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "benchmarks", "manifest.json")


def load_manifest(path=DEFAULT_MANIFEST):
    """读清单并校验：version / cases[].id / .sim / .tolerances。返回 {ok, cases, reason}。"""
    if not os.path.isfile(path):
        return {"ok": False, "cases": [], "reason": "清单不存在: %s" % path}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "cases": [], "reason": "清单解析失败: %s" % exc}
    if not isinstance(data, dict) or data.get("version") is None:
        return {"ok": False, "cases": [], "reason": "清单缺 version"}
    cases, problems = [], []
    for idx, c in enumerate(data.get("cases") or []):
        for key in ("id", "sim", "params", "tolerances"):
            if key not in c:
                problems.append("cases[%d] 缺字段 %s" % (idx, key))
        c = dict(c)
        c["sim_exists"] = bool(c.get("sim")) and os.path.isfile(c["sim"])
        cases.append(c)
    if problems:
        return {"ok": False, "cases": cases, "reason": "; ".join(problems)}
    return {"ok": True, "cases": cases, "reason": "", "ours_block": data.get("ours") or {}}


def extract_reference(case, fields=False):
    """从官方 .sim 现场抽取参考量（现场抽取 → 清单里的数字不会过期）。"""
    if not case.get("sim_exists"):
        return {"ok": False, "reason": "官方 .sim 不存在: %s" % case.get("sim")}
    try:
        from sim_parser import SimFile
        from official_diff import reference_case
        sim = SimFile(case["sim"])
        ref = reference_case(sim, nu=float(case["params"].get("nu", 1e-5)), fields=fields)
    except MemoryError:
        return {"ok": False, "reason": "内存不足，跳过抽取"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": "抽取失败: %s" % exc}
    if not ref.get("ok"):
        return {"ok": False, "reason": ref.get("reason") or "无参考量"}
    st = ref.get("strouhal") or {}
    return {"ok": True, "u_ref": ref.get("u_ref"), "diameter": ref.get("diameter"),
            "reynolds": ref.get("reynolds"), "st": st.get("st"),
            "st_fft": st.get("st_fft"), "st_cross": st.get("st_cross"),
            "amplitude": st.get("amplitude"), "n_samples": st.get("n"),
            "t_span": st.get("t_span"), "residual_final": (ref.get("residual") or {}).get("final"),
            "lift_name": (ref.get("lift") or {}).get("name"), "reason": ""}

def compare(case, ref, ours):
    """按清单容差判定：返回 {verdict, ok, items}。

    verdict ∈ {"通过","未通过","未提供","不可比"}；ok ∈ {True, False, None, None}。
    """
    if not ref or not ref.get("ok"):
        return {"verdict": "不可比", "ok": None, "items": {},
                "reason": "官方参考量不可用: %s" % ((ref or {}).get("reason") or "?")}
    if not ours:
        return {"verdict": "未提供", "ok": None, "items": {},
                "reason": "自研未提供该算例结果（不计通过）"}
    periods = ours.get("n_periods")
    if ours.get("st") is None:
        why = ours.get("reason") or ("未达脱落周期（周期数 %s < 3）" % periods
                                     if periods is not None else "自研无 St")
        return {"verdict": "不可比", "ok": None, "items": {}, "reason": why}
    tol = case.get("tolerances") or {}
    st_rel = float(tol.get("st_rel", 0.15))
    band = (1.0 - st_rel, 1.0 + st_rel)
    items, ok_all = {}, True
    ref_st, our_st = ref.get("st"), ours.get("st")
    if ref_st and our_st:
        ratio = float(our_st) / float(ref_st)
        ok = band[0] <= ratio <= band[1]
        items["st"] = {"ours": float(our_st), "official": float(ref_st), "ratio": ratio,
                       "ok": ok, "band": list(band)}
        ok_all = ok_all and ok
    amp_band = tol.get("amplitude_ratio") or [0.3, 3.0]
    ref_amp, our_amp = ref.get("amplitude"), ours.get("amplitude")
    if ref_amp and our_amp:
        ratio = float(our_amp) / float(ref_amp)
        ok = amp_band[0] <= ratio <= amp_band[1]
        items["amplitude"] = {"ours": float(our_amp), "official": float(ref_amp),
                              "ratio": ratio, "ok": ok, "band": list(amp_band)}
        ok_all = ok_all and ok
    return {"verdict": "通过" if ok_all else "未通过", "ok": ok_all, "items": items,
            "reason": ""}


def run_bench(manifest_path=DEFAULT_MANIFEST, ours_path=None, do_extract=True,
              fields=False):
    """跑一遍对标：抽参考（可关）→ 与自研结果比对 → 结构化报告。"""
    man = load_manifest(manifest_path)
    if not man.get("ok"):
        return {"ok": False, "reason": man.get("reason"), "cases": []}
    ours_map = {}
    if ours_path:
        try:
            with open(ours_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            ours_map = payload.get("cases") if isinstance(payload, dict) else {}
            if not isinstance(ours_map, dict):
                ours_map = {}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": "自研结果读取失败: %s" % exc, "cases": []}
    rows = []
    for case in man["cases"]:
        ref = extract_reference(case, fields=fields) if do_extract else {"ok": False,
                                                                        "reason": "未抽取"}
        ours = ours_map.get(case["id"])
        cmp_ = compare(case, ref, ours)
        rows.append({"id": case["id"], "title": case.get("title"), "sim": case.get("sim"),
                     "params": case.get("params"), "provenance": case.get("provenance"),
                     "reference": ref, "ours": ours, "verdict": cmp_})
    n_pass = sum(1 for r in rows if r["verdict"]["verdict"] == "通过")
    n_fail = sum(1 for r in rows if r["verdict"]["verdict"] == "未通过")
    n_na = sum(1 for r in rows if r["verdict"]["verdict"] in ("未提供", "不可比"))
    return {"ok": True, "n_cases": len(rows), "n_pass": n_pass, "n_fail": n_fail,
            "n_not_applicable": n_na, "cases": rows,
            "note": "「未提供/不可比」不计通过；官方参考量现场抽取，不写死", "reason": ""}


def render_report(report):
    """文本报告（每条算例一行结论 + 明细）。"""
    if not report.get("ok"):
        return "对标报告不可用: %s" % report.get("reason")
    lines = ["== S6 官方参考对标 ==",
             "算例 %d：通过 %d / 未通过 %d / 未提供或不可比 %d"
             % (report["n_cases"], report["n_pass"], report["n_fail"],
                report["n_not_applicable"])]
    for r in report["cases"]:
        v, ref, ours = r["verdict"], r["reference"], r["ours"] or {}
        lines.append("  [%s] %s" % (v["verdict"], r["id"]))
        if ref.get("ok"):
            lines.append("      官方: U=%s D=%s Re=%s St=%s（FFT %s / 过零 %s）振幅 %s 残差 %s"
                         % (ref.get("u_ref"), ref.get("diameter"), ref.get("reynolds"),
                            _fmt(ref.get("st")), _fmt(ref.get("st_fft")), _fmt(ref.get("st_cross")),
                            _fmt(ref.get("amplitude")), _fmt(ref.get("residual_final"))))
        else:
            lines.append("      官方: 不可用（%s）" % ref.get("reason"))
        if v.get("items"):
            for key, item in v["items"].items():
                lines.append("      对标 %s: 自研 %s vs 官方 %s → 比 %s（带 %s）%s"
                             % (key, _fmt(item["ours"]), _fmt(item["official"]),
                                _fmt(item["ratio"]), item["band"],
                                "达标" if item["ok"] else "未达标"))
        elif v.get("reason"):
            lines.append("      结论: %s" % v["reason"])
    return "\n".join(lines)


def _fmt(v, nd=4):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return ("%%.%dg" % nd) % v
    return str(v)


def _main(argv=None):
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    import argparse
    ap = argparse.ArgumentParser(description="S6 官方参考语料库与自动对标")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST, help="清单路径")
    ap.add_argument("--ours", default=None, help="自研结果 JSON（{cases: {id: {...}}}）")
    ap.add_argument("--no-extract", action="store_true", help="只校验清单，不抽取官方参考")
    ap.add_argument("--fields", action="store_true", help="附解场统计（较慢）")
    ap.add_argument("--out", default=None, help="报告 JSON 输出路径")
    ap.add_argument("--json", action="store_true", help="stdout 输出 JSON")
    args = ap.parse_args(argv)
    report = run_bench(args.manifest, ours_path=args.ours,
                       do_extract=not args.no_extract, fields=args.fields)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1, default=str)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    else:
        print(render_report(report))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(_main())

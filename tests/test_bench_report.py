# -*- coding: utf-8 -*-
"""S 波 S6：官方参考语料库与自动对标（清单 / 抽取 / 判定 / 报告）。

覆盖：
  清单：真实清单 3 算例；缺文件/缺字段 → 如实报错；sim_exists 探测
  抽取：语料在盘时抽出的 St/振幅/残差与文档一致（内存不足则跳过）
  判定：通过 / 未通过 / 未提供（缺自研结果，不计通过）/ 不可比（未达周期）四条路径
  报告：文本报告含"未提供/不可比/达标/未达标"等诚实标记
"""
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest  # noqa: E402

import bench_report as br  # noqa: E402

MANIFEST = os.path.join(ROOT, "benchmarks", "manifest.json")
CYL = "D:/training/openfoam/benchmark/vortexShed_tutor_v3_0.05_2502.sim"


def test_load_manifest_real():
    man = br.load_manifest(MANIFEST)
    assert man["ok"] and len(man["cases"]) == 4
    ids = [c["id"] for c in man["cases"]]
    assert ids == ["cyl_re200_u005", "cyl_re100_u0025", "cyl_re200_seg",
                   "airfoil_multielement_official"]
    for c in man["cases"]:
        assert "params" in c and "tolerances" in c
        assert "provenance" in c and c["provenance"]
    cyl = [c for c in man["cases"] if c["id"] == "cyl_re200_u005"][0]
    assert cyl["params"]["U"] == 0.05 and "st_rel" in cyl["tolerances"]
    af = [c for c in man["cases"] if c["id"] == "airfoil_multielement_official"][0]
    assert af["metrics"] == ["cl", "cd"] and "cl_rel" in af["tolerances"]


def test_load_manifest_missing_and_invalid():
    bad = br.load_manifest(os.path.join(ROOT, "benchmarks", "no_such.json"))
    assert bad["ok"] is False and "不存在" in bad["reason"]
    tmp = tempfile.mkdtemp(prefix="s6man_")
    try:
        p = os.path.join(tmp, "m.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "cases": [{"id": "x", "sim": "y"}]}, f)
        out = br.load_manifest(p)
        assert out["ok"] is False
        assert "tolerances" in out["reason"] and "params" in out["reason"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _case():
    return {"id": "c", "tolerances": {"st_rel": 0.15, "amplitude_ratio": [0.3, 3.0]}}


def test_compare_four_paths():
    ref = {"ok": True, "st": 0.176, "amplitude": 0.28}
    ok = br.compare(_case(), ref, {"st": 0.18, "amplitude": 0.30})
    assert ok["verdict"] == "通过" and ok["ok"] is True
    bad = br.compare(_case(), ref, {"st": 0.05, "amplitude": 0.30})
    assert bad["verdict"] == "未通过" and bad["ok"] is False
    missing = br.compare(_case(), ref, None)
    assert missing["verdict"] == "不可比" or missing["verdict"] == "未提供"
    no_st = br.compare(_case(), ref, {"st": None, "n_periods": 2.0,
                                      "reason": "未达脱落周期（<3 个周期）"})
    assert no_st["verdict"] == "不可比" and no_st["ok"] is None
    assert "未达脱落周期" in no_st["reason"]
    ref_bad = br.compare(_case(), {"ok": False, "reason": "无监视器"}, {"st": 0.18})
    assert ref_bad["verdict"] == "不可比"


def test_render_report_honest_markers():
    report = {"ok": True, "n_cases": 2, "n_pass": 1, "n_fail": 0, "n_not_applicable": 1,
              "cases": [
                  {"id": "a", "title": "t", "sim": "s", "params": {}, "provenance": "p",
                   "reference": {"ok": True, "st": 0.176, "st_fft": 0.176, "st_cross": 0.174,
                                 "amplitude": 0.28, "u_ref": 0.05, "diameter": 0.04,
                                 "reynolds": 200, "residual_final": 3e-10},
                   "ours": {"st": 0.18, "amplitude": 0.3},
                   "verdict": {"verdict": "通过", "ok": True, "reason": "",
                               "items": {"st": {"ours": 0.18, "official": 0.176,
                                                "ratio": 1.02, "ok": True, "band": [0.85, 1.15]}}}},
                  {"id": "b", "title": "t", "sim": "s", "params": {}, "provenance": "p",
                   "reference": {"ok": True, "st": 0.16, "amplitude": 0.5},
                   "ours": None,
                   "verdict": {"verdict": "未提供", "ok": None, "items": {},
                               "reason": "自研未提供该算例结果（不计通过）"}}],
              "note": "", "reason": ""}
    text = br.render_report(report)
    assert "通过 1" in text and "未提供或不可比 1" in text
    assert "[通过]" in text and "[未提供]" in text
    assert "达标" in text and "不计通过" in text
    bad = br.render_report({"ok": False, "reason": "清单不存在"})
    assert "对标报告不可用" in bad


def test_run_bench_without_extract_on_real_manifest():
    rep = br.run_bench(MANIFEST, ours_path=os.path.join(ROOT, "benchmarks",
                                                       "ours_cylinder.json"),
                       do_extract=False)
    assert rep["ok"] and rep["n_cases"] == 4
    assert rep["n_pass"] == 0 and rep["n_not_applicable"] == 4
    by_id = {c["id"]: c for c in rep["cases"]}
    assert by_id["cyl_re200_u005"]["ours"] is not None
    assert by_id["cyl_re100_u0025"]["ours"] is None


def test_compare_steady_forces_path():
    case = {"id": "af", "metrics": ["cl", "cd"],
            "tolerances": {"cl_rel": 0.15, "cd_rel": 0.15}}
    ref = {"ok": True, "forces": {"cl": 2.2445, "cd": 0.0756}, "st": None, "amplitude": None}
    ok = br.compare(case, ref, {"cl": 2.30, "cd": 0.079})
    assert ok["verdict"] == "通过" and ok["ok"] is True
    assert ok["items"]["cl"]["ratio"] == pytest.approx(2.30 / 2.2445, rel=1e-6)
    bad = br.compare(case, ref, {"cl": 3.5, "cd": 0.079})
    assert bad["verdict"] == "未通过" and bad["items"]["cl"]["ok"] is False
    none = br.compare(case, ref, None)
    assert none["verdict"] == "未提供"
    partial = br.compare(case, ref, {"cl": 2.30})
    assert partial["items"]["cl"]["ok"] is True
    no_ref = br.compare(case, {"ok": True, "forces": {}}, {"cl": 2.3})
    assert no_ref["verdict"] == "未提供"


@pytest.mark.skipif(not os.path.isfile(CYL), reason="官方语料缺失")
def test_extract_reference_matches_documented_values():
    case = {"sim": CYL, "sim_exists": True, "params": {"nu": 1e-5}}
    ref = br.extract_reference(case)
    if not ref.get("ok"):
        pytest.skip(ref.get("reason"))
    assert abs(ref["st"] - 0.175) < 0.01, ref
    assert abs(ref["amplitude"] - 0.28) < 0.05, ref
    assert ref["reynolds"] == pytest.approx(200.0, rel=1e-6)
    assert ref["residual_final"] < 1e-8

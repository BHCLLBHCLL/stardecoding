# -*- coding: utf-8 -*-
"""S 波 S1：官方 STAR-CCM+ 桥（探测 / 宏模板 / 日志解析 / 门控）。

默认只跑离线用例；官方集成用例需 STARDECODING_OFFICIAL=1（每次官方 -batch 约 1–3 分钟）。
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest  # noqa: E402

import star_bridge as sb  # noqa: E402

CORPUS = r"D:/training/starccm/startutorialsdata"
SMALL = os.path.join(CORPUS, "adjoint", "data", "adjointWing_start.sim")
needs_small = pytest.mark.skipif(not os.path.isfile(SMALL), reason="语料缺失")
needs_official = pytest.mark.skipif(os.environ.get("STARDECODING_OFFICIAL") != "1",
                                    reason="需 STARDECODING_OFFICIAL=1（官方 -batch 较慢）")


# ---------------------------------------------------------------- 探测
def test_verified_apis_table_shape():
    assert sb.VERIFIED_APIS and all(isinstance(x, str) for x in sb.VERIFIED_APIS)
    assert any("getRegionManager" in a for a in sb.VERIFIED_APIS)
    assert any("saveState(" in a for a in sb.VERIFIED_APIS)


def test_candidate_exes_missing_roots_is_empty():
    assert sb.candidate_exes([r"Z:\\definitely_not_here"]) == []


def test_bridge_status_reports_unavailable_without_candidates(monkeypatch):
    monkeypatch.delenv("STARCCM_HOME", raising=False)
    monkeypatch.setattr(sb, "ROOT_HINTS", (r"Z:\\definitely_not_here",))
    st = sb.bridge_status()
    assert st["available"] is False and st["exe"] is None
    assert "starccmw.exe" in st["reason"]
    out = sb.official_run(SMALL, sb.SMOKE_MACRO, "X", timeout=5)
    assert out["ok"] is False and out.get("skipped") is True
    assert sb.official_open_stats(SMALL)["skipped"] is True
    assert sb.compare_official_view(SMALL)["skipped"] is True
    assert sb.official_resave(SMALL)["skipped"] is True


def test_find_star_exe_or_skip():
    exe = sb.find_star_exe()
    if exe is None:
        pytest.skip("本机未安装 STAR-CCM+（桥不可用属预期）")
    assert exe.lower().endswith("starccmw.exe") and os.path.isfile(exe)


# ---------------------------------------------------------------- 宏模板
def test_macro_templates_only_use_known_apis():
    for name, macro in (("smoke", sb.SMOKE_MACRO), ("stats", sb.OPEN_STATS_MACRO),
                        ("resave", sb.RESAVE_MACRO), ("run_case", sb.RUN_CASE_MACRO),
                        ("mesh_case", sb.MESH_CASE_MACRO)):
        assert "package macro;" in macro and "extends StarMacro" in macro, name
        assert "getActiveSimulation()" in macro, name
        assert "sim.println(" in macro, name
        # 之前踩过的自造 API 必须不出现
        for bad in ("getStarVersion", "sim.getVersion(", "createCoSimulation(",
                    "sim.getSolver("):
            assert bad not in macro, (name, bad)
    for label in ("Part", "Region", "Scene", "Continuum", "Report", "Plot", "Monitor"):
        assert ("dump(sim, \"%s\"" % label) in sb.OPEN_STATS_MACRO


def test_mesh_case_macro_uses_only_verified_apis():
    """S6 同网格官方算例宏：模板只含 VERIFIED_APIS 已核对的调用链。"""
    text = sb.MESH_CASE_MACRO % {"cgns": "D:/tmp/m.cgns", "out": "D:/tmp/o.sim",
                                 "target": 300, "rho": "1.0", "mu": "1e-05",
                                 "u_in": "0.05", "poll": 50}
    assert 'String cgns = "D:/tmp/m.cgns";' in text
    assert 'String out = "D:/tmp/o.sim";' in text
    assert "long target = 300;" in text
    assert "double mu = 1e-05;" in text and "double rho = 1.0;" in text
    for api in ("importFile(", "newRegionsFromParts(", "createContinuum(",
                "setBoundaryType(", "setPhysicsContinuum(", "getSimulationIterator()",
                "getMaterialProperties().getMaterialProperty("):
        assert api in text, api
    # 诚实失败路径：任何一步失败都要打印 SAMEMESH_FAIL（绝不静默吞掉）
    assert text.count("SAMEMESH_FAIL") >= 5 and "SAMEMESH_DONE" in text
    # getObjects() 逐 Object 迭代 + instanceof GeometryPart 过滤后
    # 才能喂 newRegionsFromParts(Collection<GeometryPart>, ...)
    # （Part 与 GeometryPart 是无关类型，直接 for(Part) + cast 编译不过）
    assert "po instanceof GeometryPart" in text


def test_mesh_case_wrapper_offline_guards(monkeypatch):
    """无 exe / 无 CGNS：诚实跳过或失败，不碰官方批。"""
    monkeypatch.delenv("STARCCM_HOME", raising=False)
    monkeypatch.setattr(sb, "ROOT_HINTS", (r"Z:\\definitely_not_here",))
    r = sb.official_mesh_case(SMALL, "Z:/no/such.cgns", "Z:/o.sim")
    assert r["ok"] is False and "CGNS" in r["reason"]
    # 模板占位符与封装一一对应（少一个 %s 就会 KeyError/漏注入）
    keys = set(re.findall(r"%\((\w+)\)[sd]", sb.MESH_CASE_MACRO))
    assert keys == {"cgns", "out", "target", "rho", "mu", "u_in", "poll"}, keys


def test_resave_macro_formatting_uses_forward_slashes():
    text = sb.RESAVE_MACRO % {"out": "D:/tmp/x.sim"}
    assert 'String out = "D:/tmp/x.sim";' in text


def test_our_counts_mapping_on_real_file():
    if not os.path.isfile(SMALL):
        pytest.skip("语料缺失")
    from sim_parser import SimFile
    counts = sb._our_counts(SimFile(SMALL))
    assert counts["Region"] >= 1 and counts["Scene"] >= 1 and counts["Part"] >= 1
    assert set(counts) == {"Region", "Scene", "Continuum", "Report", "Plot", "Monitor", "Part"}


def test_compare_official_resave_self_is_perfect():
    if not os.path.isfile(SMALL):
        pytest.skip("语料缺失")
    diff = sb.compare_official_resave(SMALL, SMALL)
    assert diff["ok"] and diff["n_value_diffs"] == 0 and diff["n_ref_diffs"] == 0
    assert diff["match_rate"] == 100.0 and diff["only_in_src"] == 0
    assert diff["verdicts"]["structure_survived"]
    assert diff["verdicts"]["value_fields_stable"]


def test_object_key_uses_class_and_name():
    class _O:
        class_name = "star.common.Region"
        name = "Fluid Domain"

    assert sb._key_of(_O()) == ("star.common.Region", "Fluid Domain")

    class _N:
        class_name = "star.common.Region"
        name = None

    assert sb._key_of(_N()) == ("star.common.Region", None)


# ---------------------------------------------------------------- 官方集成（慢）
@needs_official
@needs_small
def test_official_smoke_on_copy():
    res = sb.official_smoke(SMALL, timeout=900)
    if res.get("skipped"):
        pytest.skip(res.get("reason"))
    assert res["ok"], res.get("reason") + (res.get("log", "")[-400:])
    assert res["sim_name"] and res["official_regions"] >= 1


@needs_official
@needs_small
def test_official_roundtrip_preserves_our_edit(tmp_path=None):
    """S1 核心：我们的编辑 → 官方打开 → 官方重存 → 编辑仍在 + 值型字段稳定。"""
    import shutil
    import tempfile

    from sim_parser import SimFile
    from sim_writer import save_sim

    tmp = tempfile.mkdtemp(prefix="s1_rt_")
    try:
        sim = SimFile(SMALL)
        target = next(o for o in sim.objects
                      if str(o.dict.get("PresentationName")) == "Fluid Domain")
        new_name = "S1 RoundTrip Name"
        target.dict["PresentationName"] = new_name
        ours = os.path.join(tmp, "ours.sim")
        save_sim(sim, ours, patches={target.id: {"PresentationName": new_name}})
        assert SimFile(ours).objmap[target.id].dict["PresentationName"] == new_name

        resaved = os.path.join(tmp, "official_resaved.sim")
        rs = sb.official_resave(ours, resaved, timeout=1500)
        if rs.get("skipped"):
            pytest.skip(rs.get("reason"))
        assert rs["ok"], rs.get("reason") + (rs.get("log") or "")[-300:]

        rb = SimFile(resaved)
        assert any(str(o.dict.get("PresentationName")) == new_name for o in rb.objects), \
            "我们的编辑必须穿过官方重存"
        diff = sb.compare_official_resave(ours, resaved)
        assert diff["verdicts"]["structure_survived"], diff["match_rate"]
        assert diff["verdicts"]["value_fields_stable"], diff["value_diffs"][:5]
        assert diff["name_field_diffs"] == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@needs_official
@needs_small
def test_official_view_matches_our_parser():
    res = sb.compare_official_view(SMALL, timeout=900)
    if res.get("skipped"):
        pytest.skip(res.get("reason"))
    assert res["ok"], res.get("reason")
    rows = res["rows"]
    for label in ("Region", "Scene", "Continuum"):
        assert rows[label]["official"] > 0
        assert rows[label]["delta"] == 0, (label, rows[label])

# -*- coding: utf-8 -*-
"""S 波 S3：解场抽取的**根节点变体**（SolutionRepresentation vs FvRepresentation）。

背景：`extract_solution_fields` 原先只认 `star.post.SolutionRepresentation`；
**官方新求解并保存**的文件（例如本仓库官方桥生成的 airfoil_official_*.sim）不创建该对象，
解场直接挂在 `star.common.FvRepresentation` 的 cells DUP 组上 → 旧实现诚实拒绝（ok=False）。
本用例固定两类根节点的行为与"两无"时的诚实拒绝。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest  # noqa: E402

from sim_parser import SimFile  # noqa: E402

GENERATED = os.path.join(ROOT, "benchmarks", "official", "airfoil_official_2000.sim")
TUTORIAL_SOLVED = "D:/training/openfoam/benchmark/vortexShed_tutor_v3_0.05_2502.sim"
NO_SOLUTION = os.path.join(ROOT, "adjointWing_start.sim")
PIPE_BLOCKAGE = "D:/training/starccm/startutorialsdata/optimate/data/pipeBlockage.sim"


def test_generated_file_uses_fvrepresentation_root():
    if not os.path.isfile(GENERATED):
        pytest.skip("官方桥生成件缺失（可跑 star_bridge.official_run_case 生成）")
    sim = SimFile(GENERATED)
    assert not [o for o in sim.objects
                if (o.class_name or "") == "star.post.SolutionRepresentation"], \
        "该文件按设计不含 SolutionRepresentation（正是本用例要覆盖的变体）"
    sf = sim.extract_solution_fields()
    assert sf["ok"], sf.get("reason")
    assert sf["cell_count"] == 16987 and sf["region_name"] == "Domain"
    assert sf["n_fields"] >= 10
    names = {f["name"] for f in sf["fields"]}
    for key in ("Pressure", "Density", "EffectiveViscosity"):
        assert key in names, sorted(names)
    assert sf["data"]["Pressure"].shape == (16987,)
    assert float(sf["data"]["Density"].min()) > 0.0


@pytest.mark.skipif(not os.path.isfile(TUTORIAL_SOLVED), reason="官方语料缺失")
def test_solutionrepresentation_root_still_works():
    sf = SimFile(TUTORIAL_SOLVED).extract_solution_fields()
    assert sf["ok"] and sf["cell_count"] == 20245 and sf["region_name"] == "Fluid_Domain"
    assert sf["n_fields"] >= 8
    assert "Pressure" in sf["data"] and sf["data"]["Pressure"].shape == (20245,)


def test_no_solution_is_honest():
    """未求解文件：有表示但无字段存储 → 诚实拒绝（不假装有解场）。"""
    if not os.path.isfile(NO_SOLUTION):
        pytest.skip("样例缺失")
    sf = SimFile(NO_SOLUTION).extract_solution_fields()
    assert sf["ok"] is False and sf["n_fields"] == 0 and sf["data"] == {}
    assert ("无字段存储" in sf["reason"]) or ("无解场表示" in sf["reason"])
    if "无解场表示" in sf["reason"]:
        assert "SolutionRepresentation" in sf["reason"] and "FvRepresentation" in sf["reason"]


@pytest.mark.skipif(not os.path.isfile(PIPE_BLOCKAGE), reason="官方语料缺失")
def test_geometry_only_fields_are_not_a_solution():
    """未求解文件的 cells 组只挂几何索引字段 → 不构成解场（S3 假通过修复）。

    实测 pipeBlockage.sim：虽有 star.common.FvRepresentation(id=130)，其 cells DUP
    组只含 CellGeometryPartIndex / ProstarCellIndex（无 Pressure/Velocity 等物理量）。
    根节点回退到 FvRepresentation 后曾把这两个几何字段当成解场 → ok=True（假通过）。
    现在：全部字段命中几何索引模式时按诚实拒绝返回，fields/data 置空，
    几何字段名另列在 geometry_only_fields（便于诊断，不冒充解场）。
    """
    sim = SimFile(PIPE_BLOCKAGE)
    assert [o for o in sim.objects
            if (o.class_name or "") == "star.common.FvRepresentation"], \
        "该文件按设计只有 FvRepresentation 根（正是本用例要覆盖的变体）"
    sf = sim.extract_solution_fields()
    assert sf["ok"] is False and sf["n_fields"] == 0 and sf["data"] == {}
    assert "无解场数据" in sf["reason"], sf["reason"]
    geom = sf.get("geometry_only_fields") or []
    assert "CellGeometryPartIndex" in geom and "ProstarCellIndex" in geom
    assert sf["cell_count"] == 14882 and sf.get("region_name") == "Region 1"

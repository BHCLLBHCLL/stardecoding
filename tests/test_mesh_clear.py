# -*- coding: utf-8 -*-
"""S5 第二轮：Mesh>Clear 的**内核级**实现（体网格存储组识别 / 真删除 / 持久化 / 恢复）。

背景（S5 第一轮遗留的 needs_kernel）：旧 cmd_clear_mesh 只清会话状态（生成结果 +
显示 actor），对象图与 .sim 不变。本轮实现对象图级真删除：
  · sim.volume_mesh_groups()     —— 与 extract_volume_mesh 同源的角色标签集，但返回
    **全部**带拓扑角色的 DuplicateStorageManager（含重复副本 3087/3092 —— 只删一个
    副本时抽取仍会成功，所以必须删干净）+ 网格绑定场组（SerialSize 恰等于拓扑组尺寸）
    + 被引用的 SimpleStorage/ListStorage 对象；
  · sim.clear_volume_mesh_patch() —— 只算补丁，不改对象（供文档层应用/撤销）；
  · SessionDocument.clear_volume_mesh()/restore_volume_mesh() —— 标记删除 + 解除 Keys
    引用 + 清缓存；保存经 sim_writer.save_sim(deleted=…) 落盘。

验收：删除后 extract_volume_mesh() 必须 ok=False；保存→重开仍然 ok=False 且对象数减少；
恢复后 ok=True。无 .sim 时必须诚实拒绝。
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest  # noqa: E402

from sim_parser import SimFile  # noqa: E402
from star_gui_document import SimDocument  # noqa: E402

CYL = "D:/training/openfoam/benchmark/vortexShed_tutor_v3_0.05_2502.sim"
LOCAL = os.path.join(ROOT, "adjointWing_start.sim")


@pytest.fixture(scope="module")
def sim_path():
    for p in (CYL, LOCAL):
        if os.path.exists(p):
            return p
    pytest.skip("缺少 .sim 语料（官方算例或仓库自带 adjointWing_start.sim）")


def test_volume_mesh_groups_identify_roles(sim_path):
    sim = SimFile(sim_path)
    g = sim.volume_mesh_groups()
    assert g["ok"], g.get("reason")
    assert len(g["topology"]) >= 3           # 顶点 / 面 / 单元至少三组
    tags = {t for hit in g["roles"].values() for t in hit}
    assert "Coord" in tags and "VertexList" in tags and "FaceCellIndex" in tags
    # 拓扑组尺寸必须覆盖真实抽取结果（顶点/面/单元计数）
    vm = sim.extract_volume_mesh()
    assert vm["ok"]
    sizes = set(g["sizes"].values())
    assert int(vm["count"]) in sizes         # 单元数
    assert len(g["storage"]) > 0             # 引用的存储对象


def test_clear_volume_mesh_is_real_and_restorable(sim_path):
    sim = SimFile(sim_path)
    doc = SimDocument(sim, sim_path)
    assert sim.extract_volume_mesh()["ok"]
    res = doc.clear_volume_mesh(include_fields=True)
    assert res["ok"], res.get("reason")
    assert res["by_role"]["topology"] >= 3
    assert len(res["removed"]) == sum(res["by_role"].values())
    # 真删除：抽取必须失败（不再假装有网格）
    vm = sim.extract_volume_mesh()
    assert not vm["ok"]
    assert "存储" in vm["reason"] or "存储组" in vm["reason"]
    # 恢复：对象图从未被破坏 → 抽取重新可用
    back = doc.restore_volume_mesh()
    assert back["ok"] and back["restored"] == len(res["removed"])
    assert sim.extract_volume_mesh()["ok"]


def test_cleared_mesh_persists_through_save(sim_path):
    from sim_writer import save_sim
    sim = SimFile(sim_path)
    n_before = len(sim.objects)
    doc = SimDocument(sim, sim_path)
    res = doc.clear_volume_mesh(include_fields=True)
    assert res["ok"]
    tmp = os.path.join(tempfile.mkdtemp(prefix="meshclear_"), "cleared.sim")
    save_sim(sim, tmp, patches=doc.patches, created=doc.created,
             src_path=sim_path, deleted=doc.deleted)
    assert os.path.getsize(tmp) > 0
    re_sim = SimFile(tmp)
    assert len(re_sim.objects) <= n_before - len(res["removed"])
    vm = re_sim.extract_volume_mesh()
    assert not vm["ok"], "清除后重开的文件里仍有体网格"
    g = re_sim.volume_mesh_groups()
    assert not g["ok"]


def test_clear_without_sim_is_honest():
    doc = SimDocument()
    res = doc.clear_volume_mesh()
    assert not res["ok"]
    assert "未绑定" in res["reason"]
    # 纯表面/无体网格存储的 .sim：补丁必须**诚实拒绝**并给出原因
    for path in (LOCAL, CYL):
        if not os.path.exists(path):
            continue
        sim = SimFile(path)
        p = sim.clear_volume_mesh_patch(include_fields=False)
        assert (p["ok"] and p["delete"]) or (not p["ok"] and p["reason"])
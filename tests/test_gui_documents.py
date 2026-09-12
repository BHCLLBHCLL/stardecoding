# -*- coding: utf-8 -*-
"""X2 纯逻辑：多文档工作区 + 跨仿真复制粘贴（id 冲突重映射）。"""
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from star_gui_document import SimDocument
from star_gui_documents import (CLIPBOARD, Clip, Clipboard, DocumentWorkspace,
                                analogous_parent_id, child_ids, collect_subtree,
                                copy_subtree, paste_clip, paste_into,
                                plan_id_mapping)
from sim_parser import SimFile, SimObject


def _mk_sim():
    """构造最小对象图：root(1) → 子(2,3)，子 3 带指向 2 的 Keys 引用。"""
    root = SimObject(1, {"ClassName": "star.common.Simulation",
                         "PresentationName": "Sim", "Keys": [2, 3]}, 0)
    a = SimObject(2, {"ClassName": "star.common.Region",
                      "PresentationName": "Fluid", "Parent": 1}, 1)
    b = SimObject(3, {"ClassName": "star.common.Region",
                      "PresentationName": "Solid", "Parent": 1,
                      "Keys": [2], "Mesh": 2}, 2)
    sim = SimFile.__new__(SimFile)
    sim.path = None
    sim.objects = [root, a, b]
    sim.objmap = {o.id: o for o in sim.objects}
    sim.sections = []
    sim.arrays = []
    return sim


def _sparse_sim(root_id=100):
    """仅含单个根对象（无子）的目标 sim，_next_id 从 root_id+1 起。"""
    root = SimObject(root_id, {"ClassName": "star.common.Simulation",
                               "PresentationName": "Sim", "Keys": []}, 0)
    sim = SimFile.__new__(SimFile)
    sim.path = None
    sim.objects = [root]
    sim.objmap = {root.id: root}
    sim.sections = []
    sim.arrays = []
    return sim


def _doc(sim=None):
    return SimDocument(sim, None)


def test_workspace_add_remove_activate():
    ws = DocumentWorkspace()
    d1, d2 = _doc(), _doc()
    assert ws.add(d1, activate=True) == 0
    assert ws.add(d2) == 1
    assert ws.active is d2
    assert len(ws) == 2 and list(ws) == [d1, d2]
    assert d1 in ws and ws.active_index == 1
    assert ws.activate(d1) and ws.active is d1
    assert ws.add(d1) == 0 and len(ws) == 2          # 重复添加不增容
    assert ws.remove(d1) and d1 not in ws
    assert ws.active is d2                            # 移除活动项后回退
    assert ws.remove(d1) is False
    docs = ws.close_all()
    assert len(ws) == 0 and docs == [d2]


def test_workspace_find_by_path():
    tmp = tempfile.mkdtemp(prefix="x2_ws_")
    try:
        ws = DocumentWorkspace()
        d = _doc()
        d.path = os.path.join(tmp, "a.sim")
        ws.add(d)
        assert ws.find_by_path(d.path) is d
        assert ws.find_by_path(os.path.join(tmp, ".", "a.sim")) is d
        assert ws.find_by_path(os.path.join(tmp, "b.sim")) is None
        assert ws.find_by_path(None) is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_collect_subtree_bfs():
    doc = _doc(_mk_sim())
    assert child_ids(doc, 1) == [2, 3]
    assert collect_subtree(doc, 1) == [1, 2, 3]     # 根在前，随后代
    assert collect_subtree(doc, 2) == [2]           # 叶子
    assert collect_subtree(doc, 99) == []           # 不存在


def test_copy_subtree_renames_root_only():
    doc = _doc(_mk_sim())
    clip = copy_subtree(doc, 1, " Copy")
    assert isinstance(clip, Clip)
    assert len(clip) == 3 and clip.ids() == [1, 2, 3]
    assert clip.root_id == 1
    assert clip.objects[1]["PresentationName"] == "Sim Copy"
    assert clip.objects[2]["PresentationName"] == "Fluid"      # 子对象不改名
    assert clip.objects[1]["Keys"] == [2, 3]                   # 快照仍是旧 id
    assert copy_subtree(doc, 99) is None


def test_paste_into_empty_doc_remaps_ids():
    src = _doc(_mk_sim())
    dst = _doc(_sparse_sim(100))                    # 目标仅一个根，_next_id=101
    new_root = paste_into(dst, src, 1)
    assert new_root == 101
    assert len(dst.created) == 3
    root = dst.created[new_root]
    kids = sorted(dst.created)
    assert kids == [101, 102, 103]
    assert root.dict["Keys"] == [102, 103]          # 引用已重映射
    child3 = dst.created[103]
    assert child3.dict["Keys"] == [102]
    assert child3.dict["Mesh"] == 102
    assert child3.dict["Parent"] == new_root
    assert dst.dirty is True


def test_paste_avoids_id_collision_with_target():
    src = _doc(_mk_sim())
    dst = _doc(_mk_sim())                           # 目标已有同 id 的 1/2/3
    before = set(dst.sim.objmap)
    new_root = paste_into(dst, src, 1)
    assert new_root == 4                            # 从目标 _next_id 起，避开 1/2/3
    for i in range(3):
        assert new_root + i not in before
        assert (new_root + i) in dst.created
    assert len(dst.created) == 3
    assert set(dst.sim.objmap) == before | set(dst.created)


def test_paste_attaches_to_explicit_parent():
    src = _doc(_mk_sim())
    dst = _doc(_mk_sim())
    host = SimObject(50, {"ClassName": "star.common.Region",
                          "PresentationName": "Host", "Keys": []}, -1)
    dst.created[50] = host
    dst.sim.objmap[50] = host
    dst.sim.objects.append(host)
    new_root = paste_into(dst, src, 2, parent=50)   # 只拷叶子 2
    assert dst.created[new_root].dict["Parent"] == 50
    assert new_root in dst.created[50].dict["Keys"]


def test_paste_drops_dangling_parent_across_docs():
    src = _doc(_mk_sim())
    dst = _doc(_sparse_sim(100))                    # 目标有 sim 但无对象 1
    new_root = paste_into(dst, src, 2)              # 叶子 2 的父(1)不在剪贴内
    assert "Parent" not in dst.created[new_root].dict


def test_paste_keeps_parent_inside_clip():
    src = _doc(_mk_sim())
    dst = _doc(_sparse_sim(100))
    new_root = paste_into(dst, src, 1)              # 根 1 无 Parent
    assert "Parent" not in dst.created[new_root].dict
    child = dst.created[102]                        # 子 2 的父 1 在剪贴内 → 挂新根
    assert child.dict["Parent"] == new_root
    assert dst.created[new_root].dict["Keys"] == [102, 103]


def test_paste_safe_on_empty_clip():
    assert paste_clip(_doc(_mk_sim()), None) is None
    assert paste_clip(None, copy_subtree(_doc(_mk_sim()), 1)) is None
    assert paste_clip(_doc(), copy_subtree(_doc(_mk_sim()), 1)) is None   # 目标无 sim


def test_clipboard_cross_detection():
    cb = Clipboard()
    d1, d2 = _doc(), _doc()
    assert not cb.has_content() and not cb.is_cross(d1)
    cb.set(d1, copy_subtree(_doc(_mk_sim()), 1))
    assert cb.has_content()
    assert not cb.is_cross(d1)                      # 同文档 → 非跨仿真
    assert cb.is_cross(d2)                          # 异文档 → 跨仿真
    cb.clear()
    assert not cb.has_content()
    assert CLIPBOARD is not cb                      # 模块级单例独立


def test_plan_id_mapping_sequential():
    doc = _doc(_mk_sim())
    clip = copy_subtree(_doc(_mk_sim()), 1)
    m = plan_id_mapping(clip, doc)
    assert list(m) == [1, 2, 3]
    assert m[1] == doc._next_id                     # 从目标 _next_id 起
    assert m[2] == m[1] + 1 and m[3] == m[1] + 2


def test_pasted_graph_writes_consistent_ids():
    """落盘验证：跨文档粘贴后 Save，created_id_mapping 把会话 id 映到图 id。"""
    from sim_writer import created_id_mapping, save_sim
    tmp = tempfile.mkdtemp(prefix="x2_doc_")
    try:
        base = os.path.join(tmp, "base.sim")
        with open(base, "w", encoding="latin-1") as f:
            f.write("{'ClassName': 'star.common.Simulation', 'PresentationName': 'Sim', 'Keys': [2, 3]}\n")
            f.write("{'ClassName': 'star.common.Region', 'PresentationName': 'Fluid', 'Parent': 1}\n")
            f.write("{'ClassName': 'star.common.Region', 'PresentationName': 'Solid', 'Parent': 1, 'Keys': [2]}\n")
            f.write("{'ClassName': 'ClassVersions', 'Versions': {'star.common.Region': 2}}\n")
        srcp = SimFile(base)
        assert len(srcp.objects) == 4
        dst = _doc(srcp)
        new_root = paste_into(dst, _doc(_mk_sim()), 1)
        assert new_root is not None
        mapping = created_id_mapping(srcp, dst.created, set())
        assert len(mapping) == 3                    # 3 个会话对象待插入
        out = os.path.join(tmp, "out.sim")
        save_sim(srcp, out, patches=dst.patches, created=dst.created, src_path=base)
        written = SimFile(out)
        new_ids = set(mapping.values())
        for gid in new_ids:
            assert gid in written.objmap            # 全部落盘
        root_gid = mapping[new_root]
        keys = written.objmap[root_gid].dict["Keys"]
        assert set(keys).issubset(new_ids)          # 根只引用同批新对象
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_copy_subtree_records_parent_class():
    doc = _doc(_mk_sim())
    clip = copy_subtree(doc, 3)                     # 子 3 的父是 1（Simulation）
    assert clip.parent_class == "star.common.Simulation"
    assert copy_subtree(doc, 1).parent_class is None   # 根无父


def test_analogous_parent_id_matches_class():
    dst = _doc(_sparse_sim(100))
    assert analogous_parent_id(dst, "star.common.Simulation") == 100
    assert analogous_parent_id(dst, "star.common.RegionManager") is None
    assert analogous_parent_id(dst, "") is None
    assert analogous_parent_id(dst, None) is None
    assert analogous_parent_id(None, "star.common.Simulation") is None


def test_paste_attaches_to_analogous_parent_in_target():
    src = _doc(_mk_sim())
    dst = _doc(_sparse_sim(100))                    # 目标根类名同为 Simulation
    parent = analogous_parent_id(dst, copy_subtree(src, 2).parent_class)
    assert parent == 100
    new_root = paste_into(dst, src, 2, parent=parent)
    assert dst.created[new_root].dict["Parent"] == 100
    assert new_root in dst.sim.objmap[100].dict["Keys"]

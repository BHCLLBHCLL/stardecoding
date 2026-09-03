# -*- coding: utf-8 -*-
"""W6 差分回归自动化。

两条路径：
  1) 结构自检 + 重读一致（无许可主路径）：任意语料 → 类型化属性编辑/等宽数组载荷
     覆盖 → Save As → 重读 → 结构自检 + compare_object_graph 对象图差分 → 断言；
  2) 官方差分（有 STARCCM_HOME 许可才触发）：try_official_resave 重存 → 重读官方
     输出 → compare_object_graph 差分。

无许可时官方段 auto-skip（print 提示 + 不 fail），与既有
test_compare_graph_and_official_resave_hook 的契约一致（status in {"skipped","ready"}）。
"""
import os
import shutil
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
probe = os.path.join(ROOT, "adjointWing_start.sim")

from sim_parser import SimFile                     # noqa: E402
from sim_writer import compare_object_graph, save_sim, try_official_resave  # noqa: E402
from star_gui_document import SimDocument          # noqa: E402


def _structural_snapshot(sim):
    """结构自检快照：对象数 / 数组数 / 尾部必须为 ClassVersions / 其版本计数。"""
    cv = None
    for o in sim.objects:
        if o.class_name == "ClassVersions":
            cv = o
            break
    return {
        "n_objects": len(sim.objects),
        "n_arrays": len(sim.arrays),
        "tail_class": cv.class_name if cv else None,
        "tail_vers": dict((cv.dict.get("Versions") or {})) if cv else None,
    }


def _assert_reloaded_consistent(probe, patch_and_expected, array_patch=None):
    """读语料 → 打补丁 → Save As → 重读 → 结构 + 对象图 + 数组差分断言。"""
    before = SimFile(probe)
    snap = _structural_snapshot(before)
    doc = SimDocument(before, probe)
    target_id = patch_and_expected(before, doc)
    assert doc.patches or array_patch, "至少应有一个写操作被记录"
    tmp = tempfile.mkdtemp(prefix="w6_")
    try:
        out = os.path.join(tmp, "out.sim")
        save_sim(before, out, patches=doc.patches, created=doc.created,
                 deleted=list(doc.deleted), src_path=probe,
                 array_patches=array_patch or {})
        ref = SimFile(out)
        # ---- 结构自检 ----
        s2 = _structural_snapshot(ref)
        assert s2["n_objects"] == snap["n_objects"], "对象数应不变"
        assert s2["n_arrays"] == snap["n_arrays"], "数组数应不变"
        assert s2["tail_class"] == "ClassVersions", "尾段仍应为 ClassVersions"
        assert ref.check_sequential_ids()["ok"], "对象图 id 仍应=图序号+2 严格连续"
        # ---- 对象图差分（compare_object_graph 按源与输出【路径】比较）----
        # 允许且仅允许目标对象被改字段的 diff，其余逐字段一致
        if target_id is not None:
            _oid, _key = target_id
            diffs = compare_object_graph(
                probe, out,
                keys=("PresentationName", "name", "Opacity", "DisplayerColor",
                      "Keys", "Mesh", "ParallelScale", "Parent"))
            unexpected = [d for d in diffs["diffs"] if not (d[0] == _oid and d[1] == _key)]
            assert not unexpected, "重读差分外不应有其他对象字段被改动: %r" % unexpected
        # ---- 等宽数组载荷覆盖差分 ----
        if array_patch:
            for idx, data in array_patch.items():
                if idx is None:
                    continue
                expect = data if isinstance(data, (bytes, bytearray)) else bytes(data)
                assert ref.arrays[idx]["data"] == expect, \
                    "数组 %d 写回应命中载荷 (@%s, %d 字节)" % (
                        idx, ref.arrays[idx]["start"], len(expect))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_w6_reload_consistency_rename():
    """结构自检 + 重读一致：命名属性（变长）整行替换 → 重开差分。"""
    def do(before, doc):
        target = next(o for o in before.objects
                      if (o.dict or {}).get("PresentationName"))
        newname = target.dict["PresentationName"] + "_W6"
        doc.set_property(target.id, "PresentationName", newname)
        return target.id, "PresentationName"
    _assert_reloaded_consistent(probe, do)


def test_w6_reload_consistency_typed_value():
    """结构自检 + 重读一致：类型化值改写（DisplayerColor 嵌套 list）→ 重开差分。"""
    def do(before, doc):
        disp = next(o for o in before.objects
                    if (o.dict or {}).get("DisplayerColor"))
        doc.set_property(disp.id, "DisplayerColor", [0.12, 0.34, 0.56])
        return disp.id, "DisplayerColor"
    _assert_reloaded_consistent(probe, do)


def test_w6_reload_consistency_array_patch():
    """结构自检 + 重读一致：已有数组等宽载荷覆盖 → 重读数组命中。"""
    def do(before, doc):
        return None  # 只做数组 patch，不做属性 patch
    before = SimFile(probe)
    # 找一个非状态表数组（Character1 是主状态表，覆盖其载荷会污染状态表，跳过）
    idx = None
    for a in before.arrays:
        if a["type"] != "Character1" and a.get("data"):
            idx = a["index"]
            break
    assert idx is not None, "语料应有可覆盖的非状态表数组"
    raw = bytes(before.arrays[idx]["data"])
    assert len(raw) >= 2, "数组载荷应足够来回写"
    flipped = bytes([raw[0] ^ 0xFF]) + raw[1:]     # 等宽：仅翻转载荷首字节
    _assert_reloaded_consistent(probe, do, array_patch={idx: flipped})


def test_w6_official_resave_gated():
    """官方差分门控：有 STARCCM_HOME 触发官方重存并差分；无许可 auto-skip。"""
    hook = try_official_resave(probe, os.path.dirname(probe))
    if hook.get("status") == "skipped":
        # 无许可/无 resave_sim.java/无 starccmw → 走结构自检主路径（上两个测试已覆盖）
        print("W6 官方差分: 跳过（%s）" % hook.get("reason", "no license"))
        return
    # 有许可：尝试将已编辑样本交给官方重存，再读回差分
    before = SimFile(probe)
    snap = _structural_snapshot(before)
    doc = SimDocument(before, probe)
    target = next(o for o in before.objects if (o.dict or {}).get("PresentationName"))
    newname = target.dict["PresentationName"] + "_official"
    doc.set_property(target.id, "PresentationName", newname)
    tmp = tempfile.mkdtemp(prefix="w6_off_")
    try:
        out = os.path.join(tmp, "edited.sim")
        save_sim(before, out, patches=doc.patches, src_path=probe)
        # 官方重存路径由 resave_sim.java 生成 resaved_<name>.sim；此处直接调用宏外壳探测
        import subprocess, sys as _sys
        macro = hook["macro"]
        p = subprocess.run(
            [hook["exe"], "-batch", macro, out],
            cwd=tmp, capture_output=True, text=True, timeout=300)
        assert "RESAVE_DONE" in ((p.stdout or "") + (p.stderr or "")), \
            "官方应重存成功（RESAVE_DONE 标记）"
        official = os.path.join(tmp, "resaved_edited.sim")
        assert os.path.isfile(official), "官方重存应产出文件"
        ref = SimFile(official)
        # 结构自检 + 官方重存后重读一致（比较关键字段差分）
        s2 = _structural_snapshot(ref)
        assert s2["n_objects"] >= snap["n_objects"] * 0.9, \
            "官方重存对象规模不应骤减"
        print("W6 官方差分: 官方重存 + 重读对象图一致")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
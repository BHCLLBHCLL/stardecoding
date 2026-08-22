# -*- coding: utf-8 -*-
"""sim_parser.py 自检：结构不变量 + 导出有效性。"""
import json, sys
sys.path.insert(0, "D:/training/caedecoder/stardecoding")
from sim_parser import SimFile, walk_sections
from collections import Counter

sim = SimFile("D:/training/caedecoder/stardecoding/adjointWing_start.sim")

assert sim.header["StatePosition"] == 221991
assert sim.header["ClassName"] == "STAR"
assert len(sim.arrays) == 33
assert sim.arrays[0]["count"] == 36153
assert len(sim.state_text) == 36153
assert sim.state_magic[0] == "CD-adapco_STAR-CCM+_ID"
assert sim.state_magic[2] == "36120"
assert len(sim.tokens) == 7217, len(sim.tokens)
assert len(sim.objects) == 2076
assert sim.object_by_id(2).class_name == "star.common.Simulation"
assert sim.object_by_id(2).name == "adjointWing_start"
assert sim.object_by_id(3).class_name == "star.common.ManagerManager"
assert sim.object_by_id(4).class_name == "NameManager"

# walker 覆盖整个文件：最后一个分区（ClassVersions）的行延伸到文件末尾
sections = walk_sections(sim.blob)
last_d, last_payload, last_start, _ = sections[-1]
assert last_d.get("ClassName") == "ClassVersions"
tail = sim.blob[last_start:].decode("latin-1").strip()
assert tail.startswith("{") and tail.endswith("}")
print("sections: %d, last section (ClassVersions) covers %d-byte tail" % (
    len(sections), len(tail)))

kinds = Counter(r["kind"] for r in sim.records)
print("record kinds:", dict(kinds))
assert kinds.get("other", 0) == 0

fmts = sorted({r["fmt"] for r in sim.records if r.get("fmt")})
print("formats:", fmts)

ptrs = [r for r in sim.records if r["kind"] == "pointer"]
unres = sorted({r["ref"] for r in ptrs if r["ref"] not in sim.objmap})
print("pointers: %d, unresolved refs: %s" % (len(ptrs), unres))

sim.export("D:/training/caedecoder/stardecoding/export_check")
objs = json.load(open("D:/training/caedecoder/stardecoding/export_check/objects.json", encoding="utf-8"))
recs = json.load(open("D:/training/caedecoder/stardecoding/export_check/state_records.json", encoding="utf-8"))
assert len(objs) == 2076 and len(recs) == len(sim.records)
print("export JSON OK:", len(objs), "objects,", len(recs), "records")

# --- 改进①：语义字典 / 分层 / 全量建树 / 校验 ---
from semantic_dict import layer_of, resolve_class, attr_direction
assert layer_of("star.vis.Scene") == "visualization"
assert layer_of("star.cadmodeler.CadModel") == "cad-geometry"
assert layer_of("star.material.Gas") == "materials"
assert layer_of("star.common.Region") == "core"
assert resolve_class("star.common.XyPlot") == "star.common.Cartesian2DPlot"
assert resolve_class("star.meshing.PolyhedralMesher") == "star.dualmesher.DualAutoMesher"
assert attr_direction("Keys") == "down" and attr_direction("Parent") == "up"
assert attr_direction("PostSweeps") is None  # 数值属性不当引用

census, named = sim.layer_census()
assert census["core"] > 1000 and census["materials"] > 0
print("layer census:", dict(census.most_common(6)), "...")

main_roots = [r for r in sim.roots if sim.children.get(r.id)]
assert len(main_roots) == 1 and main_roots[0].id == 2  # 全量建树：Simulation 为唯一主根
loose = [r for r in sim.roots if not sim.children.get(r.id)]
assert len(loose) < 100  # 语义字典建树后游离对象应大幅减少（原 304）
print("tree: single main root", main_roots[0].id, "; loose:", len(loose))

v = sim.validate_class_versions()
assert v["status"] == "diagnostic" and v["expected_classes"] > 400
print("classversions diagnostic:", v["expected_classes"], "classes,",
      v["matched"], "matched,", v["expected_total"], "/", v["actual_total"])

# --- 改进②：网格抽取 ---
m = sim.extract_mesh()
assert m["faces"] is not None and m["faces"].shape[0] == 2824
assert m["vertices"] is not None and m["vertices"].shape[0] == 1412
assert m["consistent"] is True
print("mesh:", m["faces"].shape[0], "faces,", m["vertices"].shape[0],
      "vertices, flags:", m["face_flag"], "/", m["vertex_flag"])
print("ALL CHECKS PASSED")

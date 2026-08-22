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
print("ALL CHECKS PASSED")

# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "D:/training/caedecoder/stardecoding")
from sim_parser import SimFile
from semantic_dict import attr_direction

sim = SimFile("D:/training/caedecoder/stardecoding/adjointWing_start.sim")
# 谁在语义边阶段把 parent_of[2] 设了？
for o in sim.objects:
    if o.class_name == "ClassVersions":
        continue
    for attr, v in o.dict.items():
        d = attr_direction(attr)
        if d is None:
            continue
        cands = [v] if isinstance(v, int) else ([x for x in v if isinstance(x, int)] if isinstance(v, list) else [])
        if 2 in cands:
            print("obj", o.id, o.class_name, "attr", attr, "=", v, "direction", d)
# Domain 对象内容
for o in sim.objects:
    if o.class_name == "Domain":
        print("\nDomain", o.id, ":", {k: v for k, v in o.dict.items() if k != "ClassName"})

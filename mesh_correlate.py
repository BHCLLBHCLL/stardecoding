# -*- coding: utf-8 -*-
"""网格抽取依据：对象图元数据（TriangleCount/VertexCount）与数组表的对应关系。"""
import json, os, sys, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "D:/training/caedecoder/stardecoding")
from sim_parser import SimFile
import numpy as _np

res = json.load(open("D:/training/caedecoder/stardecoding/batch_results.json", encoding="utf-8"))
for r in res:
    if not r.get("ok"):
        continue
    sim = SimFile(r["file"])
    # 对象图里的网格规模元数据
    meta = {}
    for o in sim.objects:
        d = o.dict
        for k in ("TriangleCount", "VertexCount", "nTriangles", "nVertices", "NumVertices"):
            if isinstance(d.get(k), int):
                meta.setdefault(k, []).append((o.class_name, d[k]))
    # 数组表候选
    f8 = [(a["index"], a["count"]) for a in sim.arrays if a["type"] == "Float8"]
    u4 = [(a["index"], a["count"]) for a in sim.arrays if a["type"] == "Unsigned4"]
    f8_div3 = [c for _, c in f8 if c % 3 == 0]
    u4_div3 = [c for _, c in u4 if c % 3 == 0]
    print("%-32s meta:%-28s | Float8%%3=%s | U4%%3=%s" % (
        os.path.basename(r["file"]),
        ", ".join("%s=%s" % (k, v[0][1]) for k, v in sorted(meta.items()))[:26],
        f8_div3[:4], u4_div3[:4]))

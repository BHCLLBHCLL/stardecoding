# -*- coding: utf-8 -*-
"""21 文件网格抽取交叉校验：抽取结果 vs 对象图元数据。"""
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "D:/training/caedecoder/stardecoding")
from sim_parser import SimFile

res = json.load(open("D:/training/caedecoder/stardecoding/batch_results.json", encoding="utf-8"))
ok = fail = skip = 0
print("%-32s %-14s %-10s %-10s %-8s %-8s %s" % ("file", "part-tris", "faces", "verts", "maxidx", "consist", "flags"))
for r in res:
    if not r.get("ok"):
        continue
    sim = SimFile(r["file"])
    try:
        m = sim.extract_mesh()
    except Exception as e:
        print("%-32s ERROR %s" % (os.path.basename(r["file"]), e))
        fail += 1
        continue
    tris = m["meta"]["TriangleCount"]
    nf = m["faces"].shape[0] if m["faces"] is not None else 0
    nv = m["vertices"].shape[0] if m["vertices"] is not None else 0
    if tris:
        match = nf == max(tris)
        if match and m.get("consistent"):
            ok += 1
        else:
            fail += 1
        print("%-32s %-14s %-10d %-10d %-8s %-8s %s/%s" % (
            os.path.basename(r["file"]), str(sorted(tris, reverse=True)[:3]), nf, nv,
            str(m.get("max_index", 0)), str(m.get("consistent")),
            m["face_flag"], m["vertex_flag"]))
    else:
        skip += 1
        print("%-32s %-14s %-10d %-10d %-8s %-8s %s/%s" % (
            os.path.basename(r["file"]), "-", nf, nv, str(m.get("max_index", 0)),
            str(m.get("consistent")), m["face_flag"], m["vertex_flag"]))
print("\n主 part 面数匹配且索引一致: %d, 不一致: %d, 无元数据: %d" % (ok, fail, skip))

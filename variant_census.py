# -*- coding: utf-8 -*-
"""变体普查：头部字段、魔数形态、嵌套子块、编码模式。"""
import json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "D:/training/caedecoder/stardecoding")
from sim_parser import SimFile

res = json.load(open("D:/training/caedecoder/stardecoding/batch_results.json", encoding="utf-8"))
out = []
for r in res:
    if not r.get("ok"):
        continue
    f = r["file"]
    sim = SimFile(f)
    hdr_keys = sorted(k for k in sim.header.keys() if k != "ClassName")
    magic0 = sim.state_magic[0] if sim.state_magic else ""
    outer_id_style = "multi-id" if len(sim.state_magic) > 4 else ("nested-style" if magic0 and not magic0.startswith("CD-adapco") else "standard")
    banner_ver = re.search(r"version (\d+)", sim.state_banner or "")
    out.append({
        "file": os.path.basename(f),
        "header_keys": hdr_keys,
        "mode": sim.state_mode,
        "magic0": magic0[:30],
        "magic_lines": len(sim.state_magic),
        "magic_style": outer_id_style,
        "banner_ver": banner_ver.group(1) if banner_ver else None,
        "objects": len(sim.objects),
        "arrays": len(sim.arrays),
        "char_arrays": sum(1 for a in sim.arrays if a["type"] == "Character1"),
        "nested": len(sim.nested_transmits),
    })

json.dump(out, open("D:/training/caedecoder/stardecoding/variant_census.json", "w", encoding="utf-8"), indent=1)
print("%-30s %-6s %-10s %-8s %-4s %-4s %-6s %-6s %-12s" % ("file", "mode", "magic", "banner", "objs", "arr", "char", "nested", "header"))
for o in out:
    print("%-30s %-6s %-10s %-8s %-4d %-4d %-4d %-6d %s" % (
        o["file"], o["mode"], o["magic_style"], o["banner_ver"], o["objects"],
        o["arrays"], o["char_arrays"], o["nested"], ",".join(o["header_keys"])))

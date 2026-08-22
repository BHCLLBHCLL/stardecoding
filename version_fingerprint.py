# -*- coding: utf-8 -*-
"""版本指纹表：banner 版本 ↔ 头部字段组合 ↔ StarVersion ↔ 状态表编码。"""
import json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "D:/training/caedecoder/stardecoding")
from sim_parser import SimFile

res = json.load(open("D:/training/caedecoder/stardecoding/batch_results.json", encoding="utf-8"))
rows = []
for r in res:
    if not r.get("ok"):
        continue
    sim = SimFile(r["file"])
    banner = sim.state_banner or ""
    m = re.search(r"version (\d+)", banner)
    bver = m.group(1) if m else None
    sv = None
    for d, payload, s0, ps in sim.sections:
        if isinstance(d, dict) and d.get("ClassName") == "StarVersion":
            sv = d.get("ReleaseNumber")
            break
    hdr = ",".join(sorted(k for k in sim.header.keys() if k != "ClassName"))
    rows.append((bver, sv, sim.state_mode, hdr, os.path.basename(r["file"])))

print("%-10s %-10s %-6s %-34s %s" % ("banner", "release", "mode", "header", "file"))
for bver, sv, mode, hdr, f in sorted(rows, key=lambda x: (x[0] or "", x[1] or "")):
    print("%-10s %-10s %-6s %-34s %s" % (bver, sv, mode, hdr, f))

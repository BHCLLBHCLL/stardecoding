# -*- coding: utf-8 -*-
"""配对标例分析 v2：真正的 T 块记录。"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "D:/training/caedecoder/stardecoding")
from sim_parser import SimFile

orig = SimFile(r"D:\training\starccm\startutorialsdata\solidStress\data\vibratingPipe_start.sim")
resaved = SimFile(r"D:\training\caedecoder\stardecoding\resaved_vibratingPipe_start.sim")

def find_T(sim):
    out = []
    for r in sim.records:
        if r.get("fmt") == "T" and r.get("token_index", -1) >= 0:
            out.append(r)
    return out

t_orig = find_T(orig)
t_res = find_T(resaved)
print("T records (non-banner): orig", len(t_orig), " resaved", len(t_res))
for r in t_orig[:2]:
    print("ORIG T rec:", {k: v for k, v in r.items() if k in ("value", "token_index", "raw", "kind")})
    if r.get("raw"):
        raw = r["raw"]
        print("  raw hex (first 160 bytes):", raw[:480])
for r in t_res[:2]:
    print("RES T rec:", {k: v for k, v in r.items() if k in ("value", "token_index", "kind")})
    vals = r.get("values") or []
    print("  first 40 values:", [v["value"] for v in vals[:40]])
    print("  n values:", len(vals))

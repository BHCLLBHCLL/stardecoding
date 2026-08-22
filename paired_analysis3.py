# -*- coding: utf-8 -*-
"""比对 orig/resaved 的 T 块原始字节（同内容双二进制）。"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "D:/training/caedecoder/stardecoding")
from sim_parser import SimFile

def find_T_raw(sim):
    for r in sim.records:
        if r["kind"] == "bitmap" and r.get("bits") == "T":
            return r
    return None

orig = SimFile(r"D:\training\starccm\startutorialsdata\solidStress\data\vibratingPipe_start.sim")
resaved = SimFile(r"D:\training\caedecoder\stardecoding\resaved_vibratingPipe_start.sim")
t1 = find_T_raw(orig)
t2 = find_T_raw(resaved)
print("orig T raw bytes:", len((t1.get("raw") or "").split()) if t1 else None)
print("resaved T raw bytes:", len((t2.get("raw") or "").split()) if t2 else None)
if t1 and t2:
    b1 = bytes.fromhex(t1["raw"])
    b2 = bytes.fromhex(t2["raw"])
    print("identical:", b1 == b2, " len:", len(b1), len(b2))
    print("orig head:", b1[:80].hex(" "))
    print("res  head:", b2[:80].hex(" "))

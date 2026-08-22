# -*- coding: utf-8 -*-
"""配对标例分析：二进制 T 块原始字节 vs 重存 ASCII 版本的数值流。"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "D:/training/caedecoder/stardecoding")
from sim_parser import SimFile

orig = SimFile(r"D:\training\starccm\startutorialsdata\solidStress\data\vibratingPipe_start.sim")
resaved = SimFile(r"D:\training\caedecoder\stardecoding\resaved_vibratingPipe_start.sim")
print("orig mode:", orig.state_mode, " resaved mode:", resaved.state_mode)
print("orig records:", len(orig.records), " resaved records:", len(resaved.records))
print()
# 找 T 记录
t_orig = [r for r in orig.records if r.get("fmt") == "T" or (r.get("kind") == "anonymous" and r.get("fmt") == "T")]
t_res = [r for r in resaved.records if r.get("fmt") == "T"]
print("orig T records:", len(t_orig), " resaved T records:", len(t_res))
for r in t_orig[:3]:
    print("  ORIG T:", {k: v for k, v in r.items() if k in ("value", "raw", "token_index")}, "raw len:", len(r.get("raw") or ""))
for r in t_res[:3]:
    print("  RES  T:", {k: v for k, v in r.items() if k in ("value", "values", "token_index")})

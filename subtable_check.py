# -*- coding: utf-8 -*-
"""验证多 id 魔数 = 外层状态表内多个子表的长度列表。"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "D:/training/caedecoder/stardecoding")
from sim_parser import SimFile

for f in [r"D:\training\starccm\startutorialsdata\motion\data\openWaterPropeller_start.sim",
          r"D:\training\starccm\startutorialsdata\casting\data\basicGravitySandCastingGeo.sim"]:
    sim = SimFile(f)
    print("=== ", f.split("\\")[-1])
    magic = sim.state_magic
    print("magic lines:", len(magic), "first3:", magic[:3])
    if len(magic) >= 4 and magic[0] == "CD-adapco_STAR-CCM+_ID":
        n = int(magic[1])
        vals = [int(x) for x in magic[2:-1]]
        print("N =", n, " values:", vals[:8], "... total", len(vals))
        print("sum of values:", sum(vals))
        print("state table len:", len(sim.state_text))
        # 表格正文（去掉魔数行 + banner 行）的实际长度
        lines = sim.state_text.split("\n")
        banner_idx = next(k for k, l in enumerate(lines) if "TRANSMIT FILE" in l)
        body = "\n".join(lines[banner_idx+1:])
        print("body len:", len(body), " (wrap-free len:", len(body.replace(chr(10), "")), ")")

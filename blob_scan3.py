# -*- coding: utf-8 -*-
"""定位 T 块字节位置并转储其后字节流。"""
import sys, struct
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "D:/training/caedecoder/stardecoding")
from sim_parser import SimFile

sim = SimFile(r"D:\training\starccm\startutorialsdata\solidStress\data\vibratingPipe_start.sim")
text = sim.state_text
lines = text.split("\n")
bi = next(k for k, l in enumerate(lines) if "TRANSMIT FILE" in l)
blob = text.encode("latin-1")[len("\n".join(lines[:bi])) + 1:]

# 定位 "finger_block" 之后的 "CZ" ... "T"
i = blob.find(b"finger_block")
print("finger_block at", i)
seg = blob[i:i+120]
print("segment:", seg.hex(" "))
print("ascii:", seg.decode("latin-1", "replace"))

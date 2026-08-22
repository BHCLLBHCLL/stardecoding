# -*- coding: utf-8 -*-
"""在二进制状态表体里扫描与顶点坐标一致的 8 字节双精度值，定位 T 块结构。"""
import sys, struct
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "D:/training/caedecoder/stardecoding")
from sim_parser import SimFile
import numpy as _np

sim = SimFile(r"D:\training\starccm\startutorialsdata\solidStress\data\vibratingPipe_start.sim")
m = sim.extract_mesh()
verts = m["vertices"]
print("mesh:", verts.shape, "faces:", m["faces"].shape)
vset = set()
for row in verts:
    vset.update(float(x) for x in row)
print("unique vertex coords:", len(vset))

# 状态表二进制体（banner 之后）
text = sim.state_text
lines = text.split("\n")
bi = next(k for k, l in enumerate(lines) if "TRANSMIT FILE" in l)
blob = text.encode("latin-1")[len("\n".join(lines[:bi])) + 1:]
print("body bytes:", len(blob))

hits = []
for off in range(len(blob) - 8):
    d = struct.unpack_from("<d", blob, off)[0]
    if any(abs(d - v) < 1e-9 for v in vset):
        hits.append((off, d))
print("double hits:", len(hits))
if hits:
    off0 = hits[0][0]
    print("first hit at", off0, "value", hits[0][1])
    print("context bytes:", blob[max(0, off0-32):off0+40].hex(" "))

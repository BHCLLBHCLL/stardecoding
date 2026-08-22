# -*- coding: utf-8 -*-
"""定位 T 块：扫描与顶点坐标（非零）完全一致的 8 字节双精度。"""
import sys, struct
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "D:/training/caedecoder/stardecoding")
from sim_parser import SimFile

sim = SimFile(r"D:\training\starccm\startutorialsdata\solidStress\data\vibratingPipe_start.sim")
m = sim.extract_mesh()
verts = m["vertices"]
vset = {float(x) for row in verts for x in row if abs(float(x)) > 1e-8}
print("unique nonzero vertex coords:", len(vset))

text = sim.state_text
lines = text.split("\n")
bi = next(k for k, l in enumerate(lines) if "TRANSMIT FILE" in l)
blob = text.encode("latin-1")[len("\n".join(lines[:bi])) + 1:]
print("body bytes:", len(blob))

hits = []
for off in range(len(blob) - 8):
    d = struct.unpack_from("<d", blob, off)[0]
    if abs(d) > 1e-8 and any(abs(d - v) < 1e-9 for v in vset):
        hits.append((off, d))
print("exact double hits:", len(hits))
for off, d in hits[:10]:
    print("  off %5d value %.9g   context: %s | %s" % (
        off, d, blob[max(0, off-16):off].hex(" "), blob[off:off+16].hex(" ")))

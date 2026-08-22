# -*- coding: utf-8 -*-
"""21 文件长度自校验 + ZIP 容器回退测试。"""
import json, os, sys, zipfile
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "D:/training/caedecoder/stardecoding")
from sim_parser import SimFile

res = json.load(open("D:/training/caedecoder/stardecoding/batch_results.json", encoding="utf-8"))
ok = bad = skip = 0
print("%-32s %-6s %s" % ("file", "result", "detail"))
for r in res:
    if not r.get("ok"):
        continue
    sim = SimFile(r["file"])
    chk = sim.check_state_length()
    if chk["ok"] is True:
        ok += 1
    elif chk["ok"] is False:
        bad += 1
    else:
        skip += 1
    print("%-32s %-6s %s" % (os.path.basename(r["file"]),
                             "PASS" if chk["ok"] else ("FAIL" if chk["ok"] is False else "skip"),
                             chk["detail"]))
print("\nPASS %d, FAIL %d, skip %d" % (ok, bad, skip))

# 嵌套子块的长度自校验
nested_ok = nested_bad = 0
for r in res:
    if not r.get("ok"):
        continue
    sim = SimFile(r["file"])
    for nt in sim.nested_transmits:
        if nt.get("error"):
            continue
        magic = nt.get("magic") or []
        if len(magic) >= 3 and magic[-1] == "@":
            try:
                n = int(magic[0])
                vals = [int(x) for x in magic[1:-1]]
            except ValueError:
                continue
            if n >= 2 and len(vals) == n:
                # 嵌套多 id：sum == banner+表体长（脚本内近似为 count - 魔数块）
                if sum(vals) == nt["count"] - len("\n".join(magic)) - 1:
                    nested_ok += 1
                else:
                    nested_bad += 1
            elif len(vals) == 1 and vals[0] == nt["count"] - len("\n".join(magic)) - 1:
                nested_ok += 1
            else:
                nested_bad += 1
print("\n嵌套子块长度校验: PASS %d, FAIL %d" % (nested_ok, nested_bad))

# ZIP 容器回退测试（合成）
zf = "D:/training/caedecoder/stardecoding/ziptest.sim"
with zipfile.ZipFile(zf, "w", zipfile.ZIP_DEFLATED) as z:
    z.write("D:/training/caedecoder/stardecoding/adjointWing_start.sim", "inner.sim")
zsim = SimFile(zf)
print("ZIP fallback:", zsim.container_entry, "objects:", len(zsim.objects),
      "state:", len(zsim.state_text))
os.remove(zf)

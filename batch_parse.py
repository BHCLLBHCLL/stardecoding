# -*- coding: utf-8 -*-
"""批量解析 startutorialsdata 下所有 .sim，收集每个文件的诊断信息。"""
import json, os, sys, time, traceback
from collections import Counter
sys.path.insert(0, "D:/training/caedecoder/stardecoding")
from sim_parser import SimFile

root = r"D:\training\starccm\startutorialsdata"
files = []
for dirpath, _, names in os.walk(root):
    for n in names:
        if n.lower().endswith(".sim"):
            files.append(os.path.join(dirpath, n))
files.sort()

results = []
for f in files:
    t0 = time.time()
    rec = {"file": f, "size": os.path.getsize(f)}
    try:
        sim = SimFile(f)
        rec.update({
            "ok": True,
            "secs": round(time.time() - t0, 2),
            "sections": len(sim.sections),
            "arrays": len(sim.arrays),
            "array_types": dict(Counter(a["type"] for a in sim.arrays)),
            "state_len": len(sim.state_text or ""),
            "tokens": len(sim.tokens),
            "records": len(sim.records),
            "record_kinds": dict(Counter(r["kind"] for r in sim.records)),
            "fmts": sorted({r["fmt"] for r in sim.records if r.get("fmt")}),
            "other_tokens": [r.get("token") for r in sim.records if r["kind"] == "other"][:10],
            "objects": len(sim.objects),
            "starversion": None,
            "banner": (sim.state_banner or "").strip()[:90],
        })
        for d, payload, s0, ps in sim.sections:
            if isinstance(d, dict) and d.get("ClassName") == "StarVersion":
                rec["starversion"] = {k: v for k, v in d.items() if k != "ClassName"}
                break
    except Exception as e:
        rec.update({"ok": False, "error": repr(e),
                    "trace": traceback.format_exc().splitlines()[-3:],
                    "secs": round(time.time() - t0, 2)})
    results.append(rec)
    status = "OK " if rec.get("ok") else "FAIL"
    print("%-4s %8.1fKB %6.2fs %-70s" % (status, rec["size"]/1024, rec.get("secs", 0), os.path.basename(f)))
    if not rec.get("ok"):
        print("        ", rec.get("error"))
        print("        ", rec.get("trace")[-1] if rec.get("trace") else "")

with open("D:/training/caedecoder/stardecoding/batch_results.json", "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=1, default=str)
print("\nwrote batch_results.json:", len(results), "files")

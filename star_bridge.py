# -*- coding: utf-8 -*-
"""S 波 S1：官方 STAR-CCM+ 桥（探测 → 宏 → 工作副本 → 日志回流 → 差分）。

本机实测（2026-09-14）：Simcenter STAR-CCM+ 2502 (20.02.007) 带 license 可用，
`starccmw.exe -batch <macro.java> <copy.sim>` 在工作副本上跑通并回传日志（returncode=0）。
本模块把这套能力封装成可复用桥：
  ① find_star_exe/bridge_status：STARCCM_HOME → 常见安装根 → star/lib 布局 依次探测；
  ② 宏模板只用**已在官方 API 中用过/核对过**的调用（见 VERIFIED_APIS）：
     getActiveSimulation / getPresentationName / getPartManager().getObjects() /
     getRegionManager().getObjects() / getSceneManager().getObjects() /
     getContinuumManager().getObjects() / getReportManager().getObjects() /
     getPlotManager().getObjects() / getMonitorManager().getObjects() / saveState；
  ③ official_smoke / official_open_stats / official_resave：官方打开、视图统计、重存；
  ④ compare_official_view：官方视图计数 ↔ 本仓库解析计数 的逐项对照。

安全约束（沿用 F7）：**只在临时目录的工作副本上跑**，绝不改教程原件；无 exe 时全部返回
skipped 且不抛错（供 W6 门控复用）。
"""
import os
import re
import shutil
import sys
import tempfile

VERIFIED_APIS = (
    "getActiveSimulation()",
    "sim.getPresentationName()",
    "sim.getPartManager().getObjects()",
    "sim.getRegionManager().getObjects()",
    "sim.getSceneManager().getObjects()",
    "sim.getContinuumManager().getObjects()",
    "sim.getReportManager().getObjects()",
    "sim.getPlotManager().getObjects()",
    "sim.getMonitorManager().getObjects()",
    "sim.saveState(",
    "sim.println(",
)

ROOT_HINTS = (
    r"C:\Program Files\Siemens",
    r"C:\Program Files\CD-adapco",
    r"D:\Program Files\Siemens",
    r"D:\Siemens",
)


def candidate_exes(roots=None, max_depth=8):
    """候选 starccmw.exe 路径：STARCCM_HOME 优先，其次常见安装根（有界深度）。"""
    out = []
    home = os.environ.get("STARCCM_HOME")
    if home:
        for rel in (os.path.join("star", "bin", "starccmw.exe"),
                    os.path.join("star", "lib", "win64")):
            p = os.path.join(home, rel)
            if p.lower().endswith(".exe") and os.path.isfile(p):
                out.append(p)
            elif os.path.isdir(p):
                for dirpath, dirnames, filenames in os.walk(p):
                    if dirpath.count(os.sep) - p.count(os.sep) > 3:
                        dirnames[:] = []
                        continue
                    for f in filenames:
                        if f.lower() == "starccmw.exe":
                            out.append(os.path.join(dirpath, f))
    for root in (roots or ROOT_HINTS):
        if not os.path.isdir(root):
            continue
        base_depth = root.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, filenames in os.walk(root):
            if dirpath.count(os.sep) - base_depth > max_depth:
                dirnames[:] = []
                continue
            for f in filenames:
                if f.lower() == "starccmw.exe":
                    out.append(os.path.join(dirpath, f))
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def find_star_exe(roots=None):
    """找到可用的 starccmw.exe（找不到返回 None，不抛错）。"""
    cands = candidate_exes(roots)
    return cands[0] if cands else None


def bridge_status(roots=None):
    """桥可用性：{available, exe, candidates, reason}。"""
    cands = candidate_exes(roots)
    if not cands:
        return {"available": False, "exe": None, "candidates": [],
                "reason": "未找到 starccmw.exe（可设 STARCCM_HOME）"}
    return {"available": True, "exe": cands[0], "candidates": cands, "reason": ""}


# ------------------------------------------------------------------ 宏模板
SMOKE_MACRO = """
package macro;

import star.common.*;

public class StarBridgeSmoke extends StarMacro {
  public void execute() {
    Simulation sim = getActiveSimulation();
    sim.println("BRIDGE_SMOKE name=" + sim.getPresentationName());
    sim.println("BRIDGE_SMOKE regions=" + sim.getRegionManager().getObjects().size());
  }
}
"""

OPEN_STATS_MACRO = """
package macro;

import star.common.*;

public class StarBridgeStats extends StarMacro {
  private void dump(Simulation sim, String label, Iterable<?> objs) {
    int n = 0;
    for (Object o : objs) {
      sim.println("BRIDGE_OBJ " + label + "|" + String.valueOf(o));
      n++;
    }
    sim.println("BRIDGE_COUNT " + label + "=" + n);
  }

  public void execute() {
    Simulation sim = getActiveSimulation();
    sim.println("BRIDGE_STATS name=" + sim.getPresentationName());
    dump(sim, "Part", sim.getPartManager().getObjects());
    dump(sim, "Region", sim.getRegionManager().getObjects());
    dump(sim, "Scene", sim.getSceneManager().getObjects());
    dump(sim, "Continuum", sim.getContinuumManager().getObjects());
    dump(sim, "Report", sim.getReportManager().getObjects());
    dump(sim, "Plot", sim.getPlotManager().getObjects());
    dump(sim, "Monitor", sim.getMonitorManager().getObjects());
    sim.println("BRIDGE_STATS_END");
  }
}
"""

RESAVE_MACRO = """
package macro;

import star.common.*;

public class StarBridgeResave extends StarMacro {
  public void execute() {
    Simulation sim = getActiveSimulation();
    String out = "%(out)s";
    sim.println("BRIDGE_RESAVE input=" + sim.getPresentationName());
    sim.saveState(out);
    sim.println("BRIDGE_RESAVE_DONE " + out);
  }
}
"""


# ------------------------------------------------------------------ S6：受控官方运行（生成参考数据）
# 以下 API 均已在本机官方 Javadoc 中核对（star/common/Simulation.html、SimulationIterator.html、
# meshing/MeshPipelineController.html）：
#   sim.getSimulationIterator() / it.run() / it.runAndWait() / it.stop() / it.getCurrentIteration()
#   it.isIterating() / it.getNumberOfSteps() / sim.clearSolution() / mpc.generateVolumeMesh()
# 注意：sim.getSolver() 在本版本**不存在**（实测编译失败），不要再写。
RUN_CASE_MACRO = """
package macro;

import star.common.*;
import star.meshing.MeshPipelineController;

public class StarBridgeRunCase extends StarMacro {
  public void execute() {
    Simulation sim = getActiveSimulation();
    String out = "%(out)s";
    long target = %(target)d;
    sim.println("BRIDGE_RUN begin mesh=%(mesh)d target=" + target);
    if (%(mesh)d == 1) {
      try {
        MeshPipelineController mpc = sim.get(MeshPipelineController.class);
        mpc.generateVolumeMesh();
        sim.println("BRIDGE_RUN mesh done");
      } catch (Exception ex) {
        sim.println("BRIDGE_RUN mesh failed: " + ex);
      }
    }
    if (%(clear)d == 1) {
      try {
        sim.clearSolution();
        sim.println("BRIDGE_RUN solution cleared");
      } catch (Exception ex) {
        sim.println("BRIDGE_RUN clear failed: " + ex);
      }
    }
    SimulationIterator it = sim.getSimulationIterator();
    sim.println("BRIDGE_RUN iterations before=" + it.getCurrentIteration());
    it.run();
    while (it.isIterating()) {
      if (target > 0 && it.getCurrentIteration() >= target) {
        it.stop();
        sim.println("BRIDGE_RUN stopped at " + it.getCurrentIteration());
        break;
      }
      try { Thread.sleep(%(poll)d); } catch (Exception ex) { }
    }
    sim.println("BRIDGE_RUN iterations after=" + it.getCurrentIteration());
    sim.saveState(out);
    sim.println("BRIDGE_RUN_DONE " + out);
  }
}
"""


def official_run_case(src_sim, out_path, do_mesh=False, max_iterations=0,
                      clear_solution=False, class_name="StarBridgeRunCase",
                      timeout=1800, on_line=None, poll_ms=50):
    """在**工作副本**上跑官方求解（受控）：可选生成网格/清解，跑到目标迭代数即停，然后 Save As。

    max_iterations=0 → 交给算例自身停止准则（runAndWait 语义由宏内轮询实现）。
    返回 {ok, out_path, iterations_before/after, log, reason}。
    """
    out = os.path.abspath(out_path)
    out_java = out.replace("\\", "/")
    macro = RUN_CASE_MACRO % {"out": out_java, "mesh": 1 if do_mesh else 0,
                             "clear": 1 if clear_solution else 0,
                             "target": int(max_iterations or 0),
                             "poll": max(int(poll_ms), 10)}
    res = official_run(src_sim, macro, class_name, timeout=timeout, on_line=on_line)
    if res.get("skipped"):
        return res
    log = res.get("log") or ""
    def _num(pattern):
        m = re.search(pattern, log)
        return int(m.group(1)) if m else None
    res["out_path"] = out
    res["iterations_before"] = _num(r"BRIDGE_RUN iterations before=(\d+)")
    res["iterations_after"] = _num(r"BRIDGE_RUN iterations after=(\d+)")
    res["ok"] = bool(res.get("ok") and "BRIDGE_RUN_DONE" in log and os.path.isfile(out))
    if not res["ok"] and not res.get("reason"):
        res["reason"] = "运行标记缺失或输出未生成"
    return res


# ------------------------------------------------------------------ 运行器
def official_run(src_sim, macro_text, class_name="StarBridgeMacro", timeout=600,
                 on_line=None, workdir=None):
    """在**工作副本**上跑官方宏；返回 {ok, returncode, log, workdir, exe, reason}。"""
    status = bridge_status()
    if not status["available"]:
        return {"ok": False, "skipped": True, "reason": status["reason"], "log": ""}
    if not os.path.isfile(src_sim):
        return {"ok": False, "reason": "源文件不存在: %s" % src_sim, "log": ""}
    from star_macro import run_star_macro_stream
    tmp = workdir or tempfile.mkdtemp(prefix="star_bridge_")
    macro_path = os.path.join(tmp, "%s.java" % class_name)
    with open(macro_path, "w", encoding="utf-8") as f:
        f.write(macro_text)
    try:
        work, code, lines = run_star_macro_stream(status["exe"], src_sim, macro_path,
                                                  on_line, timeout)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": "官方宏执行失败: %s" % exc, "log": "",
                "exe": status["exe"]}
    log = "\n".join(lines)
    return {"ok": code == 0, "returncode": code, "log": log, "workdir": work,
            "exe": status["exe"], "reason": "" if code == 0 else "returncode=%s" % code}


def official_smoke(src_sim, timeout=600):
    """官方打开 + 冒烟（仿真名 / region 数）。"""
    res = official_run(src_sim, SMOKE_MACRO, "StarBridgeSmoke", timeout)
    if not res.get("ok") and res.get("skipped"):
        return res
    m = re.search(r"BRIDGE_SMOKE name=(.*)", res.get("log", ""))
    r = re.search(r"BRIDGE_SMOKE regions=(\d+)", res.get("log", ""))
    res["sim_name"] = m.group(1).strip() if m else None
    res["official_regions"] = int(r.group(1)) if r else None
    res["ok"] = bool(res.get("ok") and m and r)
    if not res["ok"] and not res.get("reason"):
        res["reason"] = "冒烟标记缺失（宏未按预期输出）"
    return res


def official_open_stats(src_sim, timeout=900):
    """官方视图统计：Part/Region/Scene/Continuum/Report/Plot/Monitor 的对象名清单。"""
    res = official_run(src_sim, OPEN_STATS_MACRO, "StarBridgeStats", timeout)
    if res.get("skipped"):
        return res
    counts, names, sim_name = {}, {}, None
    for line in res.get("log", "").splitlines():
        line = line.strip()
        m = re.match(r"BRIDGE_STATS name=(.*)", line)
        if m:
            sim_name = m.group(1).strip()
        m = re.match(r"BRIDGE_COUNT ([A-Za-z]+)=(\d+)", line)
        if m:
            counts[m.group(1)] = int(m.group(2))
        m = re.match(r"BRIDGE_OBJ ([A-Za-z]+)\|(.*)", line)
        if m:
            names.setdefault(m.group(1), []).append(m.group(2).strip())
    res["sim_name"] = sim_name
    res["counts"] = counts
    res["names"] = names
    res["ok"] = bool(res.get("ok") and counts and "BRIDGE_STATS_END" in res.get("log", ""))
    if not res["ok"] and not res.get("reason"):
        res["reason"] = "统计标记缺失"
    return res


def official_resave(src_sim, out_path=None, timeout=1200):
    """官方重存（Save As）到 out_path；返回 {ok, out_path, log}。"""
    out = out_path or os.path.join(os.path.dirname(os.path.abspath(src_sim)),
                                   "bridge_resaved.sim")
    out_java = out.replace("\\", "/")
    res = official_run(src_sim, RESAVE_MACRO % {"out": out_java}, "StarBridgeResave",
                       timeout)
    if res.get("skipped"):
        return res
    res["out_path"] = out
    res["ok"] = bool(res.get("ok") and "BRIDGE_RESAVE_DONE" in res.get("log", "")
                     and os.path.isfile(out))
    if not res["ok"] and not res.get("reason"):
        res["reason"] = "重存标记缺失或输出文件未生成"
    return res


# ------------------------------------------------------------------ 官方 ↔ 本地 对照
def _our_counts(sim):
    """本仓库解析结果的同口径计数（官方 Manager 视角）。"""
    def n(pred):
        return sum(1 for o in sim.objects if pred(o.class_name or ""))
    return {
        "Region": n(lambda c: c == "star.common.Region"),
        "Scene": n(lambda c: c.endswith(".Scene") or c.endswith("Scene") and "Manager" not in c),
        "Continuum": n(lambda c: c.endswith("Continuum") and "Manager" not in c),
        "Report": n(lambda c: c.endswith("Report") and "Manager" not in c
                    and "Monitor" not in c),
        "Plot": n(lambda c: c.endswith("Plot") and "Manager" not in c
                    and not c.endswith("MonitorPlot")),
        "Monitor": n(lambda c: c.endswith("Monitor")),
        # 注意：官方 getPartManager() 与对象图里的 *Part 类不完全同口径（后者含
        # CadPart/SimpleBlockPart 等几何部件）；此列标注为"口径近似"。
        "Part": n(lambda c: c.endswith("Part") and "Manager" not in c),
    }


def compare_official_view(src_sim_path, timeout=900):
    """官方视图 ↔ 本仓库解析：逐类计数对照（含 Δ 与口径说明）。"""
    from sim_parser import SimFile
    stats = official_open_stats(src_sim_path, timeout=timeout)
    if stats.get("skipped"):
        return {"ok": False, "skipped": True, "reason": stats.get("reason")}
    if not stats.get("ok"):
        return {"ok": False, "reason": stats.get("reason"), "log": stats.get("log", "")[-800:]}
    ours = _our_counts(SimFile(src_sim_path))
    rows = {}
    for label, oc in stats["counts"].items():
        mc = ours.get(label)
        rows[label] = {"official": oc, "ours": mc,
                       "delta": (None if mc is None else oc - mc),
                       "note": "" if mc is not None else "本仓库无同口径计数"}
    return {"ok": True, "sim_name": stats.get("sim_name"), "rows": rows,
            "official_names": stats.get("names"), "ours": ours, "reason": ""}


def _key_of(obj):
    """官方重存会重编号 id → 语义身份用「类名 + 名字」。"""
    return ((obj.class_name or ""), (str(obj.name) if obj.name is not None else None))


def _is_ref_value(v):
    """引用型取值：整数或整数列表（官方重存会重编号 id，不能直接跨文件比较）。"""
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return True
    if isinstance(v, (list, tuple)) and v:
        return all(isinstance(x, int) and not isinstance(x, bool) for x in v)
    return False


def compare_official_resave(src_path, resaved_path, keys=None, max_diffs=12,
                            min_match_rate=80.0):
    """官方重存件 ↔ 本仓库产物：**按类名+名字**配对差分（官方会重编号 id）。

    字段差异分两类：
      · value 类（字符串/浮点/bool/混合列表）→ 语义应一致，是判定依据；
      · ref 类（整数或整数列表 = 对象 id 引用）→ 官方重编号后必然不同，**不计入判定**，
        仅计数并在样例里展示（如需语义比较，须先按类名+名字建立 id 映射）。
    返回 {ok, matched, match_rate, value_diffs, ref_diffs, verdicts, ...}。
    """
    from sim_parser import SimFile
    a, b = SimFile(src_path), SimFile(resaved_path)
    keys = keys or ("PresentationName", "name", "Opacity", "DisplayerColor", "Mesh",
                    "ParallelScale", "Keys", "Parent", "Value", "Values")
    map_a, map_b = {}, {}
    for o in a.objects:
        map_a.setdefault(_key_of(o), []).append(o)
    for o in b.objects:
        map_b.setdefault(_key_of(o), []).append(o)
    common = sorted(set(map_a) & set(map_b), key=lambda k: (k[0] or "", k[1] or ""))
    only_src = sorted(set(map_a) - set(map_b))
    only_res = sorted(set(map_b) - set(map_a))
    value_diffs, ref_diffs, default_diffs = [], [], []
    for key in common:
        oa, ob = map_a[key][0], map_b[key][0]
        for k in keys:
            va, vb = oa.dict.get(k), ob.dict.get(k)
            if va == vb or (k not in oa.dict and k not in ob.dict):
                continue
            row = {"class": key[0], "name": key[1], "field": k,
                   "ours": str(va)[:70], "official": str(vb)[:70]}
            if _is_ref_value(va) or _is_ref_value(vb):
                # 任一侧是 id/列表 → 引用型（官方重编号，或不写出该引用）
                ref_diffs.append(row)
            elif va is None and vb is not None:
                # 官方加载时会补默认值（如 Opacity 1.0）—— 归一化，不计失败
                default_diffs.append(row)
            elif k == "Values" and vb is None and (key[0] or "").endswith("LookupTable"):
                # 预定义查找表：本仓库保留断点数组，官方按类名表达（保存时不写 Values）
                # —— 官方归一化，不计失败
                default_diffs.append(row)
            else:
                value_diffs.append(row)
            break
    match_rate = round(100.0 * len(common) / max(len(map_a), 1), 1)
    return {
        "ok": True,
        "n_src": len(a.objects), "n_resaved": len(b.objects),
        "keys_src": len(map_a), "keys_resaved": len(map_b),
        "matched": len(common), "match_rate": match_rate,
        "only_in_src": len(only_src), "only_in_resaved": len(only_res),
        "value_diffs": value_diffs[:max_diffs], "n_value_diffs": len(value_diffs),
        "ref_diffs": ref_diffs[:max_diffs], "n_ref_diffs": len(ref_diffs),
        "default_diffs": default_diffs[:max_diffs], "n_default_diffs": len(default_diffs),
        "name_field_diffs": sum(1 for d in value_diffs
                               if d["field"] in ("PresentationName", "name")),
        "only_src_sample": [list(x) for x in only_src[:8]],
        "only_resaved_sample": [list(x) for x in only_res[:8]],
        "min_match_rate": min_match_rate,
        "verdicts": {
            "structure_survived": match_rate >= min_match_rate,
            "value_fields_stable": len(value_diffs) == 0,
            "normalized_diffs": len(default_diffs),
            "ref_diffs": len(ref_diffs),
        },
    }

def _main(argv=None):
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    import argparse
    ap = argparse.ArgumentParser(description="S1 官方 STAR-CCM+ 桥（探测/冒烟/视图对照/重存）")
    ap.add_argument("--status", action="store_true", help="打印桥可用性")
    ap.add_argument("--smoke", metavar="SIM", help="官方打开冒烟")
    ap.add_argument("--view", metavar="SIM", help="官方视图 ↔ 本仓库计数对照")
    ap.add_argument("--resave", metavar="SIM", help="官方重存（Save As）")
    ap.add_argument("--out", metavar="PATH", help="--resave 输出路径")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args(argv)
    if args.status or not any([args.smoke, args.view, args.resave]):
        st = bridge_status()
        print("可用:", st["available"])
        print("exe:", st["exe"])
        for c in st["candidates"][:5]:
            print("   候选:", c)
        if not st["available"]:
            print("原因:", st["reason"])
        return 0 if st["available"] else 1
    if args.smoke:
        r = official_smoke(args.smoke, args.timeout)
        print("ok=%s name=%s regions=%s reason=%s"
              % (r.get("ok"), r.get("sim_name"), r.get("official_regions"),
                 r.get("reason")))
        return 0 if r.get("ok") else 1
    if args.view:
        r = compare_official_view(args.view, args.timeout)
        if not r.get("ok"):
            print("对照未完成:", r.get("reason"))
            return 1
        print("仿真:", r["sim_name"])
        for label, row in r["rows"].items():
            print("   %-10s 官方 %-5s 本仓库 %-5s Δ=%s %s"
                  % (label, row["official"], row["ours"], row["delta"], row["note"]))
        return 0
    if args.resave:
        r = official_resave(args.resave, args.out, args.timeout)
        print("ok=%s out=%s reason=%s" % (r.get("ok"), r.get("out_path"), r.get("reason")))
        return 0 if r.get("ok") else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


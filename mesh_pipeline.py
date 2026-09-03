# -*- coding: utf-8 -*-
"""N 波 N1：网格流水线执行引擎（操作图、进度、预览、取消）。

- 操作图：`MeshPipeline` 按序持有若干 `MeshStage`（run/preview 回调），阶段间共享
  `ctx` 上下文；每个阶段产出的网格快照通过 `preview` 取出，供 3D 预览。
- 进度：`run(..., progress=i,stage,total)` 每阶段回调，支持分阶段权重；
- 取消：`run(..., canceled=lambda: bool_flag)` 每阶段检查，置位即抛 `PipelineCanceled`；
- 空流水线：`MeshPipeline()` 无阶段也能 `run` 成功（验收「自动网格节点右键可跑通
  空流水线」基座）。
- 默认流水线 `default_volume_mesh_pipeline`：表面细分 → 体网格 → 质量诊断 → 单元
  重编号，四条真实操作串成图。

纯 Python / numpy，两环境（默认 3.14 / conda occ）皆可用。
"""
import numpy as np


class PipelineCanceled(Exception):
    """流水线被取消。"""


class MeshStage:
    def __init__(self, name, run=None, preview=None):
        self.name = name
        self.run = run            # run(ctx) -> result（写 ctx）
        self.preview = preview    # preview(ctx) -> 网格快照（供显示）

    def __repr__(self):
        return "<MeshStage %s>" % self.name


class MeshPipeline:
    """网格流水线执行引擎。"""

    def __init__(self, name="mesh_pipeline"):
        self.name = name
        self.stages = []

    @property
    def is_empty(self):
        return len(self.stages) == 0

    def add(self, name, run=None, preview=None, weight=1.0):
        self.stages.append((MeshStage(name, run, preview), float(weight)))
        return self

    def run(self, ctx=None, progress=None, canceled=None):
        """执行整条流水线。

        - ctx：跨阶段共享上下文（缺省新建 dict）。
        - progress(i, stage_name, total)：每阶段开始前回调（i 从 1 起）。
        - canceled：无参回调，返回真则抛 PipelineCanceled。
        返回 {name, ok, cancelled, stages_ran, results, previews, ctx}。
        """
        ctx = dict(ctx or {})
        if canceled is None:
            canceled = lambda: False
        results = []
        previews = {}
        total = len(self.stages)
        for idx, (stage, weight) in enumerate(self.stages, start=1):
            if canceled():
                return {"name": self.name, "ok": False, "cancelled": True,
                        "stages_ran": idx - 1, "results": results,
                        "previews": previews, "ctx": ctx}
            if progress:
                progress(idx, stage.name, total)
            if stage.run is None:
                res = None
            else:
                res = stage.run(ctx)
            results.append({"stage": stage.name, "result": res})
            if stage.preview is not None:
                try:
                    previews[stage.name] = stage.preview(ctx)
                except Exception:
                    previews[stage.name] = None
        return {"name": self.name, "ok": True, "cancelled": False,
                "stages_ran": len(self.stages), "results": results,
                "previews": previews, "ctx": ctx}


# ---------------------------------------------------------------- 表面细分
def refine_surface(vertices, faces, levels=1):
    """一致性边中点细分：每级把每条共享边只拆一次 → 每三角 4 个，水密保持。

    返回 (V, F)。用于「生成表面网格」阶段（曲率网格粗基上加密）。纯 numpy。
    """
    V = [list(v) for v in vertices]
    F = [tuple(map(int, t)) for t in faces]
    for _ in range(max(levels, 1)):
        mid = {}
        new_F = []
        for tri in F:
            sub = []
            for ia, ib in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                key = (ia, ib) if ia < ib else (ib, ia)
                if key not in mid:
                    pa = V[ia]
                    pb = V[ib]
                    mid[key] = len(V)
                    V.append([0.5 * (pa[0] + pb[0]),
                              0.5 * (pa[1] + pb[1]),
                              0.5 * (pa[2] + pb[2])])
                sub.append(mid[key])
            a, b, c = tri
            mab, mbc, mca = sub
            new_F.append((a, mab, mca))
            new_F.append((b, mbc, mab))
            new_F.append((c, mca, mbc))
            new_F.append((mab, mbc, mca))
        F = new_F
    return V, F


# ---------------------------------------------------------------- 默认流水线
def default_volume_mesh_pipeline(vertices, faces, cell_size, spacing=None,
                                 refine_levels=1):
    """默认「自动网格」流水线：表面细分→体网格→质量→重编号。"""
    V0 = np.asarray(vertices, float)
    F0 = np.asarray(faces, np.int64)
    spacing = spacing if spacing is not None else float(cell_size) * 2.0

    from mesh_tet import tet_mesh, tet_quality

    pipe = MeshPipeline("自动网格")

    def st_surface(ctx):
        vs, fs = refine_surface(V0, F0, refine_levels)
        ctx["surface"] = (vs, fs)
        return {"n_vertices": len(vs), "n_faces": len(fs)}

    def st_volume(ctx):
        vs, fs = ctx["surface"]
        out = tet_mesh(vs, fs, spacing=spacing)
        ctx["volume"] = (out["vertices"], out["cells"])
        ctx["volume_meta"] = out
        return {"n_cells": out["n_cells"], "method": out["method"]}

    def st_quality(ctx):
        surf = ctx["surface"]
        vol = ctx["volume"]
        q = {"surface": None, "volume": tet_quality(vol[0], vol[1])}
        from occ_repair import quality_metrics
        q["surface"] = quality_metrics(surf[0], surf[1])
        ctx["quality"] = q
        return q

    def st_renumber(ctx):
        V, C = ctx["volume"]
        arr = np.asarray(C, np.int64)
        order = np.unique(arr.ravel())
        remap = {int(idx): i for i, idx in enumerate(order)}
        Cn = np.array([[remap[int(x)] for x in row] for row in arr], np.int64)
        Vn = np.asarray(V, float)[order]
        ctx["volume"] = (Vn, Cn)
        return {"n_nodes": len(Vn), "n_cells": Cn.shape[0],
                "compact": int(np.max(Cn)) + 1 == Cn.shape[0]}

    pipe.add("生成表面网格", st_surface,
             preview=lambda ctx: ctx.get("surface"))
    pipe.add("生成体网格", st_volume,
             preview=lambda ctx: ctx.get("volume"))
    pipe.add("质量诊断", st_quality,
             preview=lambda ctx: ctx.get("volume"))
    pipe.add("单元重编号", st_renumber,
             preview=lambda ctx: ctx.get("volume"))
    return pipe
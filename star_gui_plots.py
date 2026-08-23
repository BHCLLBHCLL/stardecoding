# -*- coding: utf-8 -*-
"""绘图 / 报告查看：从对象图抽出监视器与报告的标量，不采样求解器。"""

from PyQt5.QtWidgets import QLabel, QPlainTextEdit, QVBoxLayout, QWidget


def _numeric_list(value):
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value):
        return None
    return [float(x) for x in value]


def sparkline(values, width=40):
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo if hi > lo else 1.0
    glyphs = " .:-=+*#"
    step = max(1, len(values) // width)
    sampled = values[::step][:width]
    out = []
    for x in sampled:
        idx = int((x - lo) / span * (len(glyphs) - 1))
        out.append(glyphs[max(0, min(len(glyphs) - 1, idx))])
    return "".join(out)


def collect_plot_series(sim):
    """从对象图抽出长度>=4 的数值序列（残差历史等）；无则空。"""
    series = []
    if sim is None:
        return series
    skip = {"Keys", "vector", "Dimensions"}
    for o in sim.objects:
        cn = o.class_name or ""
        if "Manager" in cn:
            continue
        if not any(tag in cn for tag in ("Monitor", "Report", "Plot", "Residual")):
            continue
        for k, v in (o.dict or {}).items():
            if k in skip:
                continue
            nums = _numeric_list(v)
            if nums:
                series.append((o.name or cn.split(".")[-1], k, nums, o.id))
    return series


def collect_plot_rows(sim):
    """[(kind, name, value, id)]。"""
    rows = []
    if sim is None:
        return rows
    for o in sim.objects:
        cn = o.class_name or ""
        short = cn.split(".")[-1]
        if "Manager" in short:
            continue
        if any(tag in short for tag in ("Monitor", "Report", "Plot")):
            val = None
            for k in ("Value", "LastValue", "CurrentValue", "Sample"):
                if k in (o.dict or {}):
                    val = o.dict.get(k)
                    break
            rows.append((short, o.name or short, val, o.id))
    return rows


class PlotPane(QWidget):
    """简单文本表：监视器 / 报告 / 绘图对象。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.hint = QLabel("绘图 / 报告（只读；无解场时仅显示对象属性）")
        lay.addWidget(self.hint)
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        font = self.view.font()
        font.setFamily("Consolas")
        font.setPointSize(9)
        self.view.setFont(font)
        lay.addWidget(self.view, 1)

    def show_sim(self, sim):
        rows = collect_plot_rows(sim)
        if not rows:
            self.view.setPlainText("（当前项目没有监视器/报告数值）")
            return
        lines = ["%-22s  %-32s  %s" % ("类型", "名称", "值")]
        lines.append("-" * 72)
        for kind, name, val, oid in rows:
            lines.append("id %-6s  %-20s  %-28s  %s" % (oid, kind[:20], (name or "")[:28], val))
        series = collect_plot_series(sim)
        lines.append("")
        if series:
            lines.append("曲线（对象图序列，非求解器采样）")
            for name, key, nums, oid in series:
                lines.append("id %s  %s.%s  n=%d  %s" % (
                    oid, name, key, len(nums), sparkline(nums)))
        else:
            lines.append("无监视器采样序列（未求解或数组未解码；不做求解器采样）")
        self.view.setPlainText("\n".join(lines))

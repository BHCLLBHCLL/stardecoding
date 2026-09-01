# -*- coding: utf-8 -*-
"""绘图 / 报告查看：G6 起接入监视器真实采样曲线（MonitorManager → 双 MasterArray）。"""

from PyQt5.QtGui import QColor, QPainter, QPen
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


def collect_array_series(sim):
    """1D Float8/Float4（长度不能被 3 整除，排除顶点表）。"""
    series = []
    if sim is None:
        return series
    for i, a in enumerate(sim.arrays or []):
        if a.get("type") not in ("Float8", "Float4"):
            continue
        n = a.get("count") or 0
        if n < 4 or n % 3 == 0:
            continue
        try:
            data = sim.array_data(i)
        except Exception:
            continue
        nums = [float(x) for x in list(data.reshape(-1)[:512])]
        if len(nums) >= 4:
            series.append(("Array[%d]" % i, a.get("type") or "values", nums, i))
    return series


def monitor_curve_items(sim, mc=None, max_pts=512):
    """G6：真实监视器曲线 → [(名称, "G6", y, id, x|None)]（降采样供 SeriesCanvas）。

    数据来自 sim.extract_monitor_curves()（MonitorManager → XAxisData /
    YAxisValues 双 MasterArray，G3 存储体系解包），与旧 collect_plot_series
    （对象 dict 内联序列，通常为空）不同：这里 X 是真实采样索引
    （迭代号或物理时间），元组第 5 元素供 SeriesCanvas 按 X 定位。
    """
    items = []
    if sim is None:
        return items
    if mc is None:
        if not hasattr(sim, "extract_monitor_curves"):
            return items
        try:
            mc = sim.extract_monitor_curves()
        except Exception:
            return items
    if not mc or not mc.get("ok"):
        return items
    for e in mc["monitors"]:
        d = mc["data"].get(e["name"])
        if not d or d.get("y") is None or d["y"].size < 2:
            continue
        y, x = d["y"], d.get("index")
        step = max(1, y.size // max_pts)
        idxs = list(range(0, y.size, step))
        if idxs[-1] != y.size - 1:
            idxs.append(y.size - 1)
        ys = [float(y[i]) for i in idxs]
        xs = ([float(x[i]) for i in idxs]
              if x is not None and x.size == y.size else None)
        items.append((e["name"], "G6", ys, e.get("id", 0), xs))
    return items


def monitor_report_lines(sim, mc=None):
    """G6：监视器曲线表 + 绘图关联 → 文本行（PlotPane 只读表格）。

    结构与 CLI --curves 输出一致：每监视器一行（n/范围/末值/==CurrentValue），
    绘图段标注标题/轴标题/单位/图例/tabular 表文件。
    无监视器数据（未求解）返回 []（回退旧路径）。
    """
    if sim is None or not hasattr(sim, "extract_monitor_curves"):
        return []
    try:
        if mc is None:
            mc = sim.extract_monitor_curves()
    except Exception:
        return []
    if not mc or not mc.get("ok"):
        return []
    lines = ["监视器曲线（G6：MonitorManager → XAxisData/YAxisValues 双 MasterArray）",
             "-" * 72]
    for e in mc["monitors"]:
        cv = e.get("cur_value")
        eq = ""
        if isinstance(cv, float) and (
                cv == e["last"] or
                abs(cv - e["last"]) <= 1e-9 * max(1.0, abs(cv))):
            eq = "  ==CurrentValue"
        lines.append("%-32s %-24s n=%-7d y[%+.4g .. %+.4g] last=%+.4g%s" % (
            e["name"], (e["class"] or "").split(".")[-1], e["n"],
            e["y_min"], e["y_max"], e["last"], eq))
    try:
        pl = sim.extract_plots(mc)
    except Exception:
        pl = None
    if pl and pl.get("ok"):
        lines.append("")
        lines.append("绘图关联（G2 标注：标题/轴标题/单位/图例）：")
        for q in pl["plots"]:
            lines.append("◇ %s (%s) X轴=%r 单位=%r" % (
                q["title"], (q["class"] or "").split(".")[-1],
                q["x_title"], q["x_units"]))
            for s in q["series"]:
                if s["kind"] == "monitor":
                    lines.append("   - %-30s X←%s Y←%s" % (
                        s["name"], s.get("x_monitor") or "-",
                        s.get("y_monitor") or "-"))
                elif s["kind"] == "tabular":
                    lines.append("   - %-30s [tabular] X列=%r Y列=%r%s" % (
                        s["name"], s.get("x_column"), s.get("y_column"),
                        (" 表文件=%r" % s.get("table_file"))
                        if s.get("table_file") else ""))
    return lines


class SeriesCanvas(QWidget):
    """QPainter 折线：有序列才画。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._series = []
        self.setMinimumHeight(90)
        self.setAutoFillBackground(True)

    def set_series(self, series):
        self._series = list(series or [])
        self.setVisible(bool(self._series))
        self.update()

    def paintEvent(self, ev):
        super(SeriesCanvas, self).paintEvent(ev)
        if not self._series:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor(250, 250, 250))
        w, h = max(1, self.width()), max(1, self.height())
        pad = 8
        colors = [QColor(40, 90, 160), QColor(180, 80, 40), QColor(40, 140, 80)]
        for si, item in enumerate(self._series[:3]):
            nums = item[2] if len(item) > 2 else []
            if len(nums) < 2:
                continue
            xs = item[4] if len(item) > 4 else None
            if xs is None or len(xs) != len(nums):
                xs = None
            if xs is not None:
                lo_x, hi_x = min(xs), max(xs)
                span_x = (hi_x - lo_x) or 1.0
            lo, hi = min(nums), max(nums)
            span = hi - lo if hi > lo else 1.0
            pen = QPen(colors[si % len(colors)])
            pen.setWidth(2)
            p.setPen(pen)
            last = None
            n = len(nums)
            for i, y in enumerate(nums):
                if xs is not None:
                    x = pad + (w - 2 * pad) * (xs[i] - lo_x) / span_x
                else:
                    x = pad + (w - 2 * pad) * i / float(max(1, n - 1))
                yy = h - pad - (h - 2 * pad) * ((y - lo) / span)
                pt = (x, yy)
                if last is not None:
                    p.drawLine(int(last[0]), int(last[1]), int(pt[0]), int(pt[1]))
                last = pt
        p.end()


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
        self.canvas = SeriesCanvas()
        self.canvas.hide()
        lay.addWidget(self.canvas)

    def show_sim(self, sim):
        mc = None
        if sim is not None and hasattr(sim, "extract_monitor_curves"):
            try:
                mc = sim.extract_monitor_curves()
            except Exception:
                mc = None
        if mc and mc.get("ok"):
            self.hint.setText("绘图 / 报告（G6：监视器真实采样曲线，X=迭代号/物理时间）")
            self.view.setPlainText(
                "\n".join(monitor_report_lines(sim, mc)))
            if getattr(self, "canvas", None) is not None:
                self.canvas.set_series(monitor_curve_items(sim, mc)[:3])
            return
        rows = collect_plot_rows(sim)
        series = collect_plot_series(sim)
        array_series = collect_array_series(sim)
        if not rows and not series and not array_series:
            self.view.setPlainText("（当前项目没有监视器/报告数值）")
            if getattr(self, "canvas", None) is not None:
                self.canvas.set_series([])
            return
        lines = ["%-22s  %-32s  %s" % ("类型", "名称", "值")]
        lines.append("-" * 72)
        for kind, name, val, oid in rows:
            lines.append("id %-6s  %-20s  %-28s  %s" % (oid, kind[:20], (name or "")[:28], val))
        lines.append("")
        if series:
            lines.append("曲线（对象图序列，非求解器采样）")
            for name, key, nums, oid in series:
                lines.append("id %s  %s.%s  n=%d  %s" % (
                    oid, name, key, len(nums), sparkline(nums)))
        elif array_series:
            lines.append("曲线（1D 数组，非求解器采样）")
            for name, key, nums, oid in array_series:
                lines.append("%s  %s  n=%d  %s" % (
                    name, key, len(nums), sparkline(nums)))
        else:
            lines.append("无监视器采样序列（未求解或数组未解码；不做求解器采样）")
        self.view.setPlainText("\n".join(lines))
        draw = series or array_series
        if getattr(self, "canvas", None) is not None:
            self.canvas.set_series(draw[:3])

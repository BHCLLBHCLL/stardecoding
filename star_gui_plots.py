# -*- coding: utf-8 -*-
"""绘图 / 报告查看：从对象图抽出监视器与报告的标量，不采样求解器。"""

from PyQt5.QtWidgets import QLabel, QPlainTextEdit, QVBoxLayout, QWidget


def collect_plot_rows(sim):
    """[(kind, name, value_text)]。"""
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
        self.view.setPlainText("\n".join(lines))

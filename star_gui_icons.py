# -*- coding: utf-8 -*-
"""star_gui_icons.py — STAR-CCM+ 语义矢量图标（对齐 cab_icons 的 QPainter 路线）。

不再用 Qt 通用 SP_* / 字母色块：每个 key 画成与 STAR-CCM+ 功能对应的图形
（等轴测立方体、RGB 轴色、网格三角、求解播放、场景相机、绘图曲线等）。
"""

import math

from PyQt5.QtCore import QPointF, QRectF, QSize, Qt
from PyQt5.QtGui import (
    QBrush, QColor, QFont, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF,
)


def _c(hex_or_rgb, a=255):
    if isinstance(hex_or_rgb, QColor):
        col = QColor(hex_or_rgb)
    elif isinstance(hex_or_rgb, (tuple, list)):
        col = QColor(*hex_or_rgb)
    else:
        col = QColor(hex_or_rgb)
    col.setAlpha(a)
    return col


class AppIcons:
    """get(key) -> QIcon。按 STAR-CCM+ 功能语义绘制。"""

    def __init__(self, style=None):
        self._style = style
        self._cache = {}

    def get(self, key, fallback="unknown"):
        if key in self._cache:
            return self._cache[key]
        name = key if self._has(key) else fallback
        icon = QIcon(self._paint(name, 24))
        self._cache[key] = icon
        return icon

    def sized(self, key, w=16, h=16, fallback="unknown"):
        return self.get(key, fallback).pixmap(QSize(w, h))

    def _has(self, key):
        return hasattr(self, "_draw_" + self._method(key))

    def _method(self, key):
        return (key.replace("+", "p").replace("-", "n").replace(">", "_")
                .replace(".", "_"))

    def _paint(self, name, size):
        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        m = max(1, size // 12)
        r = QRectF(m, m, size - 2 * m, size - 2 * m)
        fn = getattr(self, "_draw_" + self._method(name), self._draw_unknown)
        fn(p, r, size)
        p.end()
        return pm

    # ----- primitives -----
    @staticmethod
    def _pen(color, w=1.4):
        pen = QPen(_c(color))
        pen.setWidthF(w)
        pen.setJoinStyle(Qt.RoundJoin)
        pen.setCapStyle(Qt.RoundCap)
        return pen

    def _iso_pts(self, r):
        """等轴测立方体可见顶点：A顶, B右上, C中, D左上, E左下, F底, G右下。"""
        l, t, w, h = r.left(), r.top(), r.width(), r.height()
        a = QPointF(l + w * 0.50, t + h * 0.06)
        b = QPointF(l + w * 0.94, t + h * 0.28)
        c = QPointF(l + w * 0.50, t + h * 0.48)
        d = QPointF(l + w * 0.06, t + h * 0.28)
        e = QPointF(l + w * 0.06, t + h * 0.68)
        f = QPointF(l + w * 0.50, t + h * 0.92)
        g = QPointF(l + w * 0.94, t + h * 0.68)
        top = QPolygonF([a, b, c, d])
        right = QPolygonF([c, b, g, f])
        left = QPolygonF([d, c, f, e])
        return top, left, right, (a, b, c, d, e, f, g)

    def _draw_cube(self, p, r, top_c, left_c, right_c, edge="#334155", fill=True,
                   edge_w=1.15, alpha=255):
        top, left, right, _ = self._iso_pts(r)
        p.setPen(self._pen(edge, edge_w))
        if fill:
            p.setBrush(QBrush(_c(top_c, alpha)))
            p.drawPolygon(top)
            p.setBrush(QBrush(_c(left_c, alpha)))
            p.drawPolygon(left)
            p.setBrush(QBrush(_c(right_c, alpha)))
            p.drawPolygon(right)
        else:
            p.setBrush(Qt.NoBrush)
            p.drawPolygon(top)
            p.drawPolygon(left)
            p.drawPolygon(right)

    # ----- file / edit -----
    def _draw_open(self, p, r, _s):
        p.setPen(self._pen("#b45309", 1.2))
        p.setBrush(QBrush(_c("#fbbf24")))
        tab = QRectF(r.left(), r.top(), r.width() * 0.42, r.height() * 0.26)
        p.drawRoundedRect(tab, 1.5, 1.5)
        p.setBrush(QBrush(_c("#f59e0b")))
        body = QRectF(r.left(), r.top() + r.height() * 0.20,
                      r.width(), r.height() * 0.74)
        p.drawRoundedRect(body, 2, 2)

    def _draw_save(self, p, r, _s):
        p.setPen(self._pen("#1e3a5f", 1.2))
        p.setBrush(QBrush(_c("#3b82f6")))
        p.drawRoundedRect(r, 2, 2)
        p.setBrush(QBrush(_c("#e0f2fe")))
        p.drawRect(QRectF(r.left() + r.width() * 0.22, r.top(),
                          r.width() * 0.56, r.height() * 0.34))
        p.setBrush(QBrush(_c("#93c5fd")))
        p.drawRoundedRect(QRectF(r.left() + r.width() * 0.16,
                                 r.top() + r.height() * 0.46,
                                 r.width() * 0.68, r.height() * 0.42), 1.5, 1.5)
        p.setBrush(QBrush(_c("#1e3a5f")))
        p.setPen(Qt.NoPen)
        p.drawRect(QRectF(r.left() + r.width() * 0.38, r.top() + r.height() * 0.08,
                          r.width() * 0.18, r.height() * 0.18))

    def _draw_file(self, p, r, _s):
        p.setPen(self._pen("#64748b", 1.2))
        p.setBrush(QBrush(_c("#f8fafc")))
        path = QPainterPath()
        fold = r.width() * 0.32
        path.moveTo(r.left() + 1, r.top() + 1)
        path.lineTo(r.right() - fold, r.top() + 1)
        path.lineTo(r.right() - 1, r.top() + fold)
        path.lineTo(r.right() - 1, r.bottom() - 1)
        path.lineTo(r.left() + 1, r.bottom() - 1)
        path.closeSubpath()
        p.drawPath(path)
        p.setPen(self._pen("#94a3b8", 1.0))
        p.drawLine(QPointF(r.right() - fold, r.top() + 1),
                   QPointF(r.right() - fold, r.top() + fold))
        p.drawLine(QPointF(r.right() - fold, r.top() + fold),
                   QPointF(r.right() - 1, r.top() + fold))

    def _draw_close(self, p, r, _s):
        self._draw_file(p, r, _s)
        p.setPen(self._pen("#dc2626", 2.0))
        m = r.width() * 0.22
        p.drawLine(QPointF(r.left() + m, r.top() + r.height() * 0.42),
                   QPointF(r.right() - m, r.bottom() - r.height() * 0.18))
        p.drawLine(QPointF(r.right() - m, r.top() + r.height() * 0.42),
                   QPointF(r.left() + m, r.bottom() - r.height() * 0.18))

    def _draw_new(self, p, r, _s):
        self._draw_file(p, r, _s)
        p.setPen(self._pen("#16a34a", 1.8))
        cx, cy = r.center().x() + r.width() * 0.08, r.center().y() + r.height() * 0.08
        s = r.width() * 0.22
        p.drawLine(QPointF(cx - s, cy), QPointF(cx + s, cy))
        p.drawLine(QPointF(cx, cy - s), QPointF(cx, cy + s))

    def _draw_folder(self, p, r, _s):
        self._draw_open(p, r, _s)

    def _draw_undo(self, p, r, _s):
        p.setPen(self._pen("#0369a1", 2.0))
        p.setBrush(Qt.NoBrush)
        p.drawArc(r.adjusted(r.width() * 0.12, r.height() * 0.18,
                             -r.width() * 0.12, -r.height() * 0.08).toRect(),
                  40 * 16, 230 * 16)
        tip = QPolygonF([
            QPointF(r.left() + r.width() * 0.08, r.center().y() - r.height() * 0.02),
            QPointF(r.left() + r.width() * 0.38, r.center().y() - r.height() * 0.28),
            QPointF(r.left() + r.width() * 0.34, r.center().y() + r.height() * 0.12),
        ])
        p.setBrush(QBrush(_c("#0369a1")))
        p.setPen(Qt.NoPen)
        p.drawPolygon(tip)

    # ----- solve -----
    def _draw_play(self, p, r, _s):
        p.setPen(self._pen("#15803d", 1.1))
        p.setBrush(QBrush(_c("#22c55e")))
        poly = QPolygonF([
            QPointF(r.left() + r.width() * 0.22, r.top() + r.height() * 0.12),
            QPointF(r.right() - r.width() * 0.08, r.center().y()),
            QPointF(r.left() + r.width() * 0.22, r.bottom() - r.height() * 0.12),
        ])
        p.drawPolygon(poly)

    def _draw_pause(self, p, r, _s):
        p.setPen(self._pen("#a16207", 1.1))
        p.setBrush(QBrush(_c("#eab308")))
        w = r.width() * 0.22
        gap = r.width() * 0.14
        x1 = r.left() + r.width() * 0.18
        p.drawRoundedRect(QRectF(x1, r.top() + r.height() * 0.12, w, r.height() * 0.76), 1.5, 1.5)
        p.drawRoundedRect(QRectF(x1 + w + gap, r.top() + r.height() * 0.12, w, r.height() * 0.76), 1.5, 1.5)

    def _draw_step(self, p, r, _s):
        p.setPen(self._pen("#15803d", 1.1))
        p.setBrush(QBrush(_c("#22c55e")))
        poly = QPolygonF([
            QPointF(r.left() + r.width() * 0.10, r.top() + r.height() * 0.16),
            QPointF(r.left() + r.width() * 0.62, r.center().y()),
            QPointF(r.left() + r.width() * 0.10, r.bottom() - r.height() * 0.16),
        ])
        p.drawPolygon(poly)
        p.drawRoundedRect(QRectF(r.left() + r.width() * 0.70, r.top() + r.height() * 0.16,
                                 r.width() * 0.18, r.height() * 0.68), 1, 1)

    def _draw_stop(self, p, r, _s):
        p.setPen(self._pen("#991b1b", 1.1))
        p.setBrush(QBrush(_c("#ef4444")))
        m = r.width() * 0.16
        p.drawRoundedRect(r.adjusted(m, m, -m, -m), 2, 2)

    def _draw_server(self, p, r, _s):
        p.setPen(self._pen("#1e40af", 1.2))
        p.setBrush(QBrush(_c("#93c5fd")))
        p.drawRoundedRect(QRectF(r.left(), r.top() + r.height() * 0.08,
                                 r.width() * 0.55, r.height() * 0.38), 2, 2)
        p.drawRoundedRect(QRectF(r.left() + r.width() * 0.40, r.top() + r.height() * 0.52,
                                 r.width() * 0.55, r.height() * 0.38), 2, 2)
        p.setPen(self._pen("#2563eb", 1.4))
        p.drawLine(QPointF(r.left() + r.width() * 0.42, r.top() + r.height() * 0.46),
                   QPointF(r.left() + r.width() * 0.58, r.top() + r.height() * 0.52))

    # ----- view / display (STAR-CCM+ cube language) -----
    def _draw_solid(self, p, r, _s):
        self._draw_cube(p, r, "#7dd3d8", "#0e7490", "#155e75")

    def _draw_wire(self, p, r, _s):
        self._draw_cube(p, r, "#ffffff", "#ffffff", "#ffffff", edge="#0f766e",
                        fill=False, edge_w=1.35)

    def _draw_edges(self, p, r, _s):
        self._draw_cube(p, r, "#99e4e8", "#148ba6", "#0f6f86", edge="#111827", edge_w=1.45)

    def _draw_transp(self, p, r, _s):
        self._draw_cube(p, r, "#a5f3fc", "#22d3ee", "#0891b2", edge="#0e7490",
                        alpha=110, edge_w=1.2)

    def _draw_view_px(self, p, r, _s):
        self._draw_cube(p, r, "#fecaca", "#fca5a5", "#dc2626")

    def _draw_view_nx(self, p, r, _s):
        self._draw_cube(p, r, "#fecaca", "#b91c1c", "#f87171")

    def _draw_view_py(self, p, r, _s):
        self._draw_cube(p, r, "#bbf7d0", "#16a34a", "#86efac")

    def _draw_view_ny(self, p, r, _s):
        self._draw_cube(p, r, "#bbf7d0", "#86efac", "#15803d")

    def _draw_view_pz(self, p, r, _s):
        self._draw_cube(p, r, "#2563eb", "#93c5fd", "#60a5fa")

    def _draw_view_nz(self, p, r, _s):
        self._draw_cube(p, r, "#1e3a8a", "#93c5fd", "#60a5fa")

    def _draw_view_iso(self, p, r, _s):
        self._draw_cube(p, r, "#93c5fd", "#86efac", "#fca5a5")

    def _draw_fit(self, p, r, _s):
        p.setPen(self._pen("#334155", 1.6))
        p.setBrush(Qt.NoBrush)
        s = r.width() * 0.26
        corners = [
            (r.left(), r.top(), 1, 1),
            (r.right(), r.top(), -1, 1),
            (r.left(), r.bottom(), 1, -1),
            (r.right(), r.bottom(), -1, -1),
        ]
        for x, y, sx, sy in corners:
            p.drawLine(QPointF(x, y), QPointF(x + sx * s, y))
            p.drawLine(QPointF(x, y), QPointF(x, y + sy * s))
        inner = r.adjusted(r.width() * 0.28, r.height() * 0.28,
                           -r.width() * 0.28, -r.height() * 0.28)
        self._draw_cube(p, inner, "#7dd3d8", "#0e7490", "#155e75", edge_w=0.9)

    def _draw_reset(self, p, r, _s):
        self._draw_cube(p, r.adjusted(r.width() * 0.08, r.height() * 0.18,
                                      -r.width() * 0.22, -r.height() * 0.02),
                        "#7dd3d8", "#0e7490", "#155e75", edge_w=1.0)
        p.setPen(self._pen("#0369a1", 1.7))
        p.setBrush(Qt.NoBrush)
        arc = QRectF(r.left() + r.width() * 0.35, r.top(),
                     r.width() * 0.62, r.height() * 0.62)
        p.drawArc(arc.toRect(), 20 * 16, 230 * 16)

    def _draw_zoom_in(self, p, r, _s):
        self._draw_fit(p, r, _s)

    def _draw_zoom_out(self, p, r, _s):
        self._draw_fit(p, r, _s)

    # ----- tree / CAE objects -----
    def _draw_layer_geometry(self, p, r, _s):
        self._draw_cube(p, r, "#bfdbfe", "#3b82f6", "#1d4ed8")

    def _draw_part(self, p, r, _s):
        self._draw_cube(p, r, "#fde68a", "#f59e0b", "#d97706")

    def _draw_mesh(self, p, r, _s):
        self._draw_cube(p, r, "#a5f3fc", "#06b6d4", "#0e7490", edge="#155e75", edge_w=0.9)
        p.setPen(self._pen("#0f766e", 0.9))
        top, left, right, pts = self._iso_pts(r)
        a, b, c, d, e, f, g = pts
        p.drawLine(a, c)
        p.drawLine(d, b)
        p.drawLine(QPointF((d.x() + e.x()) / 2, (d.y() + e.y()) / 2), c)
        p.drawLine(c, QPointF((b.x() + g.x()) / 2, (b.y() + g.y()) / 2))

    def _draw_layer_meshing(self, p, r, _s):
        self._draw_mesh(p, r, _s)

    def _draw_region(self, p, r, _s):
        self._draw_cube(p, r, "#fde047", "#eab308", "#ca8a04")

    def _draw_folder_regions(self, p, r, _s):
        self._draw_region(p, r, _s)

    def _draw_boundary(self, p, r, _s):
        p.setPen(self._pen("#7c3aed", 1.25))
        p.setBrush(QBrush(_c("#c4b5fd")))
        poly = QPolygonF([
            QPointF(r.left() + r.width() * 0.08, r.bottom() - 1),
            QPointF(r.left() + r.width() * 0.38, r.top() + 1),
            QPointF(r.right() - 1, r.top() + r.height() * 0.12),
            QPointF(r.right() - r.width() * 0.30, r.bottom() - 1),
        ])
        p.drawPolygon(poly)

    def _draw_layer_physics(self, p, r, _s):
        self._draw_cube(p, r, "#bbf7d0", "#22c55e", "#15803d")
        p.setPen(self._pen("#14532d", 1.1))
        font = QFont("Segoe UI", max(6, int(r.height() * 0.32)))
        font.setBold(True)
        p.setFont(font)
        p.drawText(r.toRect(), Qt.AlignCenter, "P")

    def _draw_continuum(self, p, r, _s):
        self._draw_layer_physics(p, r, _s)

    def _draw_solver(self, p, r, _s):
        p.setPen(self._pen("#c2410c", 1.2))
        p.setBrush(QBrush(_c("#fb923c")))
        cx, cy = r.center().x(), r.center().y()
        rad = min(r.width(), r.height()) * 0.42
        path = QPainterPath()
        n = 8
        for i in range(n * 2):
            ang = math.radians(-90 + i * 180.0 / n)
            rr = rad if i % 2 == 0 else rad * 0.58
            x, y = cx + math.cos(ang) * rr, cy + math.sin(ang) * rr
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        path.closeSubpath()
        p.drawPath(path)
        p.setBrush(QBrush(_c("#fff7ed")))
        p.drawEllipse(QPointF(cx, cy), rad * 0.32, rad * 0.32)

    def _draw_scene(self, p, r, _s):
        p.setPen(self._pen("#334155", 1.2))
        p.setBrush(QBrush(_c("#64748b")))
        body = QRectF(r.left() + r.width() * 0.08, r.top() + r.height() * 0.28,
                      r.width() * 0.62, r.height() * 0.50)
        p.drawRoundedRect(body, 2, 2)
        p.setBrush(QBrush(_c("#e2e8f0")))
        p.drawEllipse(body.adjusted(body.width() * 0.18, body.height() * 0.12,
                                    -body.width() * 0.18, -body.height() * 0.12))
        p.setBrush(QBrush(_c("#0ea5e9")))
        lens = body.adjusted(body.width() * 0.32, body.height() * 0.28,
                             -body.width() * 0.32, -body.height() * 0.28)
        p.drawEllipse(lens)
        p.setBrush(QBrush(_c("#475569")))
        p.setPen(Qt.NoPen)
        p.drawPolygon(QPolygonF([
            QPointF(body.right() - 1, body.center().y() - body.height() * 0.18),
            QPointF(r.right() - r.width() * 0.04, r.top() + r.height() * 0.22),
            QPointF(r.right() - r.width() * 0.04, r.bottom() - r.height() * 0.22),
            QPointF(body.right() - 1, body.center().y() + body.height() * 0.18),
        ]))

    def _draw_layer_visualization(self, p, r, _s):
        self._draw_scene(p, r, _s)

    def _draw_plot(self, p, r, _s):
        p.setPen(self._pen("#1e3a8a", 1.2))
        p.setBrush(QBrush(_c("#dbeafe")))
        p.drawRoundedRect(r, 2, 2)
        p.setPen(self._pen("#1d4ed8", 1.15))
        ox = r.left() + r.width() * 0.18
        oy = r.bottom() - r.height() * 0.18
        p.drawLine(QPointF(ox, r.top() + r.height() * 0.16), QPointF(ox, oy))
        p.drawLine(QPointF(ox, oy), QPointF(r.right() - r.width() * 0.12, oy))
        p.setPen(self._pen("#dc2626", 1.5))
        pts = [(0.22, 0.72), (0.40, 0.42), (0.58, 0.55), (0.78, 0.28)]
        qpts = [QPointF(r.left() + r.width() * x, r.top() + r.height() * y) for x, y in pts]
        for a, b in zip(qpts, qpts[1:]):
            p.drawLine(a, b)

    def _draw_monitor(self, p, r, _s):
        p.setPen(self._pen("#0f766e", 1.2))
        p.setBrush(QBrush(_c("#ccfbf1")))
        p.drawRoundedRect(r, 2, 2)
        p.setPen(self._pen("#0d9488", 1.5))
        path = QPainterPath()
        path.moveTo(r.left() + r.width() * 0.12, r.center().y())
        path.lineTo(r.left() + r.width() * 0.28, r.center().y() - r.height() * 0.22)
        path.lineTo(r.left() + r.width() * 0.46, r.center().y() + r.height() * 0.18)
        path.lineTo(r.left() + r.width() * 0.64, r.center().y() - r.height() * 0.08)
        path.lineTo(r.left() + r.width() * 0.86, r.center().y() + r.height() * 0.12)
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)

    def _draw_report(self, p, r, _s):
        self._draw_file(p, r, _s)
        p.setPen(self._pen("#2563eb", 1.15))
        for i in range(3):
            y = r.top() + r.height() * (0.42 + i * 0.16)
            p.drawLine(QPointF(r.left() + r.width() * 0.22, y),
                       QPointF(r.right() - r.width() * 0.22, y))

    def _draw_layer_post(self, p, r, _s):
        self._draw_plot(p, r, _s)

    def _draw_tools(self, p, r, _s):
        p.setPen(self._pen("#57534e", 1.6))
        p.setBrush(Qt.NoBrush)
        # wrench
        p.save()
        p.translate(r.center())
        p.rotate(-35)
        p.drawEllipse(QRectF(-r.width() * 0.28, -r.height() * 0.38,
                             r.width() * 0.32, r.height() * 0.32))
        p.setBrush(QBrush(_c("#a8a29e")))
        p.drawRoundedRect(QRectF(-r.width() * 0.10, -r.height() * 0.12,
                                 r.width() * 0.16, r.height() * 0.62), 1, 1)
        p.restore()

    def _draw_field(self, p, r, _s):
        p.setPen(self._pen("#6d28d9", 1.2))
        p.setBrush(QBrush(_c("#ede9fe")))
        p.drawRoundedRect(r, 2, 2)
        p.setPen(self._pen("#5b21b6", 1.2))
        font = QFont("Segoe UI", max(7, int(r.height() * 0.42)))
        font.setBold(True)
        font.setItalic(True)
        p.setFont(font)
        p.drawText(r.toRect(), Qt.AlignCenter, "f")

    def _draw_coord(self, p, r, _s):
        o = r.center() + QPointF(0, r.height() * 0.12)
        axes = [
            ("#dc2626", QPointF(r.width() * 0.42, r.height() * 0.08)),
            ("#16a34a", QPointF(-r.width() * 0.38, r.height() * 0.12)),
            ("#2563eb", QPointF(0, -r.height() * 0.46)),
        ]
        for col, d in axes:
            p.setPen(self._pen(col, 1.7))
            p.drawLine(o, o + d)
            tip = o + d
            p.setBrush(QBrush(_c(col)))
            p.setPen(Qt.NoPen)
            p.drawEllipse(tip, 2.2, 2.2)

    def _draw_units(self, p, r, _s):
        p.setPen(self._pen("#44403c", 1.2))
        p.setBrush(QBrush(_c("#f5f5f4")))
        p.drawRoundedRect(r, 2, 2)
        p.setPen(self._pen("#78716c", 1.0))
        for i in range(5):
            x = r.left() + r.width() * (0.16 + i * 0.17)
            h = r.height() * (0.42 if i % 2 == 0 else 0.22)
            p.drawLine(QPointF(x, r.bottom() - r.height() * 0.18),
                       QPointF(x, r.bottom() - r.height() * 0.18 - h))

    def _draw_table(self, p, r, _s):
        p.setPen(self._pen("#0369a1", 1.15))
        p.setBrush(QBrush(_c("#e0f2fe")))
        p.drawRoundedRect(r, 2, 2)
        p.setPen(self._pen("#0284c7", 1.0))
        p.drawLine(QPointF(r.left() + 2, r.top() + r.height() * 0.38),
                   QPointF(r.right() - 2, r.top() + r.height() * 0.38))
        p.drawLine(QPointF(r.center().x(), r.top() + 2),
                   QPointF(r.center().x(), r.bottom() - 2))

    def _draw_simulation(self, p, r, _s):
        self._draw_cube(p, r, "#7dd3d8", "#0e7490", "#155e75")
        p.setPen(self._pen("#134e4a", 1.0))
        font = QFont("Segoe UI", max(6, int(r.height() * 0.28)))
        font.setBold(True)
        p.setFont(font)
        p.drawText(r.toRect(), Qt.AlignCenter, "S")

    def _draw_model(self, p, r, _s):
        self._draw_layer_physics(p, r, _s)

    def _draw_displayer(self, p, r, _s):
        self._draw_scene(p, r, _s)

    def _draw_surface(self, p, r, _s):
        self._draw_boundary(p, r, _s)

    # ----- tools / window -----
    def _draw_info(self, p, r, _s):
        p.setPen(self._pen("#1d4ed8", 1.2))
        p.setBrush(QBrush(_c("#dbeafe")))
        p.drawEllipse(r)
        p.setPen(self._pen("#1e40af", 1.3))
        font = QFont("Segoe UI", max(8, int(r.height() * 0.55)))
        font.setBold(True)
        p.setFont(font)
        p.drawText(r.toRect(), Qt.AlignCenter, "i")

    def _draw_fingerprint(self, p, r, _s):
        p.setPen(self._pen("#4338ca", 1.3))
        p.setBrush(Qt.NoBrush)
        cx, cy = r.center().x(), r.center().y()
        for k in (0.28, 0.52, 0.78):
            p.drawEllipse(QPointF(cx, cy), r.width() * k * 0.5, r.height() * k * 0.5)

    def _draw_ruler(self, p, r, _s):
        self._draw_units(p, r, _s)

    def _draw_validate(self, p, r, _s):
        p.setPen(self._pen("#15803d", 1.2))
        p.setBrush(QBrush(_c("#dcfce7")))
        p.drawEllipse(r)
        p.setPen(self._pen("#16a34a", 2.0))
        p.drawLine(QPointF(r.left() + r.width() * 0.22, r.center().y()),
                   QPointF(r.left() + r.width() * 0.42, r.bottom() - r.height() * 0.28))
        p.drawLine(QPointF(r.left() + r.width() * 0.42, r.bottom() - r.height() * 0.28),
                   QPointF(r.right() - r.width() * 0.20, r.top() + r.height() * 0.28))

    def _draw_tree(self, p, r, _s):
        p.setPen(self._pen("#0369a1", 1.2))
        p.setBrush(QBrush(_c("#7dd3d8")))
        boxes = [
            (0.32, 0.06, 0.36, 0.22),
            (0.08, 0.42, 0.32, 0.22),
            (0.58, 0.42, 0.32, 0.22),
            (0.08, 0.74, 0.32, 0.20),
        ]
        for x, y, w, h in boxes:
            p.drawRoundedRect(QRectF(r.left() + r.width() * x, r.top() + r.height() * y,
                                     r.width() * w, r.height() * h), 1.5, 1.5)
        p.setPen(self._pen("#0e7490", 1.15))
        p.drawLine(QPointF(r.center().x(), r.top() + r.height() * 0.28),
                   QPointF(r.center().x(), r.top() + r.height() * 0.42))
        p.drawLine(QPointF(r.left() + r.width() * 0.24, r.top() + r.height() * 0.42),
                   QPointF(r.left() + r.width() * 0.74, r.top() + r.height() * 0.42))
        p.drawLine(QPointF(r.left() + r.width() * 0.24, r.top() + r.height() * 0.64),
                   QPointF(r.left() + r.width() * 0.24, r.top() + r.height() * 0.74))

    def _draw_properties(self, p, r, _s):
        p.setPen(self._pen("#334155", 1.15))
        p.setBrush(QBrush(_c("#f8fafc")))
        p.drawRoundedRect(r, 2, 2)
        p.setBrush(QBrush(_c("#e2e8f0")))
        p.drawRect(QRectF(r.left() + 1, r.top() + 1, r.width() * 0.45, r.height() - 2))
        p.setPen(self._pen("#64748b", 1.0))
        for i in range(3):
            y = r.top() + r.height() * (0.28 + i * 0.24)
            p.drawLine(QPointF(r.left() + 2, y), QPointF(r.right() - 2, y))

    def _draw_messages(self, p, r, _s):
        p.setPen(self._pen("#334155", 1.2))
        p.setBrush(QBrush(_c("#0f172a")))
        p.drawRoundedRect(r, 2, 2)
        p.setPen(self._pen("#22c55e", 1.15))
        p.drawLine(QPointF(r.left() + r.width() * 0.18, r.top() + r.height() * 0.32),
                   QPointF(r.right() - r.width() * 0.18, r.top() + r.height() * 0.32))
        p.drawLine(QPointF(r.left() + r.width() * 0.18, r.top() + r.height() * 0.52),
                   QPointF(r.left() + r.width() * 0.70, r.top() + r.height() * 0.52))
        p.drawLine(QPointF(r.left() + r.width() * 0.18, r.top() + r.height() * 0.72),
                   QPointF(r.left() + r.width() * 0.55, r.top() + r.height() * 0.72))

    def _draw_progress(self, p, r, _s):
        p.setPen(self._pen("#0369a1", 1.15))
        p.setBrush(QBrush(_c("#e0f2fe")))
        bar = QRectF(r.left(), r.center().y() - r.height() * 0.16,
                     r.width(), r.height() * 0.32)
        p.drawRoundedRect(bar, 3, 3)
        p.setBrush(QBrush(_c("#0284c7")))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(QRectF(bar.left() + 1, bar.top() + 1,
                                 bar.width() * 0.62, bar.height() - 2), 3, 3)

    def _draw_error(self, p, r, _s):
        p.setPen(self._pen("#991b1b", 1.2))
        p.setBrush(QBrush(_c("#fecaca")))
        p.drawEllipse(r)
        p.setPen(self._pen("#dc2626", 1.8))
        p.drawLine(QPointF(r.center().x(), r.top() + r.height() * 0.22),
                   QPointF(r.center().x(), r.center().y() + r.height() * 0.08))
        p.setBrush(QBrush(_c("#dc2626")))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(r.center().x(), r.bottom() - r.height() * 0.22), 1.6, 1.6)

    def _draw_warning(self, p, r, _s):
        p.setPen(self._pen("#a16207", 1.2))
        p.setBrush(QBrush(_c("#fde047")))
        poly = QPolygonF([
            QPointF(r.center().x(), r.top() + 1),
            QPointF(r.right() - 1, r.bottom() - 1),
            QPointF(r.left() + 1, r.bottom() - 1),
        ])
        p.drawPolygon(poly)

    def _draw_unknown(self, p, r, _s):
        p.setPen(self._pen("#94a3b8", 1.2))
        p.setBrush(QBrush(_c("#e2e8f0")))
        p.drawRoundedRect(r, 3, 3)

    def _draw_layer_unknown(self, p, r, _s):
        self._draw_unknown(p, r, _s)


# 模块级默认实例（QApplication 创建后再 get）
_default = None


def icons():
    global _default
    if _default is None:
        _default = AppIcons()
    return _default

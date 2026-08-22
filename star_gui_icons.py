# -*- coding: utf-8 -*-
"""star_gui_icons.py — AppIcons：集中图标引擎。

对齐 cabdecoding/cstdecoding 的 AppIcons 范式：先用 Qt 标准图标（QStyle::SP_*）
起步，后续替换为内嵌 SVG/PNG 资产。每个 key 绑定一个图标，主题化经 QIcon 重载。
"""

from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QStyle
from PyQt5.QtGui import QIcon


class AppIcons:
    """图标引擎：get(key) -> QIcon。"""

    # key -> (QStyle 标准像素图枚举, 大小)
    _STANDARD = {
        "open": "SP_DialogOpenButton",
        "save": "SP_DialogSaveButton",
        "new": "SP_FileIcon",
        "folder": "SP_DirIcon",
        "file": "SP_FileIcon",
        "fit": "SP_ArrowUp",
        "reset": "SP_BrowserReload",
        "zoom_in": "SP_ArrowUp",
        "zoom_out": "SP_ArrowDown",
        "play": "SP_MediaPlay",
        "stop": "SP_MediaStop",
        "error": "SP_MessageBoxCritical",
        "warning": "SP_MessageBoxWarning",
        "info": "SP_MessageBoxInformation",
        "tree": "SP_FileDialogDetailedView",
        "properties": "SP_FileDialogInfoView",
        "messages": "SP_MessageBoxInformation",
        "progress": "SP_BrowserReload",
        "part": "SP_FileDialogContentsView",
        "mesh": "SP_FileDialogContentsView",
        "scene": "SP_DesktopIcon",
        "plot": "SP_FileDialogDetailedView",
        "monitor": "SP_ComputerIcon",
        "report": "SP_FileDialogInfoView",
        "region": "SP_DirOpenIcon",
        "boundary": "SP_DirClosedIcon",
        "units": "SP_DriveHDIcon",
        "simulation": "SP_ComputerIcon",
        "layer_geometry": "SP_DirOpenIcon",
        "layer_meshing": "SP_DirClosedIcon",
        "layer_physics": "SP_ComputerIcon",
        "layer_visualization": "SP_DesktopIcon",
        "layer_post": "SP_FileDialogDetailedView",
        "layer_unknown": "SP_FileIcon",
        "unknown": "SP_FileIcon",
    }

    def __init__(self, style=None):
        self._style = style  # QStyle，缺省在 get 时用 QApplication.style()
        self._cache = {}

    def get(self, key, fallback="unknown"):
        """按 key 返回 QIcon；未知 key 回退到 fallback。"""
        name = self._STANDARD.get(key) or self._STANDARD.get(fallback, "SP_FileIcon")
        if key not in self._cache:
            self._cache[key] = self._standard_icon(name)
        return self._cache[key]

    def _standard_icon(self, style_enum_name):
        from PyQt5.QtWidgets import QApplication
        style = self._style or QApplication.style()
        try:
            sp = getattr(QStyle, style_enum_name)
        except AttributeError:
            sp = QStyle.SP_FileIcon
        return style.standardIcon(sp)

    def sized(self, key, w=16, h=16, fallback="unknown"):
        return self.get(key, fallback).pixmap(QSize(w, h))


# 模块级默认实例（QApplication 创建后再 get）
_default = None


def icons():
    global _default
    if _default is None:
        _default = AppIcons()
    return _default

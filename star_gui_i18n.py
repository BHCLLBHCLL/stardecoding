# -*- coding: utf-8 -*-
"""star_gui_i18n.py — 界面字符串表（中文默认 + 英文术语，对齐 STAR-CCM+ 截图菜单）。"""

LANG = "zh"

_T = {
    ("zh", "File"): "文件",
    ("zh", "Edit"): "编辑",
    ("zh", "Mesh"): "网格",
    ("zh", "Solution"): "求解",
    ("zh", "Tools"): "工具",
    ("zh", "Connection"): "连接",
    ("zh", "Window"): "窗口",
    ("zh", "Help"): "帮助",
    ("zh", "Scene"): "场景",
    ("zh", "Plot"): "绘图",
    ("zh", "Open..."): "打开...",
    ("zh", "Close"): "关闭",
    ("zh", "Exit"): "退出",
    ("zh", "Save"): "保存",
    ("zh", "Save As..."): "另存为...",
    ("zh", "Export STL..."): "导出 STL...",
    ("zh", "Export Summary..."): "导出摘要...",
    ("zh", "Export Report (JSON)..."): "导出报告 (JSON)...",
    ("zh", "Recent Files"): "最近文件",
    ("zh", "Fit View"): "适配视图",
    ("zh", "Reset View"): "重置视图",
    ("zh", "Wireframe"): "线框",
    ("zh", "Solid"): "实体",
    ("zh", "Edges Only"): "仅边线",
    ("zh", "Transparency"): "透明",
    ("zh", "Isometric"): "等轴测",
    ("zh", "Run"): "运行",
    ("zh", "Pause"): "暂停",
    ("zh", "Step"): "步进",
    ("zh", "Stop"): "停止",
    ("zh", "Version Fingerprint"): "版本指纹",
    ("zh", "State Length Check"): "状态表长度校验",
    ("zh", "ClassVersions Validate"): "ClassVersions 校验",
    ("zh", "About"): "关于",
    ("zh", "Simulation Tree"): "模型 / 场景/绘图",
    ("zh", "Properties"): "属性",
    ("zh", "Graphics Window"): "图形窗口",
    ("zh", "Messages"): "输出",
    ("zh", "Progress"): "进度",
    ("zh", "Output"): "输出",
    ("zh", "File Toolbar"): "文件工具栏",
    ("zh", "Solve Toolbar"): "求解工具栏",
    ("zh", "View Toolbar"): "视图工具栏",
    ("zh", "Display Toolbar"): "显示工具栏",
    ("zh", "Connect to Server"): "连接到服务器",
}


def set_language(lang):
    global LANG
    LANG = lang


def tr(text, lang=None):
    """翻译文本；未收录的字符串原样返回。"""
    key = (lang or LANG, text)
    return _T.get(key, text)

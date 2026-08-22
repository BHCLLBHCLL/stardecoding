# -*- coding: utf-8 -*-
"""star_gui_i18n.py — 界面字符串表（可选双语基础设施，对齐 cabdecoding cab_i18n）。

用法:
    from star_gui_i18n import tr, set_language
    set_language("zh")          # 默认中文
    tr("Simulation Tree")       # 返回当前语言字符串

目前界面默认中文 + 英文术语（与仓库文档一致）；如需全量双语切换，把界面内
硬编码字符串逐步迁移到 _T 表即可。标记 NYI 项与解析层（sim_parser）保持不动。
"""

LANG = "zh"

_T = {
    ("zh", "Simulation Tree"): "仿真树",
    ("zh", "Properties"): "属性",
    ("zh", "Graphics Window"): "图形窗口",
    ("zh", "Messages"): "消息",
    ("zh", "Progress"): "进度",
    ("zh", "File"): "文件",
    ("zh", "Open..."): "打开...",
    ("zh", "Close"): "关闭",
    ("zh", "Exit"): "退出",
    ("zh", "Scene"): "场景",
    ("zh", "Fit View"): "适配视图",
    ("zh", "Reset View"): "重置视图",
    ("zh", "Wireframe / Solid"): "线框 / 实体",
    ("zh", "Tools"): "工具",
    ("zh", "Help"): "帮助",
    ("zh", "About"): "关于",
}


def set_language(lang):
    global LANG
    LANG = lang


def tr(text, lang=None):
    """翻译文本；未收录的字符串原样返回。"""
    key = (lang or LANG, text)
    return _T.get(key, text)

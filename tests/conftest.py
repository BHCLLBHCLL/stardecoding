# -*- coding: utf-8 -*-
"""GUI 测试公共配置：minimal 平台（offscreen 下 QVTK 会段错误）。"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

# -*- coding: utf-8 -*-
"""GUI 测试公共配置。

平台说明（本机实测）：
  - offscreen：纯 Qt 测试（树/属性/摘要）稳定；
  - minimal：QVTK 3D 视口测试稳定（offscreen 下 QVTK 段错误）；
各测试文件在模块级自设 QT_QPA_PLATFORM，见 test_gui_m{0,1}.py（offscreen）与
test_gui_m{2,3}.py（minimal）。tests/run_all.py 逐文件子进程运行，保证平台独立。
"""

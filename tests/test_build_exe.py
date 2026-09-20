# -*- coding: utf-8 -*-
"""S5 第三项：真实打包（PyInstaller）—— 构建脚本与产物冒烟。

背景："真实打包"此前一直是未动项（PyInstaller 未安装）。本轮装上并生成
`dist/stardecoding-cli.exe`（onefile，236.3 MB），冒烟用真实 .sim 跑 `--report`
输出正确解析结果、退出码 0。

本文件验证：① 构建脚本能生成**语法合法**的冻结入口；② 产物存在时能在无 Python
环境下解析真实 .sim（不存在则如实 skip —— 产物 236 MB 不入库）。
"""
import ast
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest  # noqa: E402

import build_exe  # noqa: E402

SIM = os.path.join(ROOT, "adjointWing_start.sim")
EXE = os.path.join(ROOT, "dist", "stardecoding-cli.exe")


def test_build_script_generates_valid_entry():
    """冻结入口必须是合法 Python（曾因三引号转义写坏 → PyInstaller 报 SyntaxError）。"""
    entry = build_exe.ensure_entry()
    assert os.path.exists(entry)
    src = open(entry, encoding="utf-8").read()
    ast.parse(src)                                  # 语法必须成立
    assert "freeze_support" in src                   # 多进程保护
    assert "from sim_parser import main" in src      # 入口名必须与解析器一致


def test_spec_excludes_heavy_stack():
    """构建参数必须排除巨型依赖（scipy hidden-import 会把 sklearn/tensorflow 拖进来）。"""
    src = open(os.path.join(ROOT, "build_exe.py"), encoding="utf-8").read()
    for mod in ("tensorflow", "sklearn", "astropy", "pyarrow"):
        assert mod in src
    assert '"--hidden-import", "scipy"' not in src


@pytest.mark.skipif(not os.path.exists(EXE),
                    reason="未构建二进制（dist/ 不入库；先跑 python build_exe.py）")
def test_built_binary_parses_real_sim():
    """产物冒烟：无 Python 环境依赖地解析真实 .sim（真实打包的验收证据）。"""
    if not os.path.exists(SIM):
        pytest.skip("缺少语料 adjointWing_start.sim")
    out = subprocess.run([EXE, "--report", SIM], cwd=ROOT,
                         capture_output=True, timeout=300)
    assert out.returncode == 0, out.stderr[-500:]
    text = out.stdout.decode("utf-8", errors="replace")
    assert "Part" in text and "Scene" in text
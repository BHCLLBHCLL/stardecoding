# -*- coding: utf-8 -*-
"""S5 打包：用 PyInstaller 生成**真实二进制**（本仓库首个可分发 exe）。

背景（S5 第三项验收）：此前"真实打包"一直是未动项（PyInstaller 未安装）。本脚本把它
变成可复现步骤：

  python build_exe.py            # 构建 CLI 二进制（默认 onefile）
  python build_exe.py --onedir   # 目录式（启动更快，便于分发 zip）
  python build_exe.py --smoke    # 构建后立刻用真实 .sim 冒烟验证

产物：dist/stardecoding-cli.exe（或 dist/stardecoding-cli/）。二进制**不入库**（数十 MB），
入库的是本脚本 + 冒烟证据（build_exe.py --smoke 会在 stdout 打印可核对的解析结果）。

诚实边界：GUI（PyQt5 + VTK）打包未做 —— 依赖体积大、构建慢，且需要额外的 Qt 插件收集；
本脚本只覆盖 CLI（解析/报告/网格统计/导出），这也足够证明"真实二进制"这条链是通的。
"""
import argparse
import os
import subprocess
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.abspath(__file__))
ENTRY = os.path.join(ROOT, "cli_main.py")
NAME = "stardecoding-cli"

ENTRY_SRC = '\n'.join([
    '# -*- coding: utf-8 -*-',
    '"""打包入口（S5）：把 sim_parser 的 CLI 暴露成独立二进制。"""',
    'import multiprocessing',
    'import sys',
    '',
    '',
    'def main(argv=None):',
    '    multiprocessing.freeze_support()          # PyInstaller + 多进程保护',
    '    from sim_parser import main as parser_main',
    '    return parser_main(argv)',
    '',
    '',
    'if __name__ == "__main__":',
    '    raise SystemExit(main())',
]) + '\n'


def ensure_entry():
    """生成冻结入口（幂等）。"""
    if not os.path.exists(ENTRY) or open(ENTRY, encoding="utf-8").read() != ENTRY_SRC:
        with open(ENTRY, "w", encoding="utf-8") as fh:
            fh.write(ENTRY_SRC)
    return ENTRY


def build(onedir=False, clean=True):
    entry = ensure_entry()
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm",
           "--name", NAME,
           "--distpath", os.path.join(ROOT, "dist"),
           "--workpath", os.path.join(ROOT, "build"),
           "--specpath", os.path.join(ROOT, "build"),
           "--onedir" if onedir else "--onefile",
           "--console"]
    # 解析器只用 numpy（scipy 仅求解器模块需要，本 CLI 不 import —— 实测把 scipy 作为
    # hidden-import 会触发 PyInstaller 的巨型 hook 图，把 sklearn/tensorflow/astropy
    # 全拖进来，构建时间 >10 分钟且产物体积失控）。
    cmd += ["--hidden-import", "numpy"]
    for excl in ("PyQt5", "vtk", "matplotlib", "pyamg", "scipy", "pandas",
                 "sklearn", "skimage", "tensorflow", "astropy", "pyarrow",
                 "distributed", "numba", "IPython", "notebook", "pytest"):
        cmd += ["--exclude-module", excl]
    cmd.append(entry)
    print("$ " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=ROOT)
    exe = os.path.join(ROOT, "dist", NAME + (".exe" if not onedir else ""))
    if onedir:
        exe = os.path.join(ROOT, "dist", NAME, NAME + ".exe")
    return rc, exe


def smoke(exe, sim_path):
    """用真实 .sim 跑一次二进制，验证它不是空壳。"""
    if not os.path.exists(exe):
        print("[smoke] 未找到二进制 %s" % exe)
        return 1
    print("[smoke] %s --report %s" % (exe, sim_path))
    rc = subprocess.call([exe, "--report", sim_path], cwd=ROOT)
    print("[smoke] 退出码 %d" % rc)
    return rc


def _main(argv=None):
    ap = argparse.ArgumentParser(description="S5 真实打包（PyInstaller）")
    ap.add_argument("--onedir", action="store_true", help="目录式产物（默认 onefile）")
    ap.add_argument("--smoke", action="store_true", help="构建后用真实 .sim 冒烟")
    ap.add_argument("--sim", default=os.path.join(ROOT, "adjointWing_start.sim"))
    ap.add_argument("--no-clean", action="store_true")
    args = ap.parse_args(argv)
    rc, exe = build(onedir=args.onedir, clean=not args.no_clean)
    if rc != 0:
        print("构建失败，退出码 %d" % rc)
        return rc
    size = 0
    if os.path.isfile(exe):
        size = os.path.getsize(exe)
    elif os.path.isdir(exe):
        for base, _d, files in os.walk(exe):
            size += sum(os.path.getsize(os.path.join(base, f)) for f in files)
    print("产物：%s（%.1f MB）" % (exe, size / 1048576.0))
    if args.smoke:
        return smoke(exe, args.sim)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

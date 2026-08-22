# -*- coding: utf-8 -*-
"""GUI 测试总入口：逐文件子进程运行 pytest（QVTK 与既有 GUI 测试同进程会崩溃）。

用法:
    python tests/run_all.py            # 全部测试文件
    python tests/run_all.py m2         # 只跑含 m2 的文件
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(ROOT, "tests")


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    files = sorted(f for f in os.listdir(TESTS)
                   if f.startswith("test_") and f.endswith(".py"))
    if only:
        files = [f for f in files if only in f]
    failed = []
    for f in files:
        print("=== %s ===" % f)
        p = subprocess.run([sys.executable, "-m", "pytest", os.path.join(TESTS, f),
                            "-q", "--tb=short"], cwd=ROOT)
        if p.returncode != 0:
            failed.append((f, p.returncode))
    print()
    if failed:
        print("FAILED files:", failed)
        return 1
    print("ALL GUI TEST FILES PASSED (%d)" % len(files))
    return 0


if __name__ == "__main__":
    sys.exit(main())

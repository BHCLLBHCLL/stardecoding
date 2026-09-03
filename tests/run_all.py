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
                            "-q", "--tb=short"], cwd=ROOT,
                           capture_output=True, text=True)
        out = (p.stdout or "") + (p.stderr or "")
        print(out.strip().splitlines()[-1] if out.strip() else "(no output)")
        # 注意：QVTK 在 headless 平台退出时会段错误（-1073741819），但测试本身
        # 已全部通过——按 pytest 摘要行判定成败，而不是进程退出码。
        import re as _re
        failed_or_err = _re.search(r"\d+ failed|\d+ error|no tests ran", out)
        # OCC 门控测试在非 OCC 环境整体 skip：全部跳过视为通过（不误报），
        # 有任何 failed/error/no tests 仍判失败。
        ok = (bool(_re.search(r"\d+ passed", out))
              or bool(_re.search(r"\d+ skipped", out))) and not failed_or_err
        if not ok:
            failed.append((f, p.returncode))
            print(out[-1500:])
    print()
    if failed:
        print("FAILED files:", failed)
        return 1
    print("ALL GUI TEST FILES PASSED (%d)" % len(files))
    return 0


if __name__ == "__main__":
    sys.exit(main())

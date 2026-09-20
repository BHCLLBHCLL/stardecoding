# -*- coding: utf-8 -*-
"""打包入口（S5）：把 sim_parser 的 CLI 暴露成独立二进制。"""
import multiprocessing
import sys


def main(argv=None):
    multiprocessing.freeze_support()          # PyInstaller + 多进程保护
    from sim_parser import main as parser_main
    return parser_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

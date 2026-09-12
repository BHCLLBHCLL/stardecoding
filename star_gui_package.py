# -*- coding: utf-8 -*-
"""star_gui_package.py — 打包 / 安装器 / 版本发布 纯逻辑（无 Qt）。

X4：PyInstaller 打包 + 安装器（Inno Setup）+ 版本发布流程。
本环境未检测到 PyInstaller/cx_Freeze/nuitka → **诚实降级**：不实际打包，
只产出可复现的打包配置（`.spec` / `--add-data` 命令行 / 自动构建脚本）、
安装器脚本（`.iss`）、版本清单与发布说明，并显式报告打包器可用性。

打包器可用时，`build_plan()` 的 `available` 为 True，生成的脚本可直接执行；
不可用时给出安装指引，绝不假装已打包。
"""
import importlib.util
import os
import platform
import re
import sys

import star_gui_help as help_

APP_NAME = help_.APP_NAME
APP_VERSION = help_.APP_VERSION

ROOT = os.path.dirname(os.path.abspath(__file__))
ENTRY_SCRIPT = "star_gui.py"
REQUIREMENTS = "requirements-gui.txt"
ICON_FILE = "star_gui.ico"

PACKAGERS = ("PyInstaller", "cx_Freeze", "nuitka")

DATA_FILES = (
    ("star_gui_theme.qss", "."),
    ("doc_javadoc_catalog.md", "."),
    ("doc_userguide_sim.md", "."),
    ("star_gui_parity.md", "."),
    ("README.md", "."),
)

DIST_DIR = "dist"
BUILD_DIR = "build"

_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([A-Za-z_][\w.]*)\s+import|import\s+([A-Za-z_][\w.]*))", re.M)


def app_slug():
    return "star_gui"


def toolchain():
    found = {}
    for name in PACKAGERS:
        try:
            found[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found[name] = False
    return found


def available_packager():
    tc = toolchain()
    for name in PACKAGERS:
        if tc.get(name):
            return name
    return None


def is_available():
    return available_packager() is not None


def is_frozen():
    return bool(getattr(sys, "frozen", False))


def read_requirements():
    path = os.path.join(ROOT, REQUIREMENTS)
    reqs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            reqs.append(line)
    return reqs


def _source_files():
    files = [ENTRY_SCRIPT]
    for name in sorted(os.listdir(ROOT)):
        if name.startswith("star_gui") and name.endswith(".py"):
            files.append(name)
    return files


def local_modules():
    found = set()
    for fn in _source_files():
        path = os.path.join(ROOT, fn)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for a, b in _IMPORT_RE.findall(text):
            mod = (a or b).split(".")[0]
            if mod == os.path.splitext(fn)[0]:
                continue
            if os.path.exists(os.path.join(ROOT, mod + ".py")):
                found.add(mod)
    return sorted(found)


def resource_manifest():
    out = []
    for src, dest in DATA_FILES:
        path = os.path.join(ROOT, src)
        out.append({"source": src, "path": path, "dest": dest,
                    "exists": os.path.exists(path)})
    return out


def missing_resources():
    return [r["source"] for r in resource_manifest() if not r["exists"]]


def version_manifest():
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "entry": ENTRY_SCRIPT,
        "python": platform.python_version(),
        "platform": platform.system(),
        "requirements": read_requirements(),
        "hidden_imports": local_modules(),
    }


def version_string():
    return "%s %s" % (APP_NAME, APP_VERSION)


def add_data_args():
    return ["%s%s%s" % (r["source"], os.pathsep, r["dest"])
            for r in resource_manifest()]


def pyinstaller_command(onefile=False, windowed=True):
    cmd = ["pyinstaller", "--noconfirm", "--clean",
           "--name", app_slug(),
           "--distpath", DIST_DIR, "--workpath", BUILD_DIR]
    if onefile:
        cmd.append("--onefile")
    if windowed:
        cmd.append("--windowed")
    for arg in add_data_args():
        cmd += ["--add-data", arg]
    for mod in local_modules():
        cmd += ["--hidden-import", mod]
    cmd += ["--paths", ROOT]
    cmd.append(ENTRY_SCRIPT)
    return cmd


def command_line(onefile=False, windowed=True):
    parts = []
    for token in pyinstaller_command(onefile=onefile, windowed=windowed):
        parts.append('"%s"' % token if " " in token else token)
    return " ".join(parts)


def pyinstaller_spec_text(onefile=False):
    datas = ",\n        ".join("('%s', '%s')" % (r["source"], r["dest"])
                              for r in resource_manifest())
    hidden = ", ".join("'%s'" % m for m in local_modules())
    lines = [
        "# -*- mode: python ; coding: utf-8 -*-",
        "",
        "block_cipher = None",
        "",
        "a = Analysis(",
        "    ['%s']," % ENTRY_SCRIPT,
        "    pathex=['%s']," % ROOT.replace("\\", "/"),
        "    binaries=[],",
        "    datas=[",
        "        %s," % datas,
        "    ],",
        "    hiddenimports=[%s]," % hidden,
        "    hookspath=[],",
        "    hooksconfig={},",
        "    runtime_hooks=[],",
        "    excludes=[],",
        "    noarchive=False,",
        ")",
        "pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)",
        "",
    ]
    if onefile:
        lines += [
            "exe = EXE(",
            "    pyz,",
            "    a.scripts,",
            "    a.binaries,",
            "    a.datas,",
            "    [],",
            "    name='%s'," % app_slug(),
            "    debug=False,",
            "    strip=False,",
            "    upx=True,",
            "    console=False,",
            ")",
        ]
    else:
        lines += [
            "exe = EXE(",
            "    pyz,",
            "    a.scripts,",
            "    [],",
            "    exclude_binaries=True,",
            "    name='%s'," % app_slug(),
            "    debug=False,",
            "    strip=False,",
            "    upx=True,",
            "    console=False,",
            ")",
            "coll = COLLECT(",
            "    exe,",
            "    a.binaries,",
            "    a.zipfiles,",
            "    a.datas,",
            "    strip=False,",
            "    upx=True,",
            "    upx_exclude=[],",
            "    name='%s'," % app_slug(),
            ")",
        ]
    return "\n".join(lines) + "\n"


def expected_artifact(onefile=False):
    exe = app_slug() + (".exe" if platform.system() == "Windows" else "")
    if onefile:
        return os.path.join(DIST_DIR, exe)
    return os.path.join(DIST_DIR, app_slug(), exe)


def inno_setup_text():
    exe = app_slug() + ".exe"
    lines = [
        "; 自动生成：Inno Setup 安装器脚本（X4）",
        "; 依赖 PyInstaller 产物（dist\\star_gui）",
        "",
        "[Setup]",
        "AppName=%s" % APP_NAME,
        "AppVersion=%s" % APP_VERSION,
        "DefaultDirName={autopf}\\%s" % app_slug(),
        "DefaultGroupName=%s" % APP_NAME,
        "OutputBaseFilename=%s-setup-%s" % (app_slug(), APP_VERSION),
        "Compression=lzma2",
        "SolidCompression=yes",
        "WizardStyle=modern",
        "",
        "[Files]",
        'Source: "%s\\%s\\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs'
        % (DIST_DIR, app_slug()),
        "",
        "[Icons]",
        'Name: "{group}\\%s"; Filename: "{app}\\%s"' % (APP_NAME, exe),
        'Name: "{group}\\卸载 %s"; Filename: "{uninstallexe}"' % APP_NAME,
        "",
        "[Run]",
        'Filename: "{app}\\%s"; Description: "启动 %s"; Flags: nowait postinstall skipifsilent'
        % (exe, APP_NAME),
    ]
    return "\n".join(lines) + "\n"


def build_script_text(onefile=False):
    cmd = pyinstaller_command(onefile=onefile)
    lines = [
        "# -*- coding: utf-8 -*-",
        '"""自动生成：PyInstaller 打包 star_gui（X4）。"""',
        "import subprocess",
        "import sys",
        "",
        "CMD = %r" % (cmd,),
        "",
        "",
        "def main():",
        "    print('run:', ' '.join(CMD))",
        "    return subprocess.call(CMD)",
        "",
        "",
        "if __name__ == '__main__':",
        "    sys.exit(main())",
    ]
    return "\n".join(lines) + "\n"


def release_notes():
    packager = available_packager()
    status = ("已检测到打包器 %s，可直接执行 build_star_gui.py 打包。" % packager
              if packager else
              "未检测到 PyInstaller（pip install pyinstaller）→ 当前仅生成配置，不实际打包。")
    lines = [
        "# %s %s" % (APP_NAME, APP_VERSION),
        "",
        "## 版本清单",
        "- 应用: %s" % APP_NAME,
        "- 版本: %s" % APP_VERSION,
        "- 入口: %s" % ENTRY_SCRIPT,
        "- Python: %s (%s)" % (platform.python_version(), platform.system()),
        "- 运行依赖: %s" % " / ".join(read_requirements()),
        "",
        "## 打包",
        "- 打包器状态: %s" % status,
        "- 命令: %s" % command_line(),
        "- 规格文件: star_gui.spec",
        "- 产物: %s" % expected_artifact(),
        "",
        "## 安装器",
        "- Inno Setup 脚本: installer.iss",
        "- 安装包: %s-setup-%s.exe" % (app_slug(), APP_VERSION),
        "",
        "## 发布流程",
        "1. pip install -r %s" % REQUIREMENTS,
        "2. pip install pyinstaller",
        "3. python build_star_gui.py",
        "4. iscc installer.iss",
        "5. 校验 %s 后归档发布" % expected_artifact(),
    ]
    return "\n".join(lines) + "\n"


def build_plan(onefile=False):
    packager = available_packager()
    notes = []
    if packager is None:
        notes.append("未检测到 PyInstaller/cx_Freeze/nuitka，本环境不实际打包（诚实降级）。")
        notes.append("安装打包器: python -m pip install pyinstaller")
    else:
        notes.append("打包器 %s 可用，可执行生成的构建脚本。" % packager)
    missing = missing_resources()
    if missing:
        notes.append("缺失资源文件: %s" % ", ".join(missing))
    return {
        "available": packager is not None,
        "packager": packager,
        "toolchain": toolchain(),
        "artifacts": sorted(BUILD_SCRIPTS),
        "expected": expected_artifact(onefile=onefile),
        "command": command_line(onefile=onefile),
        "missing": missing,
        "notes": notes,
    }


BUILD_SCRIPTS = {
    "star_gui.spec": pyinstaller_spec_text,
    "installer.iss": inno_setup_text,
    "build_star_gui.py": build_script_text,
    "RELEASE_NOTES.md": release_notes,
}


def write_build_scripts(outdir):
    written = []
    for name in sorted(BUILD_SCRIPTS):
        path = os.path.join(outdir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(BUILD_SCRIPTS[name]())
        written.append(path)
    return written

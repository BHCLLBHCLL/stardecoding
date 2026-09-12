# -*- coding: utf-8 -*-
"""X4：PyInstaller 打包 + 安装器 + 版本发布流程（诚实降级）。"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import star_gui_package as pk
import star_gui_help as hs


def test_toolchain_probe():
    tc = pk.toolchain()
    assert set(tc) == set(pk.PACKAGERS)
    assert all(isinstance(v, bool) for v in tc.values())
    expected = next((n for n in pk.PACKAGERS if tc[n]), None)
    assert pk.available_packager() == expected


def test_is_available_and_frozen():
    assert pk.is_available() == (pk.available_packager() is not None)
    assert pk.is_frozen() is False


def test_read_requirements():
    reqs = pk.read_requirements()
    assert "PyQt5>=5.15" in reqs
    assert "vtk>=9.3" in reqs
    assert "numpy>=1.24" in reqs
    assert all(not r.startswith("#") for r in reqs)


def test_local_modules():
    mods = pk.local_modules()
    assert "star_gui_panes" in mods
    assert "star_gui_helpwin" in mods
    assert "sim_writer" in mods
    assert "os" not in mods and "sys" not in mods
    assert mods == sorted(mods)


def test_resource_manifest():
    manifest = pk.resource_manifest()
    names = [r["source"] for r in manifest]
    assert "star_gui_theme.qss" in names
    assert "doc_javadoc_catalog.md" in names
    assert all({"source", "path", "dest", "exists"} <= set(r) for r in manifest)


def test_core_resources_present():
    missing = pk.missing_resources()
    assert "star_gui_theme.qss" not in missing
    assert "doc_javadoc_catalog.md" not in missing
    assert missing == []


def test_version_manifest():
    vm = pk.version_manifest()
    assert vm["name"] == hs.APP_NAME
    assert vm["version"] == hs.APP_VERSION
    assert vm["entry"] == pk.ENTRY_SCRIPT
    assert vm["hidden_imports"] == pk.local_modules()
    assert vm["requirements"]
    assert pk.version_string() == "%s %s" % (hs.APP_NAME, hs.APP_VERSION)


def test_add_data_args():
    args = pk.add_data_args()
    assert len(args) == len(pk.DATA_FILES)
    assert all(os.pathsep in a for a in args)


def test_pyinstaller_command():
    cmd = pk.pyinstaller_command()
    assert cmd[0] == "pyinstaller"
    assert "--name" in cmd and "--windowed" in cmd and "--add-data" in cmd
    assert "--hidden-import" in cmd
    assert cmd[-1] == pk.ENTRY_SCRIPT
    assert pk.expected_artifact().endswith(pk.app_slug() + ".exe")


def test_pyinstaller_spec_text():
    spec = pk.pyinstaller_spec_text()
    assert "Analysis(" in spec
    assert "COLLECT(" in spec
    assert "EXE(" in spec
    assert "datas=[" in spec
    assert "hiddenimports=[" in spec
    assert "star_gui_theme.qss" in spec
    assert "star_gui.py" in spec
    assert "PYZ(" in spec


def test_inno_setup_text():
    iss = pk.inno_setup_text()
    assert "[Setup]" in iss
    assert "[Files]" in iss
    assert "[Icons]" in iss
    assert "AppName=%s" % hs.APP_NAME in iss
    assert "AppVersion=%s" % hs.APP_VERSION in iss
    assert pk.DIST_DIR in iss


def test_build_script_text():
    text = pk.build_script_text()
    assert "subprocess" in text
    assert "pyinstaller" in text
    assert pk.ENTRY_SCRIPT in text


def test_release_notes():
    notes = pk.release_notes()
    assert hs.APP_VERSION in notes
    assert hs.APP_NAME in notes
    assert "PyInstaller" in notes
    assert "installer.iss" in notes
    assert "star_gui.spec" in notes


def test_build_plan_honest_degradation():
    plan = pk.build_plan()
    assert plan["available"] == (plan["packager"] is not None)
    assert plan["artifacts"] == sorted(pk.BUILD_SCRIPTS)
    assert plan["notes"]
    if not plan["available"]:
        joined = " ".join(plan["notes"])
        assert "PyInstaller" in joined
        assert "pip install pyinstaller" in joined
        assert plan["expected"] == pk.expected_artifact()


def test_write_build_scripts():
    outdir = tempfile.mkdtemp(prefix="x4pkg_")
    try:
        written = pk.write_build_scripts(outdir)
        assert len(written) == len(pk.BUILD_SCRIPTS)
        for name in pk.BUILD_SCRIPTS:
            path = os.path.join(outdir, name)
            assert os.path.exists(path)
            with open(path, encoding="utf-8") as f:
                assert f.read() == pk.BUILD_SCRIPTS[name]()
    finally:
        for name in pk.BUILD_SCRIPTS:
            path = os.path.join(outdir, name)
            if os.path.exists(path):
                os.remove(path)
        os.rmdir(outdir)

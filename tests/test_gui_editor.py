# -*- coding: utf-8 -*-
"""E0–E8：查看器+编辑器（命令总线 / 会话 / 回写 / 外壳）。"""
import os
import shutil
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest

SIM = os.path.join(ROOT, "adjointWing_start.sim")


@pytest.fixture(scope="module")
def app():
    from PyQt5.QtWidgets import QApplication
    qapp = QApplication.instance() or QApplication([])
    return qapp


def _wait_loaded(win, app, timeout=30):
    t0 = time.time()
    while win.sim is None and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.02)
    assert win.sim is not None


def test_command_bus_set_property_undo():
    from sim_parser import SimFile
    from star_gui_commands import SetPropertyCommand
    from star_gui_document import SimDocument

    sim = SimFile(SIM)
    obj = next(o for o in sim.objects if o.dict.get("PresentationName") == "Fluid Domain"
               and o.class_name == "star.common.Region")
    doc = SimDocument(sim, SIM)
    old = obj.dict.get("PresentationName")
    doc.execute(SetPropertyCommand(obj.id, "PresentationName", "Renamed Domain", old))
    assert obj.dict["PresentationName"] == "Renamed Domain"
    assert doc.dirty
    assert doc.bus.can_undo()
    assert doc.undo()
    assert obj.dict["PresentationName"] == old
    assert doc.bus.can_redo()
    assert doc.redo()
    assert obj.dict["PresentationName"] == "Renamed Domain"


def test_sim_document_bind_clears_dirty():
    from sim_parser import SimFile
    from star_gui_commands import SetPropertyCommand
    from star_gui_document import SimDocument

    sim = SimFile(SIM)
    obj = next(o for o in sim.objects if o.dict.get("PresentationName"))
    doc = SimDocument(sim, SIM)
    doc.execute(SetPropertyCommand(obj.id, "PresentationName", "X", obj.dict.get("PresentationName")))
    assert doc.dirty
    doc.bind(sim, SIM)
    assert not doc.dirty
    assert not doc.bus.can_undo()


def test_tree_has_editor_folders():
    from sim_parser import SimFile
    from star_gui_model import StarSceneModel

    m = StarSceneModel(SimFile(SIM))
    folders = [c.label for c in m.sim_tree()[0].children]
    for name in ("Geometry", "Operations", "Derived Parts", "3D-CAD",
                 "Continua", "Regions", "Scenes", "Tools"):
        assert name in folders, folders


def test_parse_property_text_types():
    from star_gui_panes import parse_property_text

    assert parse_property_text(True, "false") is False
    assert parse_property_text(True, "1") is True
    assert parse_property_text(3, "12") == 12
    assert parse_property_text(1.5, "2.25") == pytest.approx(2.25)
    assert parse_property_text([0.1, 0.2, 0.3], "0.4, 0.5, 0.6") == [0.4, 0.5, 0.6]
    assert parse_property_text("abc", "xyz") == "xyz"
    assert parse_property_text(3, "nope") is None


def test_writer_roundtrip_presentation_name():
    from sim_parser import SimFile
    from sim_writer import format_repr, save_sim
    from star_gui_document import SimDocument
    from star_gui_commands import SetPropertyCommand

    assert format_repr(None) == "None"
    assert format_repr(True) == "True"
    assert format_repr([1, 2]) == "[1, 2]"
    assert "'hello'" in format_repr("hello")

    tmp = tempfile.mkdtemp(prefix="star_e4_")
    try:
        dest = os.path.join(tmp, "wing.sim")
        sim = SimFile(SIM)
        obj = next(o for o in sim.objects
                   if o.dict.get("PresentationName") == "Fluid Domain"
                   and o.class_name == "star.common.Region")
        doc = SimDocument(sim, SIM)
        old = obj.dict["PresentationName"]
        doc.execute(SetPropertyCommand(obj.id, "PresentationName", "Fluid Domain Edit", old))
        save_sim(sim, dest, patches=doc.patches, src_path=SIM)
        reloaded = SimFile(dest)
        hit = reloaded.objmap.get(obj.id)
        assert hit is not None
        assert hit.dict.get("PresentationName") == "Fluid Domain Edit"
        assert reloaded.object_by_id(2).class_name == "star.common.Simulation"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_transform_command_undo():
    from sim_parser import SimFile
    from star_gui_commands import TransformPartCommand
    from star_gui_document import SimDocument

    sim = SimFile(SIM)
    part = next(o for o in sim.objects if (o.class_name or "").endswith("MeshPart")
                or (o.class_name or "").endswith("CadPart"))
    doc = SimDocument(sim, SIM)
    doc.execute(TransformPartCommand(part.id, translate=(1, 2, 3), scale=(2, 2, 2)))
    t = doc.transforms[part.id]
    assert t[0] == pytest.approx(1.0)
    assert t[3] == pytest.approx(2.0)
    doc.undo()
    t2 = doc.transforms[part.id]
    assert t2[0] == pytest.approx(0.0)
    assert t2[3] == pytest.approx(1.0)


def test_copy_delete_session():
    from sim_parser import SimFile
    from star_gui_commands import CopyObjectCommand, DeleteObjectCommand
    from star_gui_document import SimDocument

    sim = SimFile(SIM)
    src = next(o for o in sim.objects if o.dict.get("PresentationName") == "Fluid Domain"
               and o.class_name == "star.common.Region")
    doc = SimDocument(sim, SIM)
    cmd = CopyObjectCommand(src.id)
    doc.execute(cmd)
    assert cmd.new_id in doc.created
    clone = doc.object(cmd.new_id)
    assert clone is not None
    assert "Copy" in (clone.name or "")
    doc.execute(DeleteObjectCommand(cmd.new_id))
    assert doc.object(cmd.new_id) is None
    doc.undo()
    assert doc.object(cmd.new_id) is not None


def test_plot_collector():
    from sim_parser import SimFile
    from star_gui_plots import collect_plot_rows

    rows = collect_plot_rows(SimFile(SIM))
    kinds = {r[0] for r in rows}
    assert any("Monitor" in k or "Plot" in k or "Report" in k for k in kinds)
    names = [r[1] for r in rows]
    assert any("Residual" in (n or "") for n in names) or len(rows) > 0


def test_menus_toolbars_and_disabled_solver(app):
    from star_gui import StarMainWindow

    win = StarMainWindow()
    titles = [a.text().replace("&", "") for a in win.menuBar().actions()]
    blob = " ".join(titles)
    for name in ("文件", "编辑", "网格", "场景", "求解", "工具", "连接", "窗口", "帮助"):
        assert name in blob, titles
    names = [tb.objectName() for tb in win.findChildren(win.tb_file.__class__)]
    for n in ("File", "Solve", "View", "Display"):
        assert n in names, names
    assert win.windowTitle().startswith("STAR-CCM+ .sim Viewer / Editor")
    assert not win.actions["Solution>Run"].isEnabled()
    assert not win.actions["Solution>Pause"].isEnabled()
    assert not win.actions["Connection>Server"].isEnabled()
    assert "File>Save" in win.actions
    assert "Edit>Undo" in win.actions
    assert "Mesh>Generate" in win.actions
    assert "Window>Plots" in win.actions
    assert "Window>Cad" in win.actions
    assert hasattr(win.tree_widget, "context_command")
    assert hasattr(win.props_widget, "property_edited")
    win.close()


def test_gui_editor_session_and_kernel_nyi(app):
    from star_gui import StarMainWindow
    from star_gui_commands import SetPropertyCommand

    win = StarMainWindow()
    win.show()
    win.load_file(SIM)
    _wait_loaded(win, app)
    folders = [win.tree_widget.tree.topLevelItem(0).child(i).text(0)
               for i in range(win.tree_widget.tree.topLevelItem(0).childCount())]
    assert "Operations" in folders
    assert "Derived Parts" in folders
    assert "3D-CAD" in folders
    assert win.plot_pane is not None
    assert "Monitor" in win.plot_pane.view.toPlainText() or "类型" in win.plot_pane.view.toPlainText()

    obj = next(o for o in win.sim.objects if o.dict.get("PresentationName") == "Fluid Domain"
               and o.class_name == "star.common.Region")
    win.document.execute(SetPropertyCommand(obj.id, "PresentationName", "X", "Fluid Domain"))
    assert win.document.dirty
    assert win.actions["Edit>Undo"].isEnabled()
    win.cmd_undo()
    assert obj.dict.get("PresentationName") == "Fluid Domain"

    win.cmd_generate_mesh()
    blob = win.messages.view.toPlainText()
    assert "内核" in blob or "宏" in blob or "禁用" in blob

    win.cmd_mesh_diag()
    win.actions["Window>Cad"].setChecked(True)
    win.cmd_toggle_cad()
    assert win.tb_cad.isVisible()
    win.cmd_cad_repair()
    oid = win.cmd_create_derived()
    assert oid is not None
    assert win.document.object(oid) is not None
    win.document.mark_clean()
    win.close()


def test_gui_import_and_assign_region(app):
    from star_gui import StarMainWindow

    win = StarMainWindow()
    win.show()
    win.load_file(SIM)
    _wait_loaded(win, app)
    n0 = len(win.sim.objects)
    oid = win.import_surface_from_path("session_wall.stl")
    assert oid is not None
    assert len(win.sim.objects) >= n0
    part = win.document.object(oid)
    assert part is not None
    win.document.mark_clean()
    win.close()


def test_writer_roundtrip_color_opacity_keys_view():
    from sim_parser import SimFile
    from sim_writer import save_sim
    from star_gui_document import SimDocument
    from star_gui_commands import SetPropertyCommand

    tmp = tempfile.mkdtemp(prefix="star_f1_")
    try:
        dest = os.path.join(tmp, "wing.sim")
        sim = SimFile(SIM)
        disp = next(o for o in sim.objects if o.name == "Mesh 1"
                    and "Displayer" in (o.class_name or ""))
        doc = SimDocument(sim, SIM)
        old_color = list(disp.dict.get("DisplayerColor"))
        new_color = [0.1, 0.2, 0.3]
        doc.execute(SetPropertyCommand(disp.id, "DisplayerColor", new_color, old_color))
        doc.execute(SetPropertyCommand(disp.id, "Opacity", 0.42, disp.dict.get("Opacity")))
        pg = sim.objmap[disp.dict["Collector"]]
        keys = list(pg.dict.get("Keys") or [])
        doc.execute(SetPropertyCommand(pg.id, "Keys", keys[1:] + keys[:1], keys))
        scene = next(o for o in sim.objects if o.class_name == "star.vis.Scene")
        assert doc.persist_view(scene.id, {
            "position": (1.0, 2.0, 3.0),
            "focal": (4.0, 5.0, 6.0),
            "view_up": (0.0, 1.0, 0.0),
            "parallel_scale": 12.5,
        })
        save_sim(sim, dest, patches=doc.patches, src_path=SIM)
        re = SimFile(dest)
        d2 = re.objmap[disp.id]
        assert d2.dict.get("DisplayerColor") == pytest.approx(new_color)
        assert d2.dict.get("Opacity") == pytest.approx(0.42)
        assert re.objmap[pg.id].dict.get("Keys") == keys[1:] + keys[:1]
        view = re.objmap[re.objmap[scene.id].dict["CurrentView"]]
        pos = re.objmap[view.dict["Position"]].dict["Value"]
        assert pos == [1.0, 2.0, 3.0]
        assert view.dict.get("ParallelScale") == pytest.approx(12.5)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_writer_inserts_copied_object():
    from sim_parser import SimFile
    from sim_writer import save_sim
    from star_gui_commands import CopyObjectCommand
    from star_gui_document import SimDocument

    tmp = tempfile.mkdtemp(prefix="star_f2_")
    try:
        dest = os.path.join(tmp, "wing.sim")
        sim = SimFile(SIM)
        src = next(o for o in sim.objects if o.dict.get("PresentationName") == "Fluid Domain"
                   and o.class_name == "star.common.Region")
        doc = SimDocument(sim, SIM)
        cmd = CopyObjectCommand(src.id)
        assert doc.execute(cmd)
        clone_name = doc.object(cmd.new_id).name
        save_sim(sim, dest, patches=doc.patches, created=doc.created, src_path=SIM)
        re = SimFile(dest)
        names = [o.name for o in re.objects]
        assert clone_name in names
        assert re.objects[-1].class_name == "ClassVersions"
        assert any(o.class_name == "star.common.Region" and o.name == clone_name
                   for o in re.objects)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_visibility_and_show_only_undo():
    from sim_parser import SimFile
    from star_gui_commands import ShowOnlyCommand, VisibilityCommand
    from star_gui_document import SimDocument

    sim = SimFile(SIM)
    obj = next(o for o in sim.objects if o.dict.get("PresentationName") == "Fluid Domain"
               and o.class_name == "star.common.Region")
    other = next(o for o in sim.objects if o.id != obj.id and o.dict.get("PresentationName"))
    doc = SimDocument(sim, SIM)
    doc.execute(VisibilityCommand(obj.id, False))
    assert doc.is_visible(obj.id) is False
    assert doc.dirty
    doc.undo()
    assert doc.is_visible(obj.id) is True
    doc.execute(ShowOnlyCommand(obj.id, [obj.id, other.id]))
    assert doc.is_visible(obj.id) is True
    assert doc.is_visible(other.id) is False
    doc.undo()
    assert doc.is_visible(other.id) is True


def test_reload_hook_and_dirty_confirm(app):
    from star_gui import StarMainWindow
    from star_gui_commands import SetPropertyCommand

    win = StarMainWindow()
    win.show()
    win.load_file(SIM)
    _wait_loaded(win, app)
    obj = next(o for o in win.sim.objects if o.dict.get("PresentationName") == "Fluid Domain"
               and o.class_name == "star.common.Region")
    win.document.execute(SetPropertyCommand(obj.id, "PresentationName", "X", "Fluid Domain"))
    assert win.document.dirty
    assert win.confirm_discard_dirty("t", "t") is True
    assert callable(win.cmd_reload)
    win.document.mark_clean()
    win.close()

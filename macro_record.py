# -*- coding: utf-8 -*-
"""A1：Java 宏录制器 —— GUI 操作 → `.java`，并复用宏桥回放。

设计要点：
  - 录制收口挂在 `CommandBus.on_execute`（菜单/右键/属性改动的唯一入口），
    `MacroRecorder.attach(bus)` 即开始记录，`detach()` 停止；不改动既有语义。
  - 非结构性的仿真动作（网格/物理/求解/后处理）通过 `record_operation`
    直接登记，操作→Java 映射表在 `star_macro.OPERATIONS`（真实 STAR-CCM+ API）。
  - 结构性命令（属性/重命名/显隐）产出可编译的 ClientServerObject 调用；
    变换/删除/复制/仅显示等私有语义无公开等价 API，产出 `// [best-effort]`
    注释行（诚实降级，不编造 API）。
  - 回放复用 `star_macro.run_star_macro_stream`（工作副本周转 + 逐行回流）。
"""

import os
import shutil
import tempfile

from star_macro import (DEFAULT_MACRO_IMPORTS, OPERATIONS, _format_body,
                        render_macro, run_star_macro_stream, write_macro)
from semantic_dict import resolve_class

# ---------------------------------------------------------------------------
# 1. 命令 -> Java 映射登记表（7 个可撤销命令全覆盖）
#    mode: "verified"   = 产出可编译的 ClientServerObject 调用
#          "best-effort" = 私有语义无公开等价 API，产出注释（诚实降级）
# ---------------------------------------------------------------------------
COMMAND_JAVA_MAP = {
    "SetPropertyCommand": {"mode": "verified", "layer": "structural"},
    "RenameCommand": {"mode": "verified", "layer": "structural"},
    "VisibilityCommand": {"mode": "verified", "layer": "structural"},
    "ShowOnlyCommand": {"mode": "best-effort", "layer": "structural"},
    "TransformPartCommand": {"mode": "best-effort", "layer": "structural"},
    "DeleteObjectCommand": {"mode": "best-effort", "layer": "structural"},
    "CopyObjectCommand": {"mode": "best-effort", "layer": "structural"},
}

# 对象检索路径：ClassName 简单名后缀 -> (manager getter, item getter)
# 与 STAR-CCM+ 的 `sim.get<X>Manager().get<X>(name)` 惯例一致。
MANAGER_PATHS = {
    "Region": ("getRegionManager", "getRegion"),
    "Scene": ("getSceneManager", "getScene"),
    "Report": ("getReportManager", "getReport"),
    "Monitor": ("getMonitorManager", "getMonitor"),
    "Plot": ("getPlotManager", "getPlot"),
    "Table": ("getTableManager", "getTable"),
    "FieldFunction": ("getFieldFunctionManager", "getFieldFunction"),
    "Part": ("getPartManager", "getPart"),
    "Continuum": ("getContinuumManager", "getContinuum"),
    "CoordinateSystem": ("getCoordinateSystemManager", "getCoordinateSystem"),
}


def command_names():
    """录制器支持的命令类名（注册表键）。"""
    return sorted(COMMAND_JAVA_MAP)


def command_mode(cmd_or_name):
    name = cmd_or_name if isinstance(cmd_or_name, str) else cmd_or_name.__class__.__name__
    spec = COMMAND_JAVA_MAP.get(name)
    return spec["mode"] if spec else None


def _path_for(simple):
    for key, spec in MANAGER_PATHS.items():
        if simple.endswith(key):
            return spec
    return None


def _java_literal(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "new double[] {%s}" % ", ".join(_java_literal(v) for v in value)
    return '"%s"' % str(value).replace("\\", "\\\\").replace('"', '\\"')


def _cap(key):
    key = str(key)
    return key[:1].upper() + key[1:]


class MacroOp(object):
    """一条录制项：kind ∈ {"command", "operation"}。"""

    __slots__ = ("kind", "name", "args")

    def __init__(self, kind, name, **args):
        self.kind = kind
        self.name = name
        self.args = args

    def __repr__(self):
        return "<MacroOp %s %s %r>" % (self.kind, self.name, self.args)


def op_from_command(cmd, doc=None):
    """把一个 Command 转成 MacroOp；非注册命令返回 None。"""
    cname = cmd.__class__.__name__
    if cname not in COMMAND_JAVA_MAP:
        return None
    oid = getattr(cmd, "obj_id", None)
    if oid is None:
        oid = getattr(cmd, "part_id", None)
    cls, nm = "Object", ""
    if doc is not None and oid is not None:
        obj = doc.object(oid)
        if obj is not None:
            cls = obj.class_name or "Object"
            nm = obj.name or ""
    args = {"obj_id": oid, "class": cls, "target_name": nm}
    if cname == "RenameCommand":
        args.update({"key": "PresentationName", "value": cmd.new_value})
        name = "rename"
    elif cname == "SetPropertyCommand":
        args.update({"key": cmd.key, "value": cmd.new_value})
        name = "set_property"
    elif cname == "VisibilityCommand":
        args.update({"visible": cmd.visible})
        name = "set_visibility"
    elif cname == "ShowOnlyCommand":
        args.update({"visible": True, "all_ids": list(cmd.all_ids)})
        name = "show_only"
    elif cname == "TransformPartCommand":
        args.update({"translate": cmd.translate, "scale": cmd.scale})
        name = "transform"
    elif cname == "DeleteObjectCommand":
        name = "delete"
    elif cname == "CopyObjectCommand":
        name = "copy"
    else:  # pragma: no cover - 注册表与分支保持同步
        return None
    return MacroOp("command", name, **args)


def _retrieval_lines(simple, cls, name, var):
    """产出 `X var = sim.getXMgr().getX("name");` 及其 import。

    无检索路径或目标名为空时返回 (None, [])（由调用方诚实降级）。
    """
    if not name:
        return None, []
    spec = _path_for(simple)
    if spec is None:
        return None, []
    mgr, getter = spec
    imports = []
    if "." in cls:
        imports.append("import %s;" % cls)
    lines = ['%s %s = sim.%s().%s("%s");' % (simple, var, mgr, getter, name)]
    return imports, lines


def command_java(op, idx=0):
    """把一条 command 类型 MacroOp 渲染为 (imports, java_lines)。"""
    name = op.name
    args = op.args
    mode = COMMAND_JAVA_MAP.get(
        {"rename": "RenameCommand", "set_property": "SetPropertyCommand",
         "set_visibility": "VisibilityCommand", "show_only": "ShowOnlyCommand",
         "transform": "TransformPartCommand", "delete": "DeleteObjectCommand",
         "copy": "CopyObjectCommand"}.get(name, ""), {}).get("mode")
    cls = resolve_class(args.get("class") or "")
    simple = cls.rsplit(".", 1)[-1] if cls else "Object"
    target = args.get("target_name") or ""
    var = "obj%d" % idx

    if name == "transform":
        return [], ["// [best-effort] transform %s '%s' translate=%s scale=%s"
                    % (cls or simple, target, _java_literal(args.get("translate") or (0, 0, 0)),
                       _java_literal(args.get("scale") or (1, 1, 1)))]
    if name in ("delete", "copy", "show_only"):
        return [], ["// [best-effort] %s %s '%s'"
                    % (name, cls or simple, target)]

    imports, lines = _retrieval_lines(simple, cls, target, var)
    if imports is None:
        return [], ['// [unmapped command] %s on %s "%s"'
                    % (name, cls or simple, target)]
    if name == "rename":
        lines.append("%s.setPresentationName(%s);" % (var, _java_literal(args.get("value"))))
    elif name == "set_property":
        key = args.get("key") or "Value"
        if key in ("PresentationName", "name"):
            lines.append("%s.setPresentationName(%s);"
                         % (var, _java_literal(args.get("value"))))
        else:
            lines.append("%s.set%s(%s);" % (var, _cap(key), _java_literal(args.get("value"))))
    elif name == "set_visibility":
        lines.append("%s.setVisible(%s);" % (var, "true" if args.get("visible") else "false"))
    else:  # pragma: no cover
        return [], ["// [unmapped command] %s" % name]
    _ = mode
    return imports, lines


class MacroRecorder(object):
    """录制 GUI 命令 + 登记仿真操作，产出可回放 Java 宏。"""

    def __init__(self, class_name="recorded_macro", save_name="out.sim", do_save=True):
        self.class_name = class_name
        self.save_name = save_name
        self.do_save = do_save
        self.ops = []
        self.bus = None
        self.recording = False
        self._hook = None

    # -- 生命周期 ----------------------------------------------------------
    def start(self):
        self.recording = True
        return self

    def stop(self):
        self.recording = False
        return self

    def clear(self):
        self.ops = []
        return self

    def __len__(self):
        return len(self.ops)

    # -- 挂载 CommandBus ---------------------------------------------------
    def attach(self, bus):
        self.bus = bus
        self._hook = self._on_execute
        bus.on_execute = self._hook
        return self

    def detach(self):
        if self.bus is not None and getattr(self.bus, "on_execute", None) is self._hook:
            self.bus.on_execute = None
        self._hook = None
        self.bus = None
        return self

    def _on_execute(self, cmd, doc):
        if not self.recording:
            return
        op = op_from_command(cmd, doc)
        if op is not None:
            self.ops.append(op)

    # -- 显式登记 ----------------------------------------------------------
    def record_command(self, cmd, doc=None):
        op = op_from_command(cmd, doc)
        if op is not None:
            self.ops.append(op)
        return op

    def record_operation(self, name, **args):
        op = MacroOp("operation", name, **args)
        self.ops.append(op)
        return op

    def operations(self, category=None):
        names = []
        for op in self.ops:
            if op.kind != "operation":
                continue
            spec = OPERATIONS.get(op.name)
            if category is None or (spec and spec.get("category") == category):
                names.append(op.name)
        return names

    def categories(self):
        cats = set()
        for op in self.ops:
            if op.kind == "operation":
                spec = OPERATIONS.get(op.name)
                if spec:
                    cats.add(spec.get("category"))
        return cats

    # -- 渲染 / 落盘 / 回放 -------------------------------------------------
    def to_java(self):
        imports = list(DEFAULT_MACRO_IMPORTS)
        body = []
        idx = 0
        for op in self.ops:
            if op.kind == "operation":
                spec = OPERATIONS.get(op.name)
                if spec is None:
                    body.append("// [unmapped operation] %s %r" % (op.name, op.args))
                    continue
                for imp in spec.get("imports", ()):
                    if imp not in imports:
                        imports.append(imp)
                for tmpl in spec.get("imports_fmt", ()):
                    from star_macro import OPERATION_DEFAULTS
                    imp = tmpl % dict(OPERATION_DEFAULTS, **op.args)
                    if imp not in imports:
                        imports.append(imp)
                body.extend(_format_body(spec.get("body", ()), op.args))
            else:
                imps, lines = command_java(op, idx)
                for imp in imps:
                    if imp not in imports:
                        imports.append(imp)
                body.extend(lines)
                idx += 1
        if not body:
            body.append("// (no recorded operations)")
        return render_macro(self.class_name, body, self.save_name, imports, self.do_save)

    def save(self, dirpath, basename=None):
        if basename is None:
            basename = self.class_name + ".java"
        return write_macro(dirpath, basename, self.to_java())

    def replay(self, exe, src_sim, on_line=None, timeout=600):
        """写宏到临时目录并复用宏桥回放，返回 (workdir, returncode, lines)。"""
        d = tempfile.mkdtemp(prefix="star_rec_")
        try:
            macro = self.save(d)
            return run_star_macro_stream(exe, src_sim, macro, on_line, timeout)
        finally:
            shutil.rmtree(d, ignore_errors=True)


def make_recorder(bus=None, class_name="recorded_macro", save_name="out.sim",
                  do_save=True, autostart=True):
    """工厂：建录制器，可选挂载到 CommandBus 并立即开始录。"""
    rec = MacroRecorder(class_name, save_name, do_save)
    if bus is not None:
        rec.attach(bus)
    if autostart:
        rec.start()
    return rec

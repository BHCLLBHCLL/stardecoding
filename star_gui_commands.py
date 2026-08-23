# -*- coding: utf-8 -*-
"""命令总线：菜单 / 右键 / 属性改动都走 Command（可撤销）。"""

from collections import deque


class Command(object):
    """一次可撤销操作。"""

    name = "Command"

    def do(self, doc):
        raise NotImplementedError

    def undo(self, doc):
        raise NotImplementedError


class CommandBus(object):
    """Undo / Redo 栈。"""

    def __init__(self, maxlen=200):
        self._undo = deque()
        self._redo = deque()
        self.maxlen = maxlen
        self.on_change = None

    def execute(self, cmd, doc):
        result = cmd.do(doc)
        self._undo.append(cmd)
        while len(self._undo) > self.maxlen:
            self._undo.popleft()
        self._redo.clear()
        self._notify()
        return result

    def undo(self, doc):
        if not self._undo:
            return False
        cmd = self._undo.pop()
        cmd.undo(doc)
        self._redo.append(cmd)
        self._notify()
        return True

    def redo(self, doc):
        if not self._redo:
            return False
        cmd = self._redo.pop()
        cmd.do(doc)
        self._undo.append(cmd)
        self._notify()
        return True

    def can_undo(self):
        return bool(self._undo)

    def can_redo(self):
        return bool(self._redo)

    def peek_undo(self):
        return self._undo[-1].name if self._undo else ""

    def peek_redo(self):
        return self._redo[-1].name if self._redo else ""

    def clear(self):
        self._undo.clear()
        self._redo.clear()
        self._notify()

    def _notify(self):
        if self.on_change is not None:
            self.on_change()


class SetPropertyCommand(Command):
    """改 SimObject.dict 的一个字段。"""

    def __init__(self, obj_id, key, new_value, old_value=None):
        self.obj_id = obj_id
        self.key = key
        self.new_value = new_value
        self.old_value = old_value
        self.name = "Set %s" % key

    def do(self, doc):
        obj = doc.object(self.obj_id)
        if obj is None:
            return False
        if self.old_value is None and self.key in obj.dict:
            self.old_value = obj.dict.get(self.key)
        doc.set_property(self.obj_id, self.key, self.new_value)
        return True

    def undo(self, doc):
        doc.set_property(self.obj_id, self.key, self.old_value)
        return True


class RenameCommand(SetPropertyCommand):
    def __init__(self, obj_id, new_name, old_name=None, key="PresentationName"):
        SetPropertyCommand.__init__(self, obj_id, key, new_name, old_name)
        self.name = "Rename"


class TransformPartCommand(Command):
    """对部件顶点做平移/缩放（会话内，可撤销）。"""

    def __init__(self, part_id, translate=(0, 0, 0), scale=(1, 1, 1)):
        self.part_id = part_id
        self.translate = tuple(float(x) for x in translate)
        self.scale = tuple(float(x) for x in scale)
        self.name = "Transform Part"

    def do(self, doc):
        doc.transform_part(self.part_id, self.translate, self.scale)
        return True

    def undo(self, doc):
        # transform_part 独立累加平移与相乘缩放，逆操作为 -t 与 1/s。
        inv_s = tuple((1.0 / s if abs(s) > 1e-15 else 1.0) for s in self.scale)
        inv_t = tuple(-t for t in self.translate)
        doc.transform_part(self.part_id, inv_t, inv_s)
        return True


class VisibilityCommand(Command):
    def __init__(self, obj_id, visible):
        self.obj_id = obj_id
        self.visible = bool(visible)
        self.name = "Visibility"

    def do(self, doc):
        prev = doc.is_visible(self.obj_id)
        doc.set_visible(self.obj_id, self.visible)
        self._prev = prev
        return True

    def undo(self, doc):
        doc.set_visible(self.obj_id, getattr(self, "_prev", True))
        return True


class ShowOnlyCommand(Command):
    """仅显示一项：记录先前显隐，便于撤销。"""

    def __init__(self, obj_id, all_ids):
        self.obj_id = obj_id
        self.all_ids = list(all_ids)
        self.name = "Show Only"
        self._prev = {}

    def do(self, doc):
        self._prev = {oid: doc.is_visible(oid) for oid in self.all_ids}
        for oid in self.all_ids:
            doc.set_visible(oid, oid == self.obj_id)
        return True

    def undo(self, doc):
        for oid, vis in self._prev.items():
            doc.set_visible(oid, vis)
        return True


class DeleteObjectCommand(Command):
    """会话内删除：从管理器 Keys 摘掉，对象仍留在图里以便撤销。"""

    def __init__(self, obj_id):
        self.obj_id = obj_id
        self.name = "Delete"
        self._removed_from = []

    def do(self, doc):
        self._removed_from = doc.unlink_object(self.obj_id)
        doc.mark_deleted(self.obj_id)
        return True

    def undo(self, doc):
        doc.unmark_deleted(self.obj_id)
        for mid, idx in self._removed_from:
            doc.relink_object(mid, self.obj_id, idx)
        return True


class CopyObjectCommand(Command):
    def __init__(self, obj_id):
        self.obj_id = obj_id
        self.name = "Copy"
        self.new_id = None

    def do(self, doc):
        self.new_id = doc.duplicate_object(self.obj_id)
        return self.new_id is not None

    def undo(self, doc):
        if self.new_id is not None:
            doc.unlink_object(self.new_id)
            doc.mark_deleted(self.new_id)
        return True

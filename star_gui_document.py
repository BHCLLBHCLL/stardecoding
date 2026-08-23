# -*- coding: utf-8 -*-
"""SimDocument：.sim 会话模型（脏标记、补丁、显隐、选择）。"""

import copy

from star_gui_commands import CommandBus


class SimDocument(object):
    """包一层 SimFile：属性补丁 + 会话覆盖，不直接改磁盘。"""

    def __init__(self, sim=None, path=None):
        self.sim = sim
        self.path = path or (sim.path if sim is not None else None)
        self.bus = CommandBus()
        self.dirty = False
        self.patches = {}          # obj_id -> {key: value}
        self.visibility = {}       # obj_id -> bool
        self.deleted = set()
        self.created = {}          # obj_id -> SimObject-like dict
        self._next_id = 0
        self.transforms = {}       # part_id -> (tx,ty,tz,sx,sy,sz) cumulative
        self.clipboard = None
        self.selection = None      # obj_id
        self.selection_history = []
        self.selection_index = -1
        self.saved_views = {}      # name -> camera dict
        self.listeners = []
        if sim is not None:
            ids = [o.id for o in sim.objects] or [2]
            self._next_id = max(ids) + 1

    def bind(self, sim, path=None):
        self.sim = sim
        self.path = path or (sim.path if sim is not None else None)
        self.bus.clear()
        self.dirty = False
        self.patches.clear()
        self.visibility.clear()
        self.deleted.clear()
        self.created.clear()
        self.transforms.clear()
        self.clipboard = None
        self.selection = None
        self.selection_history = []
        self.selection_index = -1
        self.saved_views.clear()
        if sim is not None:
            ids = [o.id for o in sim.objects] or [2]
            self._next_id = max(ids) + 1
        self._notify("bound")

    def add_listener(self, fn):
        self.listeners.append(fn)

    def _notify(self, kind, **kw):
        for fn in list(self.listeners):
            try:
                fn(kind, **kw)
            except Exception:
                pass

    def object(self, oid):
        if self.sim is None or oid is None:
            return None
        if oid in self.deleted:
            return None
        if oid in self.created:
            return self.created[oid]
        return self.sim.objmap.get(oid)

    def set_property(self, oid, key, value):
        obj = self.object(oid)
        if obj is None:
            return False
        bag = self.patches.setdefault(oid, {})
        bag[key] = value
        obj.dict[key] = value
        if key in ("PresentationName", "name"):
            obj.dict[key] = value
        self.dirty = True
        self._notify("property", obj_id=oid, key=key, value=value)
        return True

    def is_visible(self, oid):
        return self.visibility.get(oid, True)

    def set_visible(self, oid, visible):
        self.visibility[oid] = bool(visible)
        self.dirty = True
        self._notify("visibility", obj_id=oid, visible=bool(visible))

    def mark_deleted(self, oid):
        self.deleted.add(oid)
        self.dirty = True
        self._notify("deleted", obj_id=oid)

    def unmark_deleted(self, oid):
        self.deleted.discard(oid)
        self.dirty = True
        self._notify("undeleted", obj_id=oid)

    def unlink_object(self, oid):
        """从所有 Keys 列表摘掉 oid，返回 [(manager_id, index), ...]。"""
        removed = []
        if self.sim is None:
            return removed
        for o in self.sim.objects:
            keys = o.dict.get("Keys")
            if not isinstance(keys, list) or oid not in keys:
                continue
            idx = keys.index(oid)
            keys.pop(idx)
            self.patches.setdefault(o.id, {})["Keys"] = list(keys)
            removed.append((o.id, idx))
        self.dirty = True
        return removed

    def relink_object(self, manager_id, oid, idx):
        mgr = self.object(manager_id)
        if mgr is None:
            return
        keys = list(mgr.dict.get("Keys") or [])
        if oid not in keys:
            idx = max(0, min(idx, len(keys)))
            keys.insert(idx, oid)
            mgr.dict["Keys"] = keys
            self.patches.setdefault(manager_id, {})["Keys"] = keys
        self.dirty = True

    def duplicate_object(self, oid):
        src = self.object(oid)
        if src is None or self.sim is None:
            return None
        new_id = self._next_id
        self._next_id += 1
        d = copy.deepcopy(src.dict)
        name = d.get("PresentationName") or d.get("name") or "Copy"
        if "PresentationName" in d:
            d["PresentationName"] = "%s Copy" % name
        elif "name" in d:
            d["name"] = "%s Copy" % name
        from sim_parser import SimObject
        clone = SimObject(new_id, d, -1)
        self.created[new_id] = clone
        self.sim.objmap[new_id] = clone
        self.sim.objects.append(clone)
        parent = src.dict.get("Parent")
        if isinstance(parent, int):
            self.relink_object(parent, new_id, 10 ** 9)
        self.dirty = True
        self._notify("created", obj_id=new_id, src_id=oid)
        return new_id

    def transform_part(self, part_id, translate, scale):
        prev = self.transforms.get(part_id, (0.0, 0.0, 0.0, 1.0, 1.0, 1.0))
        nxt = (prev[0] + translate[0], prev[1] + translate[1], prev[2] + translate[2],
               prev[3] * scale[0], prev[4] * scale[1], prev[5] * scale[2])
        self.transforms[part_id] = nxt
        self.dirty = True
        self._notify("transform", obj_id=part_id, translate=translate, scale=scale)
        return nxt

    def push_selection(self, oid):
        if self.selection_index < len(self.selection_history) - 1:
            self.selection_history = self.selection_history[: self.selection_index + 1]
        self.selection_history.append(oid)
        self.selection_index = len(self.selection_history) - 1
        self.selection = oid

    def select_prev(self):
        if self.selection_index <= 0:
            return None
        self.selection_index -= 1
        self.selection = self.selection_history[self.selection_index]
        return self.selection

    def select_next(self):
        if self.selection_index >= len(self.selection_history) - 1:
            return None
        self.selection_index += 1
        self.selection = self.selection_history[self.selection_index]
        return self.selection

    def execute(self, cmd):
        return self.bus.execute(cmd, self)

    def undo(self):
        return self.bus.undo(self)

    def redo(self):
        return self.bus.redo(self)

    def create_session_object(self, class_name, name, extra=None, parent=None):
        """会话级新对象（无 file offset，Save 时不写盘）。"""
        from sim_parser import SimObject
        new_id = self._next_id
        self._next_id += 1
        d = {"ClassName": class_name, "PresentationName": name}
        if extra:
            d.update(extra)
        if parent is not None:
            d.setdefault("Parent", parent)
        obj = SimObject(new_id, d, -1)
        self.created[new_id] = obj
        if self.sim is not None:
            self.sim.objmap[new_id] = obj
            self.sim.objects.append(obj)
            if parent is not None:
                self.relink_object(parent, new_id, 10 ** 9)
        self.dirty = True
        self._notify("created", obj_id=new_id)
        return new_id

    def mark_clean(self):
        self.dirty = False
        self._notify("clean")

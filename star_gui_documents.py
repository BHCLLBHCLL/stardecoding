# -*- coding: utf-8 -*-
"""X2：多仿真文档工作区 + 跨仿真对象图复制粘贴（id 冲突重映射）。

纯逻辑（无 Qt 依赖）：
- DocumentWorkspace：登记所有已打开的 SimDocument（可跨主窗口共享）。
- 跨仿真复制粘贴：把源文档的对象子树快照成 Clip，粘贴进目标文档时分配
  全新会话 id 并用 sim_writer.remap_value 递归重写 Keys/Parent 等整型引用，
  避免与目标文档既有对象 id 冲突；落盘时由 sim_writer.created_id_mapping
  把会话 id 映射到图序号 id，保证写出的对象图引用自洽。
"""

import copy
import os


class Clip(object):
    """一次复制的对象图快照（旧会话 id → 深拷贝 dict）。"""

    def __init__(self, root_id, objects, suffix=" Copy", source_path=None,
                 parent_class=None):
        self.root_id = root_id
        self.objects = objects
        self.suffix = suffix
        self.source_path = source_path
        self.parent_class = parent_class

    def __len__(self):
        return len(self.objects)

    def ids(self):
        return sorted(self.objects)

    def has(self, oid):
        return oid in self.objects


def child_ids(doc, oid):
    """按 Keys 取直接子对象 id（保持文件顺序，去重）。"""
    obj = doc.object(oid)
    if obj is None:
        return []
    keys = (obj.dict or {}).get("Keys")
    if not isinstance(keys, list):
        return []
    out = []
    for k in keys:
        if isinstance(k, int) and doc.object(k) is not None and k not in out:
            out.append(k)
    return out


def collect_subtree(doc, oid):
    """广度优先收集 oid 及其后代 id（根在前）。"""
    if doc is None or doc.object(oid) is None:
        return []
    seen = []
    queue = [oid]
    while queue:
        cur = queue.pop(0)
        if cur in seen:
            continue
        seen.append(cur)
        queue.extend(c for c in child_ids(doc, cur) if c not in seen)
    return seen


def copy_subtree(doc, oid, suffix=" Copy"):
    """快照 oid 子树为 Clip（不改源文档）；根对象按 suffix 改名。"""
    ids = collect_subtree(doc, oid)
    if not ids:
        return None
    objects = {}
    for i in ids:
        obj = doc.object(i)
        objects[i] = copy.deepcopy(obj.dict or {})
    root = objects[oid]
    if suffix:
        if "PresentationName" in root:
            root["PresentationName"] = "%s%s" % (root.get("PresentationName") or "", suffix)
        elif "name" in root:
            root["name"] = "%s%s" % (root.get("name") or "", suffix)
    src_parent = root.get("Parent")
    parent_class = None
    if isinstance(src_parent, int):
        pobj = doc.object(src_parent)
        if pobj is not None:
            parent_class = (pobj.dict or {}).get("ClassName")
    return Clip(oid, objects, suffix, getattr(doc, "path", None), parent_class)


def analogous_parent_id(doc, class_name):
    """在目标文档中按类名寻找源父对象的类比节点（跨仿真粘贴挂靠用）。

    返回目标文档中第一个同 ClassName 的对象 id；找不到返回 None。
    """
    if not class_name or doc is None or doc.sim is None:
        return None
    for o in doc.sim.objects:
        if (o.dict or {}).get("ClassName") == class_name:
            return o.id
    return None


def plan_id_mapping(clip, doc):
    """为目标文档分配不冲突的会话 id（从 doc._next_id 起，按旧 id 升序）。"""
    mapping = {}
    nxt = getattr(doc, "_next_id", 1)
    for old in clip.ids():
        mapping[old] = nxt
        nxt += 1
    return mapping


def paste_clip(doc, clip, parent=None):
    """把 Clip 粘贴进目标文档，返回新根 id（失败 None）。

    步骤：分配新会话 id → 用 remap_value 重写剪贴内整型引用 → 登记进
    doc.created / sim.objmap / sim.objects → 归属父（显式 parent 优先；
    否则原父同在剪贴内则挂其新 id；否则删掉指向源文档的悬空 Parent）。
    """
    if clip is None or doc is None or doc.sim is None or len(clip) == 0:
        return None
    from sim_parser import SimObject
    from sim_writer import remap_value
    mapping = plan_id_mapping(clip, doc)
    if not mapping:
        return None
    src_root = clip.objects[clip.root_id]
    for old in clip.ids():
        new_id = mapping[old]
        d = remap_value(copy.deepcopy(clip.objects[old]), mapping)
        obj = SimObject(new_id, d, -1)
        doc.created[new_id] = obj
        doc.sim.objmap[new_id] = obj
        doc.sim.objects.append(obj)
    doc._next_id = mapping[clip.ids()[-1]] + 1
    new_root = mapping[clip.root_id]
    new_root_obj = doc.created[new_root]
    src_parent = src_root.get("Parent")
    target_parent = parent
    if target_parent is None and isinstance(src_parent, int) and src_parent in mapping:
        target_parent = mapping[src_parent]
        # remap_value 已把剪贴内 Parent 改写成 target_parent
    if isinstance(target_parent, int):
        new_root_obj.dict["Parent"] = target_parent
        doc.relink_object(target_parent, new_root, 10 ** 9)
    elif isinstance(src_parent, int) and src_parent not in mapping:
        new_root_obj.dict.pop("Parent", None)
    doc.dirty = True
    doc._notify("created", obj_id=new_root, src_id=clip.root_id, pasted=True)
    return new_root


def paste_into(target_doc, source_doc, oid, parent=None, suffix=" Copy"):
    """复制源文档 oid 子树并粘贴进目标文档，返回新根 id。"""
    return paste_clip(target_doc, copy_subtree(source_doc, oid, suffix), parent)


class Clipboard(object):
    """跨文档剪贴板：记录来源文档，用于判定是否跨仿真。"""

    def __init__(self):
        self.clip = None
        self.source = None

    def set(self, source_doc, clip):
        self.clip = clip
        self.source = source_doc

    def clear(self):
        self.clip = None
        self.source = None

    def has_content(self):
        return self.clip is not None

    def is_cross(self, target_doc):
        return self.clip is not None and self.source is not target_doc


CLIPBOARD = Clipboard()


class DocumentWorkspace(object):
    """多仿真文档工作区：登记所有已打开文档（可跨主窗口共享）。"""

    def __init__(self):
        self._docs = []
        self._owners = {}
        self._active = None

    def add(self, doc, owner=None, activate=True):
        if doc is None:
            return -1
        if doc not in self._docs:
            self._docs.append(doc)
        if owner is not None:
            self._owners[doc] = owner
        if activate:
            self._active = doc
        return self._docs.index(doc)

    def remove(self, doc):
        if doc in self._docs:
            self._docs.remove(doc)
            self._owners.pop(doc, None)
            if self._active is doc:
                self._active = self._docs[-1] if self._docs else None
            return True
        return False

    def activate(self, doc):
        if doc in self._docs:
            self._active = doc
            return True
        return False

    def owner_of(self, doc):
        return self._owners.get(doc)

    @property
    def active(self):
        return self._active

    @property
    def active_index(self):
        return self._docs.index(self._active) if self._active in self._docs else -1

    @property
    def documents(self):
        return list(self._docs)

    def paths(self):
        return [d.path for d in self._docs]

    def find_by_path(self, path):
        if not path:
            return None
        want = os.path.normcase(os.path.abspath(path))
        for d in self._docs:
            if d.path and os.path.normcase(os.path.abspath(d.path)) == want:
                return d
        return None

    def close_all(self):
        docs = list(self._docs)
        self._docs = []
        self._owners = {}
        self._active = None
        return docs

    def __len__(self):
        return len(self._docs)

    def __iter__(self):
        return iter(list(self._docs))

    def __contains__(self, doc):
        return doc in self._docs


WORKSPACE = DocumentWorkspace()

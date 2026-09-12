# -*- coding: utf-8 -*-
"""star_gui_help.py — 帮助/文档/许可 文本后端（纯逻辑，无 Qt）。

数据源:
  - doc_javadoc_catalog.md（574 个 star.* 包语义目录，来自官方 Javadoc）
  - semantic_dict.py（ClassName → 语义层 / 现名别名）

供 Help 菜单（帮助内容 / 文档目录 / 许可信息）与「关于」对话框复用；
按选中对象 ClassName 给出语义层 + 包作用 + 类条目，实现上下文相关帮助。
"""
import os
import re

import semantic_dict as sd

APP_NAME = "STAR-CCM+ .sim Viewer / Editor"
APP_VERSION = "1.0.0"

_ROOT = os.path.dirname(os.path.abspath(__file__))
CATALOG_PATH = os.path.join(_ROOT, "doc_javadoc_catalog.md")

_OVERVIEW_RE = re.compile(r"^\|\s*([A-Za-z][A-Za-z0-9_.]*)\s*\|\s*(.+?)\s*\|\s*$")
_SECTION_RE = re.compile(r"^###\s+(\d+)\.\s+(.+?)\s*$")
_STAR_TOKEN_RE = re.compile(r"star\.[a-z0-9]+(?:\.[a-z0-9]+)*")
_BULLET_PKG_RE = re.compile(r"^-\s+(star\.[a-z0-9.]+)")

_CATALOG = None


def _read_catalog():
    with open(CATALOG_PATH, encoding="utf-8") as f:
        return f.read()


def _parse_catalog():
    packages = {}
    sections = []
    title = ""
    bullets = []
    in_overview = False
    for raw in _read_catalog().splitlines():
        line = raw.rstrip()
        if line.startswith("## "):
            in_overview = "包总览" in line
            continue
        m = _SECTION_RE.match(line)
        if m:
            if title:
                sections.append({"index": int(m.group(1)) - 1, "title": title,
                                 "packages": [], "bullets": bullets})
            title = m.group(2)
            bullets = []
            continue
        if line.startswith("- ") and title:
            bullets.append(line[2:].strip())
            continue
        if in_overview:
            m = _OVERVIEW_RE.match(line)
            if m and m.group(1) not in ("包",):
                packages[m.group(1)] = m.group(2)
    if title:
        sections.append({"index": 0, "title": title, "packages": [], "bullets": bullets})
    for idx, sec in enumerate(sections):
        sec["index"] = idx
        found = _STAR_TOKEN_RE.findall(sec["title"])
        for b in sec["bullets"]:
            bm = _BULLET_PKG_RE.match("- " + b) or _BULLET_PKG_RE.match(b)
            if bm:
                found.append(bm.group(1))
            else:
                head = b.split("：", 1)[0]
                if head.startswith("star."):
                    found.append(head.split()[0])
        seen = []
        for p in found:
            if p not in seen:
                seen.append(p)
        sec["packages"] = seen
    return {"packages": packages, "sections": sections}


def catalog():
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = _parse_catalog()
    return _CATALOG


def packages():
    return catalog()["packages"]


def sections():
    return catalog()["sections"]


def stats():
    return {"packages": len(packages()), "sections": len(sections())}


def package_of(class_name):
    if not class_name:
        return ""
    if not class_name.startswith("star."):
        return class_name
    parts = class_name.split(".")
    pkgs = packages()
    for i in range(len(parts) - 1, 1, -1):
        pkg = ".".join(parts[:i])
        if pkg in pkgs:
            return pkg
    for i in range(len(parts) - 1, 1, -1):
        pkg = ".".join(parts[:i])
        if pkg in sd.PACKAGE_LAYERS:
            return pkg
    return ".".join(parts[:2])


def semantic_layer(class_name):
    key = sd.layer_of(class_name)
    return key, sd.LAYER_CN.get(key, key)


def package_description(pkg):
    if not pkg:
        return ""
    pkgs = packages()
    if pkg in pkgs:
        return pkgs[pkg]
    for known in sorted(pkgs, key=len, reverse=True):
        if pkg.startswith(known + ".") or known.startswith(pkg + "."):
            return pkgs[known]
    return ""


def package_section(pkg):
    if not pkg:
        return None
    for sec in sections():
        for p in sec["packages"]:
            if p == pkg or pkg.startswith(p + ".") or p.startswith(pkg + "."):
                return sec
    return None


def class_bullets(class_name, limit=6):
    short = class_name.rsplit(".", 1)[-1]
    out = []
    if len(short) < 3:
        return out
    for sec in sections():
        for b in sec["bullets"]:
            if short in b:
                out.append((sec["title"], b))
                if len(out) >= limit:
                    return out
    return out


def explain(class_name):
    resolved = sd.resolve_class(class_name) if class_name else ""
    layer_key, layer_cn = semantic_layer(resolved)
    pkg = package_of(resolved)
    sec = package_section(pkg)
    return {
        "class_name": class_name or "",
        "resolved": resolved,
        "package": pkg,
        "layer": layer_key,
        "layer_cn": layer_cn,
        "package_description": package_description(pkg),
        "section_title": sec["title"] if sec else "",
        "bullets": class_bullets(resolved),
    }


def class_help_text(class_name):
    if not class_name:
        return "未选择对象。"
    info = explain(class_name)
    lines = ["类：%s" % info["class_name"]]
    if info["resolved"] and info["resolved"] != info["class_name"]:
        lines.append("现名：%s（旧版本类名）" % info["resolved"])
    lines.append("包：%s" % (info["package"] or "—"))
    if info["package_description"]:
        lines.append("作用：%s" % info["package_description"])
    lines.append("语义层：%s（%s）" % (info["layer_cn"], info["layer"]))
    if info["section_title"]:
        lines.append("目录章节：%s" % info["section_title"])
    if info["bullets"]:
        lines.append("")
        lines.append("相关条目：")
        for _, b in info["bullets"]:
            lines.append("  · %s" % b)
    return "\n".join(lines)


def search(query, limit=20):
    q = (query or "").strip().lower()
    if not q:
        return []
    hits = []
    for pkg, desc in packages().items():
        if q in pkg.lower() or q in desc.lower():
            hits.append({"kind": "package", "name": pkg, "text": desc})
            if len(hits) >= limit:
                return hits
    for sec in sections():
        for b in sec["bullets"]:
            if q in b.lower():
                hits.append({"kind": "class", "name": sec["title"], "text": b})
                if len(hits) >= limit:
                    return hits
    return hits


def help_contents_text():
    st = stats()
    lines = ["%s 帮助内容" % APP_NAME,
             "对象图语义目录：%d 个 star.* 包 / %d 个章节" % (st["packages"], st["sections"]),
             "",
             "一、包总览"]
    for pkg, desc in packages().items():
        lines.append("  %-28s %s" % (pkg, desc))
    lines.append("")
    lines.append("二、章节详解")
    for sec in sections():
        lines.append("")
        lines.append("%s." % (sec["index"] + 1) + sec["title"])
        for b in sec["bullets"]:
            lines.append("  · %s" % b)
    return "\n".join(lines)


def documentation_text():
    st = stats()
    lines = ["文档目录",
             "",
             "语义目录文件：%s" % CATALOG_PATH,
             "来源：STAR-CCM+ 官方 Java API Javadoc（%d 个 star.* 包）" % st["packages"],
             "",
             "本客户端相关文档：",
             "  · doc_javadoc_catalog.md — 包/类语义目录（帮助内容数据源）",
             "  · star_gui_parity.md — GUI 能力对标表（官方菜单 ↔ 本实现）",
             "  · parity_100pct_plan.md — 100%% 对标分波计划与验收",
             "",
             "章节："]
    for sec in sections():
        lines.append("  %2d. %s" % (sec["index"] + 1, sec["title"]))
    return "\n".join(lines)


def licensing_text():
    return (
        "%s %s\n\n"
        "本程序为独立实现的 .sim 文件查看 / 编辑器（数据层 sim_parser.py、"
        "回写 sim_writer.py、界面 PyQt5 + VTK），不含 STAR-CCM+ 求解内核，"
        "不隶属于、亦未获 Dassault Systèmes / Siemens 授权或背书。\n\n"
        "STAR-CCM+ / Simcenter 为其各自权利人的商标，此处仅用于描述文件格式兼容性。\n\n"
        "第三方组件许可：\n"
        "  · PyQt5 — GPL v3 / 商业双许可（分发前需遵循相应条款）\n"
        "  · VTK — BSD 3-Clause\n"
        "  · NumPy / SciPy — BSD 3-Clause\n"
        "  · Gmsh（可选）— GPL\n\n"
        "本程序按「现状」提供，不附带任何明示或默示担保。"
        % (APP_NAME, APP_VERSION)
    )


def about_text():
    return (
        "%s %s\nPyQt5 + VTK\n"
        "查看 + 编辑几何/网格操作/场景/属性；求解运算禁用。\n"
        "数据层: sim_parser.py · 回写: sim_writer.py\n"
        "帮助数据: doc_javadoc_catalog.md（%d 个包）\n"
        "对标: star_gui_parity.md"
        % (APP_NAME, APP_VERSION, stats()["packages"])
    )

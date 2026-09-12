# -*- coding: utf-8 -*-
"""A3：Design Manager 式参数研究 —— DOE 扫描 / 响应表 / 并行批次。

路线 A（自研）：
  - `DesignParameter`：一个设计参数（显式取值表，或 lo/hi 连续区间 + levels 离散化）。
  - DOE 采样器：`full_factorial` / `ofat` / `latin_hypercube` / `random_sampling`。
  - `DesignTable`：设计表（case_id + 各参数列的矩阵）。
  - `ResponseTable`：响应表（设计列 + 响应列 + 统计 + 最优 + CSV 落地）。
  - `DesignStudy`：门面 —— 参数/响应登记、参数→对象属性绑定（对接 A2 star.* 对象）、
    `generate()` 出设计表、`run(evaluator, workers=N)` 并行批次跑算例并回收响应。

与 STAR-CCM+ Design Manager 的对应：
  design set（参数）→ design table（DOE 行）→ run（批次）→ response table（响应）。
"""

import csv
import itertools
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

DOE_METHODS = ("full_factorial", "ofat", "latin_hypercube", "random_sampling")


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


class DesignParameter(object):
    """一个设计参数：显式取值表，或 [lo, hi] 连续区间按 levels 离散化。"""

    def __init__(self, name, values=None, lo=None, hi=None, levels=3, kind=None):
        self.name = name
        if values is not None:
            self.values = list(values)
            self.lo = self.hi = None
            self._continuous = False
        else:
            if lo is None or hi is None:
                raise ValueError("参数 %r 需要 values，或 lo/hi 区间" % name)
            self.lo, self.hi = float(lo), float(hi)
            lv = max(2, int(levels))
            step = (self.hi - self.lo) / (lv - 1)
            self.values = [self.lo + step * i for i in range(lv)]
            self._continuous = True
        if not self.values:
            raise ValueError("参数 %r 取值表为空" % name)
        self.levels = len(self.values)
        self.kind = kind or self._infer_kind()

    def _infer_kind(self):
        if self._continuous:
            return "float"
        if all(isinstance(v, str) for v in self.values):
            return "choice"
        if all(isinstance(v, int) and not isinstance(v, bool) for v in self.values):
            return "int"
        return "float"

    def is_continuous(self):
        return self._continuous

    @property
    def grid(self):
        """离散取值表（full_factorial / ofat 用）。"""
        return list(self.values)

    def sample(self, rng):
        """连续区间均匀采样；离散参数随机取一档。"""
        if self._continuous:
            return self.lo + (self.hi - self.lo) * rng.random()
        return rng.choice(self.values)

    def __repr__(self):
        if self._continuous:
            return "DesignParameter(%r, lo=%g, hi=%g, levels=%d)" % (
                self.name, self.lo, self.hi, self.levels)
        return "DesignParameter(%r, values=%r)" % (self.name, self.values)


# ---------------------------------------------------------------------------
# DOE 采样器：返回 [{参数名: 值}, ...]
# ---------------------------------------------------------------------------
def full_factorial(parameters, base=None):
    """全因子：各参数取值表做笛卡尔积。"""
    names = [p.name for p in parameters]
    grids = [p.grid for p in parameters]
    if not parameters:
        return []
    return [dict(zip(names, combo)) for combo in itertools.product(*grids)]


def ofat(parameters, base=None):
    """一次一因子：基准算例 + 逐个参数扫其所有档位。"""
    if not parameters:
        return []
    base = dict(base) if base else {}
    center = {p.name: p.grid[len(p.grid) // 2] for p in parameters}
    center.update(base)
    rows = [dict(center)]
    seen = {tuple(sorted(center.items()))}
    for p in parameters:
        for v in p.grid:
            row = dict(center)
            row[p.name] = v
            key = tuple(sorted(row.items()))
            if key not in seen:
                seen.add(key)
                rows.append(row)
    return rows


def latin_hypercube(parameters, n, seed=None):
    """拉丁超立方：每个参数分层后随机置换，保证一维投影均匀。"""
    n = int(n)
    if n <= 0:
        return []
    rng = random.Random(seed)
    cols = {}
    for p in parameters:
        vals = []
        if p.is_continuous():
            perm = list(range(n))
            rng.shuffle(perm)
            for i in range(n):
                frac = (perm[i] + rng.random()) / n
                vals.append(p.lo + (p.hi - p.lo) * frac)
        else:
            while len(vals) < n:
                order = list(p.values)
                rng.shuffle(order)
                vals.extend(order)
            vals = vals[:n]
        cols[p.name] = vals
    return [{p.name: cols[p.name][i] for p in parameters} for i in range(n)]


def random_sampling(parameters, n, seed=None):
    """蒙特卡洛：连续参数均匀、离散参数等概率。"""
    rng = random.Random(seed)
    n = int(n)
    return [{p.name: p.sample(rng) for p in parameters} for _ in range(n)]


_DOE = {
    "full_factorial": lambda ps, n, seed, base: full_factorial(ps, base),
    "ofat": lambda ps, n, seed, base: ofat(ps, base),
    "latin_hypercube": lambda ps, n, seed, base: latin_hypercube(ps, n, seed),
    "random_sampling": lambda ps, n, seed, base: random_sampling(ps, n, seed),
}


def generate_design(parameters, method="full_factorial", n=0, seed=None, base=None):
    """按方法生成设计行；method ∈ DOE_METHODS。"""
    if method not in _DOE:
        raise ValueError("未知 DOE 方法 %r（可选：%s）" % (method, ", ".join(DOE_METHODS)))
    return _DOE[method](parameters, n, seed, base)


# ---------------------------------------------------------------------------
# 设计表 / 响应表
# ---------------------------------------------------------------------------
def _write_rows(path, rows, columns):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in columns})
    return path


class DesignTable(object):
    """设计表：每行一个算例（case_id + 各参数取值）。"""

    def __init__(self, rows, parameter_names):
        self.parameter_names = list(parameter_names)
        self.rows = []
        for i, r in enumerate(rows):
            row = {"case_id": i}
            row.update(r)
            self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]

    def case(self, idx):
        return self.rows[idx]

    def parameters(self, name):
        return [r.get(name) for r in self.rows]

    def to_dicts(self):
        return [dict(r) for r in self.rows]

    def to_csv(self, path):
        return _write_rows(path, self.rows, ["case_id"] + self.parameter_names)


class ResponseTable(object):
    """响应表：设计列 + 响应列，附统计、最优与 CSV 落地。"""

    def __init__(self, parameter_names, response_names):
        self.parameter_names = list(parameter_names)
        self.response_names = list(response_names)
        self.rows = []
        self.errors = {}

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]

    def add_row(self, case, responses=None):
        responses = responses or {}
        row = {"case_id": case.get("case_id", len(self.rows))}
        for n in self.parameter_names:
            row[n] = case.get(n)
        for n in self.response_names:
            row[n] = responses.get(n)
        self.rows.append(row)
        return row

    @property
    def columns(self):
        return ["case_id"] + self.parameter_names + self.response_names

    def response(self, name):
        return [r.get(name) for r in self.rows]

    def parameter(self, name):
        return [r.get(name) for r in self.rows]

    def statistics(self, name):
        vals = [r.get(name) for r in self.rows if _is_number(r.get(name))]
        if not vals:
            return {"n": 0, "min": None, "max": None, "mean": None, "std": None}
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        return {"n": n, "min": min(vals), "max": max(vals),
                "mean": mean, "std": var ** 0.5}

    def summary(self):
        return {n: self.statistics(n) for n in self.response_names}

    def best(self, name, maximize=True):
        cand = [r for r in self.rows if _is_number(r.get(name))]
        if not cand:
            return None
        key = lambda r: r[name]
        return (max if maximize else min)(cand, key=key)

    def to_csv(self, path):
        ordered = sorted(self.rows, key=lambda r: r.get("case_id", 0))
        return _write_rows(path, ordered, self.columns)


# ---------------------------------------------------------------------------
# 门面：DesignStudy
# ---------------------------------------------------------------------------
class DesignStudy(object):
    """参数研究门面：登记 → 生成 DOE → 绑定对象属性 → 批次跑 → 收响应。"""

    def __init__(self, parameters=None, responses=None, name="Design Study"):
        self.name = name
        self.parameters = list(parameters or [])
        self.responses = list(responses or [])
        self.bindings = []          # [(参数名, obj_id, key)]
        self.table = None
        self.responses_table = None

    # -- 登记 --------------------------------------------------------------
    def add_parameter(self, *args, **kwargs):
        if args and isinstance(args[0], DesignParameter):
            p = args[0]
        else:
            p = DesignParameter(*args, **kwargs)
        self.parameters.append(p)
        return p

    def add_response(self, name):
        if name not in self.responses:
            self.responses.append(name)
        return name

    def bind(self, param_name, obj_id, key):
        """把某参数绑定到对象属性（obj_id, key）——对应 Design Manager 的链接。"""
        self.bindings.append((param_name, obj_id, key))
        return self

    def bind_object(self, param_name, simulation, key, name=None, oid=None):
        """按 A2 对象（名/类名/id）解析后绑定。"""
        if oid is None:
            obj = simulation.get(name) if name is not None else None
            if obj is None:
                raise ValueError("找不到绑定目标 %r" % name)
            oid = obj.id
        return self.bind(param_name, oid, key)

    @property
    def parameter_names(self):
        return [p.name for p in self.parameters]

    # -- DOE ---------------------------------------------------------------
    def generate(self, method="full_factorial", n=0, seed=None, base=None):
        rows = generate_design(self.parameters, method=method, n=n, seed=seed, base=base)
        self.table = DesignTable(rows, self.parameter_names)
        return self.table

    # -- 绑定应用 ----------------------------------------------------------
    def apply_case(self, case, simulation):
        """把算例参数写到绑定的对象属性上，返回已应用列表。"""
        applied = []
        for pname, oid, key in self.bindings:
            if pname not in case:
                continue
            obj = simulation.get_object(oid)
            if obj is None:
                continue
            obj.set(key, case[pname])
            applied.append((pname, oid, key, case[pname]))
        return applied

    # -- 批次跑 ------------------------------------------------------------
    def run(self, evaluator, workers=1, on_result=None, simulation=None,
            apply=True):
        """逐算例调用 evaluator(case)->{响应名:值}，workers>1 时并发。

        任一算例抛错不中断批次：错误记入 ResponseTable.errors[case_id]。
        """
        if self.table is None:
            self.generate()
        rt = ResponseTable(self.parameter_names, self.responses)

        def work(case):
            if simulation is not None and apply:
                self.apply_case(case, simulation)
            return evaluator(case) or {}

        if workers and int(workers) > 1:
            with ThreadPoolExecutor(max_workers=int(workers)) as ex:
                futs = {ex.submit(work, c): c for c in self.table}
                for fut in as_completed(futs):
                    case = futs[fut]
                    try:
                        res = fut.result()
                    except Exception as exc:
                        res = {}
                        rt.errors[case["case_id"]] = repr(exc)
                    rt.add_row(case, res)
                    if on_result is not None:
                        on_result(case, res)
        else:
            for case in self.table:
                try:
                    res = work(case)
                except Exception as exc:
                    res = {}
                    rt.errors[case["case_id"]] = repr(exc)
                rt.add_row(case, res)
                if on_result is not None:
                    on_result(case, res)

        rt.rows.sort(key=lambda r: r.get("case_id", 0))
        self.responses_table = rt
        return rt

    def run_serial(self, evaluator, on_result=None, simulation=None, apply=True):
        return self.run(evaluator, workers=1, on_result=on_result,
                        simulation=simulation, apply=apply)

    def __len__(self):
        return len(self.table) if self.table is not None else 0

    def __repr__(self):
        return "<DesignStudy %r params=%d responses=%d cases=%d>" % (
            self.name, len(self.parameters), len(self.responses), len(self))


def make_design_study(parameters=None, responses=None, name="Design Study",
                      method=None, n=0, seed=None):
    """工厂：建 study；给 method 则顺手生成设计表。"""
    study = DesignStudy(parameters=parameters, responses=responses, name=name)
    if method is not None:
        study.generate(method=method, n=n, seed=seed)
    return study

# -*- coding: utf-8 -*-
"""A3：Design Manager 式参数研究 —— DOE 采样 / 设计表 / 响应表 / 并行批次。"""
import os
import shutil
import tempfile

import pytest

import design_study as ds


# -- DesignParameter -------------------------------------------------------
def test_design_parameter_explicit_interval_and_kinds():
    p = ds.DesignParameter("alpha", values=[1, 2, 3])
    assert p.kind == "int"
    assert p.grid == [1, 2, 3]
    assert not p.is_continuous()

    c = ds.DesignParameter("scheme", values=["upwind", "central"])
    assert c.kind == "choice"

    f = ds.DesignParameter("angle", lo=0.0, hi=10.0, levels=3)
    assert f.is_continuous()
    assert f.kind == "float"
    assert f.values == [0.0, 5.0, 10.0]

    with pytest.raises(ValueError):
        ds.DesignParameter("empty", values=[])
    with pytest.raises(ValueError):
        ds.DesignParameter("nobounds")


def test_design_parameter_sample_within_bounds():
    import random
    rng = random.Random(1)
    f = ds.DesignParameter("angle", lo=-2.0, hi=2.0, levels=5)
    for _ in range(50):
        assert -2.0 <= f.sample(rng) <= 2.0
    d = ds.DesignParameter("alpha", values=[1, 2, 3])
    for _ in range(20):
        assert d.sample(rng) in [1, 2, 3]


# -- DOE 采样器 ------------------------------------------------------------
def test_full_factorial_cartesian_product():
    ps = [ds.DesignParameter("a", values=[1, 2, 3]),
          ds.DesignParameter("b", values=["x", "y"])]
    rows = ds.full_factorial(ps)
    assert len(rows) == 6
    assert {"a": 1, "b": "x"} in rows
    assert len(ds.full_factorial([])) == 0


def test_ofat_center_plus_one_at_a_time():
    ps = [ds.DesignParameter("a", values=[1, 2, 3]),
          ds.DesignParameter("b", values=["x", "y"])]
    rows = ds.ofat(ps)
    assert rows[0] == {"a": 2, "b": "y"}
    assert len(rows) == 4


def test_latin_hypercube_deterministic_and_stratified():
    ps = [ds.DesignParameter("angle", lo=0.0, hi=1.0, levels=4),
          ds.DesignParameter("scheme", values=["a", "b"])]
    r1 = ds.latin_hypercube(ps, 4, seed=7)
    r2 = ds.latin_hypercube(ps, 4, seed=7)
    assert r1 == r2
    assert len(r1) == 4
    for row in r1:
        assert 0.0 <= row["angle"] < 1.0
        assert row["scheme"] in ["a", "b"]
    assert ds.latin_hypercube(ps, 0) == []


def test_random_sampling_deterministic():
    ps = [ds.DesignParameter("angle", lo=0.0, hi=4.0, levels=4)]
    a = ds.random_sampling(ps, 5, seed=3)
    b = ds.random_sampling(ps, 5, seed=3)
    assert a == b and len(a) == 5
    assert all(0.0 <= r["angle"] <= 4.0 for r in a)


def test_generate_design_dispatch_and_unknown():
    ps = [ds.DesignParameter("a", values=[1, 2])]
    assert len(ds.generate_design(ps, "full_factorial")) == 2
    assert len(ds.generate_design(ps, "ofat")) == 2
    assert len(ds.generate_design(ps, "latin_hypercube", n=3)) == 3
    assert len(ds.generate_design(ps, "random_sampling", n=3)) == 3
    with pytest.raises(ValueError):
        ds.generate_design(ps, "bogus")


# -- DesignTable -----------------------------------------------------------
def test_design_table_case_ids_and_csv():
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    table = ds.DesignTable(rows, ["a", "b"])
    assert len(table) == 2
    assert table.case(0)["case_id"] == 0
    assert table.case(1)["case_id"] == 1
    assert table[1]["a"] == 2
    assert table.parameters("a") == [1, 2]
    assert [r["case_id"] for r in table] == [0, 1]

    tmp = tempfile.mkdtemp(prefix="star_doe_")
    try:
        path = table.to_csv(os.path.join(tmp, "design.csv"))
        text = open(path, encoding="utf-8").read()
        assert "case_id,a,b" in text
        assert "0,1,x" in text
        assert "1,2,y" in text
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# -- ResponseTable ---------------------------------------------------------
def test_response_table_stats_best_and_csv():
    rt = ds.ResponseTable(["a"], ["drag", "lift"])
    rt.add_row({"case_id": 0, "a": 1}, {"drag": 2.0, "lift": 10.0})
    rt.add_row({"case_id": 1, "a": 2}, {"drag": 4.0, "lift": 5.0})
    assert len(rt) == 2
    assert rt.columns == ["case_id", "a", "drag", "lift"]
    assert rt.response("drag") == [2.0, 4.0]

    st = rt.statistics("drag")
    assert st["n"] == 2 and st["min"] == 2.0 and st["max"] == 4.0
    assert st["mean"] == 3.0
    assert abs(st["std"] - 1.0) < 1e-12
    assert rt.statistics("lift")["mean"] == 7.5

    assert rt.best("drag", maximize=True)["case_id"] == 1
    assert rt.best("drag", maximize=False)["case_id"] == 0
    assert rt.best("drag")["drag"] == 4.0
    assert rt.summary()["drag"]["mean"] == 3.0

    empty = ds.ResponseTable(["a"], ["y"])
    assert empty.statistics("y")["n"] == 0
    assert empty.best("y") is None

    tmp = tempfile.mkdtemp(prefix="star_doe_")
    try:
        path = rt.to_csv(os.path.join(tmp, "resp.csv"))
        text = open(path, encoding="utf-8").read()
        assert "case_id,a,drag,lift" in text
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# -- DesignStudy -----------------------------------------------------------
def test_design_study_add_bind_generate_apply():
    study = ds.DesignStudy(name="Wing Sweep")
    study.add_parameter("angle", values=[1.0, 2.0, 3.0])
    study.add_parameter(ds.DesignParameter("speed", values=[10, 20]))
    study.add_response("drag")
    assert study.parameter_names == ["angle", "speed"]

    class _Obj(object):
        def __init__(self, oid):
            self.id = oid
            self.d = {}

        def set(self, k, v):
            self.d[k] = v

    class _Sim(object):
        def __init__(self):
            self.o = _Obj(192)

        def get_object(self, oid):
            return self.o if oid == 192 else None

        def get(self, name):
            return self.o if name == "Fluid Domain" else None

    sim = _Sim()
    study.bind("angle", 192, "SomeAngle")
    study.bind_object("speed", sim, "SomeSpeed", name="Fluid Domain")

    table = study.generate("full_factorial")
    assert len(table) == 6
    case = table.case(0)
    applied = study.apply_case(case, sim)
    assert (applied[0][0], applied[0][3]) == ("angle", case["angle"])
    assert sim.o.d["SomeAngle"] == case["angle"]
    assert sim.o.d["SomeSpeed"] == case["speed"]


def test_design_study_run_serial_and_callbacks():
    study = ds.make_design_study(
        [ds.DesignParameter("a", values=[1, 2, 3])], ["y"],
        method="full_factorial")
    seen = []
    rt = study.run_serial(lambda case: {"y": case["a"] * 10},
                          on_result=lambda c, r: seen.append((c["case_id"], r["y"])))
    assert len(rt) == 3
    assert rt.response("y") == [10, 20, 30]
    assert seen == [(0, 10), (1, 20), (2, 30)]
    assert rt.errors == {}


def test_design_study_run_parallel_and_error_isolation():
    study = ds.DesignStudy()
    study.add_parameter("a", values=[1, 2, 3, 4])
    study.add_response("y")
    study.generate("full_factorial")

    calls = []

    def evaluator(case):
        calls.append(case["case_id"])
        if case["a"] == 2:
            raise RuntimeError("boom")
        return {"y": case["a"] * 2}

    rt = study.run(evaluator, workers=4)
    assert len(rt) == 4
    assert rt.errors == {1: "RuntimeError('boom')"}
    assert rt.response("y") == [2, None, 6, 8]
    assert sorted(calls) == [0, 1, 2, 3]
    assert [r["case_id"] for r in rt] == [0, 1, 2, 3]


def test_make_design_study_factory_repr_and_lazy_generate():
    study = ds.make_design_study(name="Lazy")
    assert study.table is None and len(study) == 0
    assert "DesignStudy" in repr(study)
    study.add_parameter("a", values=[1, 2])
    study.add_response("y")
    rt = study.run(lambda case: {"y": case["a"]})
    assert len(rt) == 2

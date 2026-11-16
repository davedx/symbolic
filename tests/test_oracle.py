"""Oracle tests: hand-checked answers on a mini world, cross-check against an independent
forward-chaining reasoner, proof validity, and independently computed hop counts."""

import pytest

from world.closure import Closure
from world.oracle import Oracle, Undefined
from world.proofcheck import check_proof
from world.queries import build_pool, make_query


# ----------------------------------------------------------------- hand-checked cases
@pytest.mark.parametrize(
    "form,args,answer,task,hops,rules",
    [
        # inheritance and exceptions
        ("prop", ("E1", "P0"), True, "inherited", 3, ["inherit", "isa", "sub"]),
        ("prop", ("E0", "P0"), False, "exception", 2, ["isa", "override"]),  # penguins don't fly
        ("prop", ("E3", "P0"), True, "exception", 1, ["override"]),  # ...except robo
        ("prop", ("E2", "P1"), True, "inherited", 3, ["inherit", "isa", "sub"]),
        ("prop", ("E0", "P2"), "V0", "inherited", 4, ["inherit", "isa", "sub"]),
        ("prop", ("E2", "P2"), "V1", "exception", 1, ["override"]),
        ("prop", ("E1", "P3"), True, "direct", 1, ["stated"]),
        # taxonomy membership
        ("isa", ("E0", "C2"), True, "direct", 1, ["isa"]),
        ("isa", ("E0", "C5"), False, "direct", 1, ["disjoint", "isa"]),
        ("isa", ("E0", "C1"), True, "inherited", 2, ["isa", "sub"]),
        ("isa", ("E0", "C4"), False, "inherited", 2, ["disjoint", "isa", "sub"]),
        ("isa", ("E2", "C0"), True, "inherited", 3, ["isa", "sub"]),
        # relation composition
        ("chain", ("E0", ("R0",)), "E1", "relation", 1, ["rel"]),
        ("chain", ("E0", ("R0", "R0", "R0")), "E3", "relation", 3, ["rel"]),
        ("chain", ("E1", ("R1",)), "E0", "relation", 1, ["symmetric"]),
        ("chain", ("E2", ("R1",)), "E3", "relation", 1, ["symmetric"]),
        ("chain", ("E3", ("R1",)), "E2", "relation", 1, ["rel"]),
        ("chain", ("E1", ("R2",)), "E3", "relation", 1, ["inverse"]),
        ("chain", ("E1", ("R3",)), "E3", "relation", 1, ["rel"]),
        ("chain", ("E0", ("R3",)), "E2", "relation", 1, ["inverse"]),
        ("chain", ("E2", ("R1", "R2", "R0")), "E2", "relation", 3, ["inverse", "rel", "symmetric"]),
        # transitive order
        ("trans", ("E0", "R4", "E1"), True, "relation", 1, ["trans"]),
        ("trans", ("E0", "R4", "E2"), True, "relation", 2, ["trans"]),
        ("trans", ("E2", "R4", "E0"), False, "relation", 2, ["asymmetric", "trans"]),
        ("trans", ("E3", "R4", "E2"), True, "relation", 1, ["trans"]),
    ],
)
def test_mini_world(mini, form, args, answer, task, hops, rules):
    q = make_query(Oracle(mini), form, args)
    assert q.answer == answer
    assert q.task_type == task
    assert q.hops == hops
    assert q.rules == rules
    check_proof(mini, q)
    assert Closure(mini).answer(form, args) == answer


def test_mini_overridden_facts(mini):
    o = Oracle(mini)
    assert o.prop("E0", "P0").overridden == ("prop", "C1", "P0", True)
    assert o.prop("E3", "P0").overridden == ("prop", "C2", "P0", False)
    assert o.prop("E1", "P0").overridden is None


@pytest.mark.parametrize(
    "form,args",
    [
        ("prop", ("E2", "P0")),  # flies is not defined for mammals
        ("prop", ("E0", "P3")),  # individual property not stated for pingu
        ("trans", ("E0", "R4", "E3")),  # incomparable under the order
        ("chain", ("E0", ("R4",))),  # transitive relations are not functional
        ("trans", ("E0", "R0", "E1")),  # R0 is not transitive
    ],
)
def test_mini_undefined(mini, form, args):
    with pytest.raises(Undefined):
        Oracle(mini).answer(form, args)


def test_mini_closure_semantics(mini):
    c = Closure(mini)
    assert ("E0", "E2") in c.edges["R4"] and ("E2", "E0") not in c.edges["R4"]
    assert ("E2", "E3") in c.edges["R1"] and ("E3", "E2") in c.edges["R1"]
    assert ("E3", "E1") in c.edges["R2"] and ("E1", "E3") in c.edges["R3"]
    assert c.values["E3"]["P0"] is True and c.values["E0"]["P0"] is False


# ------------------------------------------------------------- generated-world checks
@pytest.fixture
def pool_and_closure(ds):
    return ds, Closure(ds.world)


def test_oracle_agrees_with_closure(pool_and_closure):
    ds, closure = pool_and_closure
    for q in ds.queries:
        assert closure.answer(q.form, q.args) == q.answer, q.qid


def test_oracle_agrees_with_closure_on_full_pool(tiny):
    """Also check the pre-split pool (including dropped queries) on the tiny world."""
    closure = Closure(tiny.world)
    for q in build_pool(tiny.world):
        assert closure.answer(q.form, q.args) == q.answer, q.qid


def test_all_proofs_valid(ds):
    for q in ds.queries:
        check_proof(ds.world, q)


def test_closure_relations_functional(ds):
    """Symmetric / inverse closure keeps all chain relations functional and total."""
    c = Closure(ds.world)
    n = len(ds.world.entities)
    for r in ds.world.chain_relations:
        succ = c.succ[r]
        assert all(len(v) == 1 for v in succ.values())
        assert len(succ) >= n - 1  # a symmetric matching leaves one entity out when n is odd


def test_exceptions_override_inherited_defaults(ds):
    """For every override statement, every entity in its scope that has no more specific
    statement gets the override value, and an entity-level query reports the exception."""
    w = ds.world
    o = Oracle(w)
    assign = w.assignments
    checked = 0
    for (node, p), v in assign.items():
        if node in w.entity_leaf:
            if w.properties[p]["kind"] == "individual":
                continue
            a = o.prop(node, p)
            assert a.answer == v and a.task_type == "exception" and a.hops == 1
            checked += 1
            continue
        parents = w.cat_ancestors(node)[1:]
        above = [assign[(n, p)] for n in parents if (n, p) in assign]
        if not above:
            continue  # a top attachment, not an override
        for e in w.entities_under(node):
            path = [e] + w.entity_categories(e)
            first = next(n for n in path if (n, p) in assign)
            if first == node:
                a = o.prop(e, p)
                assert a.answer == v and a.task_type == "exception"
                assert a.overridden[3] == above[0]
                checked += 1
    assert checked > 0


def test_hop_counts_independent(ds):
    """Recompute hop counts from world structure without using proofs."""
    w = ds.world
    tdepth = {}
    for r in w.relations_of_kind("transitive"):
        parent = {f[2]: f[3] for f in w.facts if f[0] == "rel" and f[1] == r}

        def depth(x):
            d = 0
            while x in parent:
                x, d = parent[x], d + 1
            return d

        tdepth[r] = {e: depth(e) for e in w.entities}

    for q in ds.queries:
        if q.form == "prop":
            e, p = q.args
            if (e, p) in w.assignments:
                expected = 1
            else:
                leaf = w.entity_leaf[e]
                node = next(c for c in w.cat_ancestors(leaf) if (c, p) in w.assignments)
                expected = 2 + w.cat_level[leaf] - w.cat_level[node]
        elif q.form == "isa":
            e, c = q.args
            expected = 1 + w.cat_level[w.entity_leaf[e]] - w.cat_level[c]
        elif q.form == "chain":
            expected = len(q.args[1])
        else:
            a, r, b = q.args
            expected = abs(tdepth[r][a] - tdepth[r][b])
        assert q.hops == expected, q.qid
        assert q.hops == len(q.proof)


def test_task_types_consistent(ds):
    w = ds.world
    for q in ds.queries:
        if q.form in ("chain", "trans"):
            assert q.task_type == "relation"
        elif q.form == "isa":
            assert q.task_type == ("direct" if q.hops == 1 else "inherited")
        elif w.properties[q.args[1]]["kind"] == "individual":
            assert q.task_type == "direct" and q.hops == 1
        else:
            assert q.task_type in ("inherited", "exception")
            assert (q.task_type == "exception") == (q.overridden is not None)

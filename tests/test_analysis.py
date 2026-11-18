"""train/analysis.py: Fisher exact test and the seed-level report."""

import pytest

from train.analysis import condition, fisher, report


def test_fisher_known_values():
    # R: fisher.test(matrix(c(5,0,0,5),2))$p.value = 0.007937; c(3,0,0,3) -> 0.1
    assert fisher((5, 5), (0, 5)) == pytest.approx(0.0079365, rel=1e-4)
    assert fisher((3, 3), (0, 3)) == pytest.approx(0.1, rel=1e-6)
    assert fisher((2, 4), (2, 4)) == pytest.approx(1.0)
    assert fisher((5, 6), (1, 6)) == pytest.approx(fisher((1, 6), (5, 6)))


def test_condition_strips_seed():
    assert condition("triples_context_M_150M_v2_s3") == "triples_context_M_150M_v2"
    assert condition("symbolic_context_L2d560_150M_v2_s12") == "symbolic_context_L2d560_150M_v2"


def test_report_on_smoke_runs(tmp_path):
    from tests.test_train import smoke_cfg
    from train.trainer import Trainer

    for seed in (0, 1):
        cfg = smoke_cfg(data__regime="context", data__rename="nodes", train__seed=seed)
        Trainer(cfg, runs_dir=tmp_path).train(verbose=False)
    summary = report([str(tmp_path / "*")])
    (cond, s), = summary.items()
    assert s["n"] == 2 and set(s) >= {"isa", "trans", "chain", "longest_chain"}
    assert all(0 <= k <= n == 2 for k, n in (s["isa"], s["trans"], s["chain"]))

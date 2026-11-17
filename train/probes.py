"""Diagnostic probes on a trained run (Phase 4 groundwork).

    uv run python -m train.probes context runs/<run>   # does the answer depend on the context?
    uv run python -m train.probes edit    runs/<run>   # do KB-format edits reach query answers?
    uv run python -m train.probes lens    runs/<run>   # where (layer, position) is the answer decodable?

context  (facts-in-context runs) On 1-hop entity-valued chains, replace the answer entity
         everywhere in the context with a fresh entity. A model that reads the context
         follows the edit. On true transitive queries, delete the proof edges from the
         context; a model that reads the context should stop answering "true".
         PASS: among chains answered correctly with the original context, >= 90% follow
         the edit (train split).
edit     (facts-in-weights runs) Change the object of 200 functional relation facts whose
         1-hop query is a train query. Fine-tune on the edited KB (KB format only), with
         rehearsal of the other train queries so query answering is kept. Report whether
         KB cloze and the *edited* facts' queries move to the new object.
         ``--direction query`` does the reverse: edit in query format, read the KB cloze.
lens     Logit lens (final norm + tied unembedding applied to each layer's residual) for
         1-hop functional chains: the object's top-1 rate at the KB-cloze position, at
         the query's subject position, and at the query's <|ans|> position.

Results are written to <run>/probe_<name>.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from encoders.base import BOS, EOS
from encoders.dataset import ContextSampler, Example, _variant, kb_documents, query_example, rename_example

from .config import RunConfig
from .evaluate import Evaluator, kb_recall
from .memguard import Slot, Watchdog
from .trainer import Trainer


def load(run_dir: Path) -> Trainer:
    cfg = RunConfig.from_yaml(str(run_dir / "config.yaml"), [])
    t = Trainer(cfg, runs_dir=str(run_dir.parent))
    if not t.load_checkpoint():
        raise SystemExit(f"no checkpoint in {run_dir}")
    return t


class _Fixed:
    """A context 'sampler' that returns a given context."""

    def __init__(self, ctx):
        self.ctx = ctx

    def sample(self, *a, **k):
        return self.ctx


def _argmax_next(model, ids) -> int:
    return int(mx.argmax(model(mx.array([ids]))[0, -1]).item())


def _name(w, x) -> str:
    return w.lexicon[x]


# ----------------------------------------------------------------------------- context
def probe_context(t: Trainer, n: int = 300, seed: int = 0) -> dict:
    enc, w, model = t.enc, t.ds.world, t.model
    dc = dataclasses.replace(t.cfg.data, seed=0, distractor_ramp_epochs=0)
    if dc.regime != "context":
        raise SystemExit("context probe needs a facts-in-context run")
    plain = dataclasses.replace(dc, rename="none")
    sampler = ContextSampler(w)
    rng = np.random.default_rng(seed)

    def example(q, ctx):
        """The eval example for q, with its (renamed, if configured) context, plus the
        renamed query; built with a fixed context so it can be edited."""
        if dc.rename != "none":
            q, ctx = rename_example(w, q, ctx, dc, 0)
        return q, ctx

    def predict(q, ctx):
        ex = query_example(enc, q, plain, 0, ctx_sampler=_Fixed(ctx))
        ids = enc.vocab.encode(ex.tokens)[: ex.prompt_len]
        return enc.vocab.decode([_argmax_next(model, ids)])[0]

    out = {}
    for split in ("train", "test_iid"):
        qs = [q for q in t.ds.queries if q.split == split and q.form == "chain" and q.hops == 1][:n]
        c = dict(n=0, correct=0, follow=0, stick=0, follow_given_correct=0)
        for q in qs:
            q2, ctx = example(q, sampler.sample(q, dc, 0))
            gold = q2.answer
            mentioned = {x for f in ctx for x in f}
            new = gold
            while new == gold or new in mentioned:
                new = w.entities[int(rng.integers(len(w.entities)))]
            sub = lambda f: tuple(new if x == gold else x for x in f)
            p0 = predict(q2, ctx)
            p1 = predict(dataclasses.replace(q2, answer=new), [sub(f) for f in ctx])
            ok = p0 == _name(w, gold)
            c["n"] += 1
            c["correct"] += ok
            c["follow"] += p1 == _name(w, new)
            c["stick"] += p1 == _name(w, gold)
            c["follow_given_correct"] += ok and p1 == _name(w, new)
        out[f"chain1/{split}"] = {
            "n": c["n"],
            "acc": c["correct"] / max(c["n"], 1),
            "follows_edit": c["follow"] / max(c["n"], 1),
            "keeps_old_answer": c["stick"] / max(c["n"], 1),
            "follows_edit_given_correct": c["follow_given_correct"] / max(c["correct"], 1),
        }
    for split in ("train", "test_iid", "test_length"):
        qs = [q for q in t.ds.queries if q.split == split and q.form == "trans" and q.answer is True][:n]
        full = dropped = 0
        for q in qs:
            needed = set(sampler.needed(q))
            ctx = sampler.sample(q, dc, 0)
            kept = [f for f in ctx if f not in needed]
            q2, ctx2 = example(q, ctx)
            # the reduced context gets its own (equally valid) renaming
            q3, kept2 = (rename_example(w, q, kept, dc, 0) if dc.rename != "none" else (q, kept))
            full += predict(q2, ctx2) == enc.answer(True)[0]
            dropped += predict(q3, kept2) == enc.answer(True)[0]
        out[f"trans_true/{split}"] = {
            "n": len(qs),
            "says_true_full_context": full / max(len(qs), 1),
            "says_true_proof_edges_removed": dropped / max(len(qs), 1),
        }
    tr = out["chain1/train"]
    out["pass"] = tr["acc"] > 0 and tr["follows_edit_given_correct"] >= 0.9
    return out


# -------------------------------------------------------------------------------- edit
def probe_edit(t: Trainer, direction: str = "kb", n_edit: int = 200, epochs: int = 10, lr: float = 1e-4) -> dict:
    enc, w, model, cfg = t.enc, t.ds.world, t.model, t.cfg
    if cfg.data.regime != "weights":
        raise SystemExit("edit probe needs a facts-in-weights run")
    rng = np.random.default_rng(0)
    train1 = {(q.args[0], q.args[1][0]): q for q in t.ds.queries if q.split == "train" and q.form == "chain" and q.hops == 1}
    cand = [
        f for f in w.facts
        if f[0] == "rel" and w.relations[f[1]]["kind"] == "functional" and (f[2], f[1]) in train1
    ]
    pick = [cand[i] for i in rng.choice(len(cand), min(n_edit, len(cand)), replace=False)]
    picked = set(pick)
    edit = {}
    for f in pick:
        o2 = f[3]
        while o2 == f[3]:
            o2 = w.entities[int(rng.integers(len(w.entities)))]
        edit[f] = (f[0], f[1], f[2], o2)
    facts2 = [edit.get(f, f) for f in w.facts]
    edited_qs = [train1[(f[2], f[1])] for f in pick]
    ctrl_qs = [train1[(f[2], f[1])] for f in cand if f not in picked][:n_edit]
    excl = {q.qid for q in edited_qs}
    keep = [q for q in t.ds.queries if q.split == "train" and q.qid not in excl]
    ev = Evaluator(enc, cfg.data)

    def measure() -> dict:
        p = ev.predict(model, edited_qs)
        return {
            "query_new": float(np.mean([r["pred"][0] == _name(w, edit[f][3]) for r, f in zip(p, pick)])),
            "query_old": float(np.mean([r["pred"][0] == _name(w, f[3]) for r, f in zip(p, pick)])),
            "kb_new": kb_recall(model, enc, [edit[f] for f in pick])["all"]["acc"],
            "kb_old": kb_recall(model, enc, pick)["all"]["acc"],
            "control_queries": float(np.mean([r["correct"] for r in ev.predict(model, ctrl_qs)])),
        }

    def kb_docs(epoch):
        order = np.random.default_rng(epoch).permutation(len(facts2))
        docs = []
        for s in range(0, len(order), cfg.data.kb_facts_per_doc):
            toks = [BOS]
            for i in order[s : s + cfg.data.kb_facts_per_doc]:
                toks += enc.fact(facts2[i], _variant(cfg.data, repr(facts2[i]), epoch))
            toks.append(EOS)
            docs.append(Example(toks, [0] + [1] * (len(toks) - 1), prompt_len=1, n_facts=16))
        return docs

    opt = optim.Adam(learning_rate=lr)

    def loss_fn(m, x, y, wt):
        ce = nn.losses.cross_entropy(m(x), y, reduction="none")
        return (ce * wt).sum() / mx.maximum(wt.sum(), 1.0)

    lg = nn.value_and_grad(model, loss_fn)
    watchdog = Watchdog()
    hist = {"before": measure()}
    for ep in range(1, epochs + 1):
        r = np.random.default_rng(ep)
        rehearse = [query_example(enc, keep[i], cfg.data, ep) for i in r.choice(len(keep), min(3000, len(keep)), replace=False)]
        if direction == "kb":
            docs = kb_docs(ep) + rehearse
        else:
            # edited answers in query format, rehearsing the *original* KB: does the
            # query-format edit reach the KB cloze?
            edited = [dataclasses.replace(q, answer=edit[f][3]) for q, f in zip(edited_qs, pick)]
            docs = [query_example(enc, q, cfg.data, ep) for q in edited] * 3 + rehearse + kb_documents(enc, cfg.data, ep)
        docs = [docs[i] for i in r.permutation(len(docs))]
        for s in range(0, len(docs), 32):
            b = t.stream.collate(docs[s : s + 32])
            _, g = lg(model, b.inputs, b.targets, b.weights)
            opt.update(model, g)
            mx.eval(model.parameters(), opt.state)
            watchdog.check()
        if ep in (1, 3, epochs):
            hist[f"epoch{ep}"] = measure()
            print(f"  epoch {ep}: {hist[f'epoch{ep}']}", flush=True)
    return {"direction": direction, "n_edit": len(pick), "lr": lr, "history": hist}


# -------------------------------------------------------------------------------- lens
def probe_lens(t: Trainer, n: int = 200) -> dict:
    enc, w, model = t.enc, t.ds.world, t.model
    if t.cfg.data.regime != "weights":
        raise SystemExit("lens probe is set up for facts-in-weights runs")
    ev = Evaluator(enc, t.cfg.data)

    def lens(ids, positions, target):
        _, hs = model(mx.array([ids]), return_hidden=True)
        W = model.embed.weight.T
        return np.array([[int(mx.argmax(model.ln_f(h[0, p]) @ W).item()) == target for p in positions] for h in hs])

    out = {}
    for split in ("train", "test_iid", "test_heldout_combo"):
        qs = [q for q in t.ds.queries if q.split == split and q.form == "chain" and q.hops == 1][:n]
        aq, ak = [], []
        for q in qs:
            f = q.proof[0][1]
            if f[0] != "rel" or w.relations[f[1]]["kind"] != "functional" or f[3] != q.answer:
                continue
            tgt = enc.vocab.index[_name(w, q.answer)]
            ids, _ = ev._prompt(q)
            toks = enc.vocab.decode(ids)
            subj = max(i for i, x in enumerate(toks) if x == _name(w, f[2]))
            aq.append(lens(ids, [subj, len(ids) - 1], tgt))
            kb = [BOS] + enc.fact(f, 0)
            last = max(i for i, x in enumerate(kb) if x == _name(w, q.answer))
            kids = enc.vocab.encode(kb[:last])
            ak.append(lens(kids, [len(kids) - 1], tgt))
        if not aq:
            continue
        aq, ak = np.mean(aq, 0), np.mean(ak, 0)
        out[split] = {
            "n": len(qs),
            "kb_cloze": ak[:, 0].round(3).tolist(),
            "query_at_subject": aq[:, 0].round(3).tolist(),
            "query_at_answer": aq[:, 1].round(3).tolist(),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", choices=["context", "edit", "lens"])
    ap.add_argument("run_dir")
    ap.add_argument("--direction", default="kb", choices=["kb", "query"])
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    slot = Slot()
    slot.acquire(f"probe {args.probe} {run_dir.name}")
    t = load(run_dir)
    if args.probe == "context":
        res = probe_context(t)
    elif args.probe == "edit":
        res = probe_edit(t, args.direction, epochs=args.epochs)
    else:
        res = probe_lens(t)
    name = f"probe_{args.probe}" + (f"_{args.direction}" if args.probe == "edit" else "")
    (run_dir / f"{name}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

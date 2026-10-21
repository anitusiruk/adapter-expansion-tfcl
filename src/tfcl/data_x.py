"""Extension text stream (declared in notes/tmlr_extension_prereg.md).

tweet_conc  NATURAL concept drift: three binary TweetEval tasks over the same tweet
            population (A offensive, B hate, C irony), schedule A B C A. The label ids
            {0, 1} are shared, but their meaning (the annotation concept) changes, so
            P(y | x) changes while P(x) changes little. Train splits only: per segment
            occurrence 160 examples per label; dev = 64 held-out training examples per
            label per task.
"""
import numpy as np
from datasets import load_dataset

from src.tfcl.data import _batches

TWEET_TASKS = [("A", "offensive"), ("B", "hate"), ("C", "irony")]


def tweet_conc(kind, seed, bs=16):
    rng = np.random.default_rng(seed + 9000)
    dev_rng = np.random.default_rng(424242)
    sched = ["A", "B", "C", "A"]
    occ = {n: sched.count(n) for n, _ in TWEET_TASKS}
    picks, dev = {}, {}
    for name, cfg in TWEET_TASKS:
        ds = load_dataset("cardiffnlp/tweet_eval", cfg, split="train")
        pools = {0: [], 1: []}
        for t, y in zip(ds["text"], ds["label"]):
            if t and len(t.split()) >= 3:
                pools[int(y)].append(t)
        xs, ys = [], []
        for y in (0, 1):
            didx = dev_rng.permutation(len(pools[y]))
            dev_part, train_part = didx[:64], didx[64:]
            xs += [pools[y][i] for i in dev_part]
            ys += [y] * 64
            pick = rng.choice(train_part, 160 * occ[name], replace=False)
            for o in range(occ[name]):
                picks[(name, o, y)] = [pools[y][i] for i in pick[o * 160:(o + 1) * 160]]
        dev[name] = {"x": xs, "y": ys}
    stream, seen = [], {}
    for name in sched:
        o = seen.get(name, 0)
        seen[name] = o + 1
        rows = [(t, y) for y in (0, 1) for t in picks[(name, o, y)]]
        stream += _batches(rows, bs, name, rng)
    return stream, dev, 2


def make_stream_t(kind, seed, bs=16):
    if kind.endswith("_iid"):
        stream, dev, n = make_stream_t(kind[:-4], seed, bs)
        rows = [(x, y) for b in stream for x, y in zip(b["x"], b["y"])]
        return _batches(rows, bs, "iid", np.random.default_rng(seed + 31337)), dev, n
    if kind == "tweet_conc":
        return tweet_conc(kind, seed, bs)
    raise ValueError(kind)

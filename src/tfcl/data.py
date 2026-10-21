"""Unified continual-learning streams.

A stream is a list of batches {"x": [...], "y": [...], "seg": str}; "seg" is
evaluation metadata only and is never shown to a learner. ``dev`` maps each
segment name to its held-out development examples. Official test splits are
never loaded here (reserved for the final frozen evaluation).

Kinds:
  banking_rec  BANKING77 A1 B1 A2 C1 B2 class-group recurrence (33 classes);
               same class groups, schedule and dev split as the original stream.
  banking_cil  BANKING77, 7 disjoint groups of 11 classes, each seen once.
  clinc_cil    CLINC150 (in-scope intents), 10 groups of 15 classes.
  amazon_dil   Amazon sentiment, same 2 labels, domain shift over 5 domains.
  amazon_rec   Original A1 B1 A2 C1 B2 Amazon domain recurrence.
  amazon_dilconf   amazon_dil with domains B and D flipped (second, independent
               concept-drift positive control).
  amazon_conflict  amazon_rec with domain B's labels flipped (concept drift;
               positive control where one shared mapping cannot fit all data).
"""
from collections import defaultdict

import numpy as np
from datasets import load_dataset


def _batches(rows, bs, seg, rng):
    rows = list(rows)
    order = rng.permutation(len(rows))
    rows = [rows[i] for i in order]
    return [{"x": [r[0] for r in rows[i:i + bs]], "y": [r[1] for r in rows[i:i + bs]], "seg": seg}
            for i in range(0, len(rows), bs)]


def _dev_split_banking(ds):
    # Identical to experiments/controller_recurrence_cau_v1.make_dev_split.
    labels = np.asarray(ds["label"])
    rng = np.random.default_rng(424242)
    tr, dv = [], []
    for c in range(77):
        ids = np.flatnonzero(labels == c).copy()
        rng.shuffle(ids)
        n = max(1, int(round(0.2 * len(ids))))
        dv += ids[:n].tolist()
        tr += ids[n:].tolist()
    return sorted(tr), sorted(dv)


def _by_class(texts, labels):
    out = defaultdict(list)
    for t, y in zip(texts, labels):
        out[int(y)].append(t)
    return out


def _class_incremental(train_by, dev_by, groups, per_class, bs, seed, schedule=None):
    rng = np.random.default_rng(seed + 9000)
    names = [chr(ord("A") + i) for i in range(len(groups))]
    schedule = schedule or [(n, 1) for n in names]
    occ_need = defaultdict(int)
    for n, _ in schedule:
        occ_need[n] += 1
    chosen = {}
    for n, cls in zip(names, groups):
        for c in cls:
            pool = list(train_by[c])
            idx = rng.permutation(len(pool))
            need = per_class * occ_need[n]
            assert len(pool) >= need, (c, len(pool), need)
            for o in range(occ_need[n]):
                chosen[(n, o + 1, c)] = [pool[i] for i in idx[o * per_class:(o + 1) * per_class]]
    stream = []
    for n, o in schedule:
        cls = groups[names.index(n)]
        rows = [(t, c) for c in cls for t in chosen[(n, o, c)]]
        stream += _batches(rows, bs, n, rng)
    dev = {n: {"x": [t for c in cls for t in dev_by[c]], "y": [c for c in cls for _ in dev_by[c]]}
           for n, cls in zip(names, groups) if any(s == n for s, _ in schedule)}
    return stream, dev


def banking(kind, seed, bs=16):
    ds = load_dataset("PolyAI/banking77")["train"]
    tr, dv = _dev_split_banking(ds)
    texts, labels = ds["text"], ds["label"]
    train_by = _by_class([texts[i] for i in tr], [labels[i] for i in tr])
    dev_by = _by_class([texts[i] for i in dv], [labels[i] for i in dv])
    if kind == "banking_rec":
        groups = [list(range(0, 11)), list(range(11, 22)), list(range(22, 33))]
        sched = [("A", 1), ("B", 1), ("A", 2), ("C", 1), ("B", 2)]
        per = min(min(len(train_by[c]) // 2 for c in range(22)), min(len(train_by[c]) for c in range(22, 33)))
        return _class_incremental(train_by, dev_by, groups, per, bs, seed, sched) + (77,)
    if kind == "banking_cil":
        perm = np.random.default_rng(777).permutation(77)  # fixed class-to-group assignment
        groups = [sorted(perm[i * 11:(i + 1) * 11].tolist()) for i in range(7)]
        per = min(len(train_by[c]) for c in range(77))  # 28: largest balanced budget
        return _class_incremental(train_by, dev_by, groups, per, bs, seed) + (77,)
    raise ValueError(kind)


def clinc(kind, seed, bs=16):
    d = load_dataset("clinc/clinc_oos", "plus")
    names = d["train"].features["intent"].names
    oos = names.index("oos")
    keep = [i for i in range(len(names)) if i != oos]
    remap = {c: i for i, c in enumerate(keep)}
    tr = [(t, remap[y]) for t, y in zip(d["train"]["text"], d["train"]["intent"]) if y != oos]
    va = [(t, remap[y]) for t, y in zip(d["validation"]["text"], d["validation"]["intent"]) if y != oos]
    train_by = _by_class(*zip(*tr))
    dev_by = _by_class(*zip(*va))
    perm = np.random.default_rng(777).permutation(150)
    groups = [sorted(perm[i * 15:(i + 1) * 15].tolist()) for i in range(10)]
    return _class_incremental(train_by, dev_by, groups, 40, bs, seed) + (150,)


AMAZON_DOMAINS = ["home", "apparel", "drugstore", "wireless", "kitchen"]


def amazon(kind, seed, bs=16):
    def collect(split):
        ds = load_dataset("goosmanlei/amazon_reviews_multi", "en", split=split)
        out = defaultdict(lambda: defaultdict(list))
        for dom, stars, body in zip(ds["product_category"], ds["stars"], ds["review_body"]):
            if dom in AMAZON_DOMAINS and body and int(stars) != 3:
                out[dom][int(int(stars) >= 4)].append(str(body))
        return out
    tr, va = collect("train"), collect("validation")
    rng = np.random.default_rng(seed + 9000)
    if kind in ("amazon_rec", "amazon_conflict"):
        doms = AMAZON_DOMAINS[:3]
        sched = [(0, 1), (1, 1), (0, 2), (2, 1), (1, 2)]
        per = 128
    else:
        doms = AMAZON_DOMAINS
        sched = [(i, 1) for i in range(5)]
        per = 160
    names = [chr(ord("A") + i) for i in range(len(doms))]
    # amazon_conflict: real concept drift. Domain B's sentiment labels are flipped, in the
    # stream and in its dev set, so a single input->label map cannot fit every domain.
    flip = {1} if kind == "amazon_conflict" else ({1, 3} if kind == "amazon_dilconf" else set())
    picks = {}
    for i, dom in enumerate(doms):
        occ = sum(1 for j, _ in sched if j == i)
        for y in (0, 1):
            pool = tr[dom][y]
            idx = rng.permutation(len(pool))
            for o in range(occ):
                picks[(i, o + 1, y)] = [pool[k] for k in idx[o * per:(o + 1) * per]]
    stream = []
    for i, o in sched:
        rows = [(t, (1 - y) if i in flip else y) for y in (0, 1) for t in picks[(i, o, y)]]
        stream += _batches(rows, bs, names[i], rng)
    dev_rng = np.random.default_rng(424242)
    dev = {}
    for i, dom in enumerate(doms):
        xs, ys = [], []
        for y in (0, 1):
            pool = va[dom][y]
            idx = dev_rng.permutation(len(pool))[:64]
            xs += [pool[k] for k in idx]
            ys += [(1 - y) if i in flip else y] * len(idx)
        dev[names[i]] = {"x": xs, "y": ys}
    return stream, dev, 2


def news(kind, seed, bs=16):
    """HELD-OUT CIL: 20 Newsgroups (train split only), 5 groups of 4 classes."""
    d = load_dataset("SetFit/20_newsgroups", split="train")
    rows = [(t, y) for t, y in zip(d["text"], d["label"]) if t and len(t.split()) >= 5]
    by = _by_class(*zip(*rows))
    rng = np.random.default_rng(424242)
    train_by, dev_by = {}, {}
    for c, ts in by.items():
        idx = rng.permutation(len(ts))
        dev_by[c] = [ts[i] for i in idx[:50]]
        train_by[c] = [ts[i] for i in idx[50:]]
    perm = np.random.default_rng(777).permutation(20)
    groups = [sorted(perm[i * 4:(i + 1) * 4].tolist()) for i in range(5)]
    return _class_incremental(train_by, dev_by, groups, 80, bs, seed) + (20,)


SENTI = [("A", "stanfordnlp/sst2", None, "sentence"), ("B", "fancyzhx/yelp_polarity", None, "text"),
         ("C", "cardiffnlp/tweet_eval", "sentiment", "text"), ("D", "stanfordnlp/imdb", None, "text")]


def senti(kind, seed, bs=16):
    """HELD-OUT same-label shift across four sentiment datasets (train splits only).
    senti_dil: SST-2 -> Yelp -> tweets -> IMDB. senti_conf: same with Yelp and IMDB flipped."""
    flip = {"B", "D"} if kind == "senti_conf" else set()
    rng = np.random.default_rng(seed + 9000)
    dev_rng = np.random.default_rng(424242)
    stream, dev = [], {}
    for name, path, cfg, col in SENTI:
        ds = load_dataset(path, cfg, split="train")
        lab = ds["label"]
        txt = ds[col]
        pools = {0: [], 1: []}
        for t, y in zip(txt, lab):
            if path.startswith("cardiffnlp"):
                if y == 1:
                    continue
                y = 0 if y == 0 else 1
            if t and len(t.split()) >= 3:
                pools[int(y)].append(t)
        rows, xs, ys = [], [], []
        for y in (0, 1):
            didx = dev_rng.permutation(len(pools[y]))
            dev_part, train_part = didx[:64], didx[64:]
            yy = (1 - y) if name in flip else y
            xs += [pools[y][i] for i in dev_part]
            ys += [yy] * 64
            pick = rng.choice(train_part, 160, replace=False)
            rows += [(pools[y][i], yy) for i in pick]
        stream += _batches(rows, bs, name, rng)
        dev[name] = {"x": xs, "y": ys}
    return stream, dev, 2


def make_stream(kind, seed, bs=16):
    """``<kind>_iid``: identical examples, globally shuffled (no-shift null stream)."""
    if kind.endswith("_iid"):
        stream, dev, n = make_stream(kind[:-4], seed, bs)
        rows = [(x, y) for b in stream for x, y in zip(b["x"], b["y"])]
        return _batches(rows, bs, "iid", np.random.default_rng(seed + 31337)), dev, n
    if kind.startswith("banking"):
        return banking(kind, seed, bs)
    if kind.startswith("clinc"):
        return clinc(kind, seed, bs)
    if kind.startswith("amazon"):
        return amazon(kind, seed, bs)
    if kind.startswith("news"):
        return news(kind, seed, bs)
    if kind.startswith("senti"):
        return senti(kind, seed, bs)
    raise ValueError(kind)

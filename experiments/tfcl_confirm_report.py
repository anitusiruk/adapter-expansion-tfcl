"""Pre-registered analysis of the confirmation study (notes/tmlr_confirmation_prereg.md)."""
import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

CIL = ["banking_rec", "banking_cil", "clinc_cil", "news_cil"]
DIL = ["amazon_rec", "amazon_dil", "senti_dil"]
DRIFT = ["amazon_conflict", "amazon_dilconf", "senti_conf"]
HELD_OUT = {"news_cil", "senti_dil", "senti_conf"}
TRIG = ["label_novel", "loss_z", "repr_z", "interf_logit", "fresh_util", "interf_proto", "conflict_z",
        "label_surprise"]


def load(d, metric="proto_mean"):
    acc = defaultdict(dict)  # (stream, policy) -> seed -> value
    pk = defaultdict(dict)
    raw = {}
    for f in glob.glob(f"{d}/*/*.json"):
        s = f.split("/")[-2]
        pol, seed = os.path.basename(f)[:-5].rsplit("_s", 1)
        r = json.loads(Path(f).read_text())
        acc[(s, pol)][int(seed)] = r["final_mean"][metric]
        pk[(s, pol)][int(seed)] = r["packages"]
        raw[(s, pol, int(seed))] = r
    return acc, pk, raw


def paired(acc, s, a, b):
    A, B = acc.get((s, a), {}), acc.get((s, b), {})
    seeds = sorted(set(A) & set(B))
    if len(seeds) < 3:
        return None
    d = np.array([A[x] - B[x] for x in seeds])
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    t = stats.t.ppf(0.975, len(d) - 1)
    p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return {"stream": s, "a": a, "b": b, "n": len(d), "mean": float(m), "lo": float(m - t * se),
            "hi": float(m + t * se), "p_wilcoxon": float(p), "wins": int((d > 0).sum()), "losses": int((d < 0).sum())}


def holm(rows, alpha=0.05):
    ps = [r["p_wilcoxon"] for r in rows]
    order = np.argsort(ps)
    m = len(ps)
    rej, running = [False] * m, True
    for k, i in enumerate(order):
        running = running and ps[i] <= alpha / (m - k)
        rej[i] = running
    for r, x in zip(rows, rej):
        r["holm_reject"] = bool(x)
    return rows


def novel_steps(stream_name, seed):
    from src.tfcl.data import make_stream
    st, _, _ = make_stream(stream_name, seed)
    seen, out = set(), set()
    for i, b in enumerate(st):
        if any(y not in seen for y in b["y"]):
            out.add(i)
        seen |= set(b["y"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/tfcl/confirm")
    ap.add_argument("--q", default="0.99")
    ap.add_argument("--out", default="results/tfcl/confirm_report.json")
    ap.add_argument("--skip-h1", action="store_true")
    a = ap.parse_args()
    acc, pk, raw = load(a.dir)
    T = lambda t: f"trigger-{t}-q{a.q}"
    rep = {"q": a.q, "dir": a.dir}

    table = {}
    for (s, pol), v in acc.items():
        table.setdefault(s, {})[pol] = {"mean": float(np.mean(list(v.values()))),
                                        "sd": float(np.std(list(v.values()), ddof=1)) if len(v) > 1 else 0.0,
                                        "n": len(v), "packages": float(np.mean(list(pk[(s, pol)].values())))}
    rep["table"] = table

    # H2: expansion harms CIL
    h2 = []
    for s in CIL:
        for pol in ["oracle", T("loss_z"), T("fresh_util"), T("label_novel")]:
            r = paired(acc, s, pol, "single")
            if r:
                h2.append(r)
    rep["H2"] = holm(h2)
    # H3: expansion helps drift; label_novel silent in DIL/DRIFT
    h3 = []
    for s in DRIFT:
        for pol in ["oracle", T("label_surprise")]:
            r = paired(acc, s, pol, "single")
            if r:
                h3.append(r)
    rep["H3"] = holm(h3)
    rep["H3_label_novel_spawns_dil_drift"] = {
        s: int(sum(v - 1 for v in pk.get((s, T("label_novel")), {}).values())) for s in DIL + DRIFT}
    # H4: non-inferiority of label_surprise in CIL and DIL
    h4 = []
    for s in CIL + DIL:
        r = paired(acc, s, T("label_surprise"), "single")
        if r:
            r["noninferior_margin_0.01"] = bool(r["lo"] > -0.010)
            h4.append(r)
    rep["H4"] = h4
    # secondary: every trigger and control vs single, every stream
    sec = []
    for s in CIL + DIL + DRIFT:
        for pol in ["frozen", "oracle", "random"] + [T(t) for t in TRIG]:
            r = paired(acc, s, pol, "single")
            if r:
                sec.append(r)
    rep["all_vs_single"] = sec
    # H1: spawn attribution to label novelty
    if not a.skip_h1:
        h1 = {}
        for s in CIL:
            cache = {}
            for t in ["loss_z", "fresh_util"]:
                hits = tot = 0
                for seed in range(2027, 2037):
                    r = raw.get((s, T(t), seed))
                    if r is None:
                        continue
                    if seed not in cache:
                        cache[seed] = novel_steps(s, seed)
                    nov = cache[seed]
                    for e in r["events"]:
                        tot += 1
                        hits += any((e["step"] - k) in nov for k in range(0, 3))
                h1[f"{s}/{t}"] = {"spawns": tot, "within2_of_novel": hits,
                                  "fraction": hits / tot if tot else float("nan")}
        rep["H1_spawn_attribution"] = h1
    Path(a.out).write_text(json.dumps(rep, indent=1))
    for fam in ["H2", "H3", "H4"]:
        print(fam)
        for r in rep[fam]:
            print(f"  {r['stream']:16s} {r['a']:34s} {100*r['mean']:+6.2f} [{100*r['lo']:+6.2f},{100*r['hi']:+6.2f}] "
                  f"W/L {r['wins']}/{r['losses']} p={r['p_wilcoxon']:.4f}" + (f" holm={r.get('holm_reject')}" if 'holm_reject' in r else "")
                  + (f" NI={r.get('noninferior_margin_0.01')}" if 'noninferior_margin_0.01' in r else ""))
    if "H1_spawn_attribution" in rep:
        print("H1", json.dumps(rep["H1_spawn_attribution"]))
    print("label_novel spawns in DIL/DRIFT:", rep["H3_label_novel_spawns_dil_drift"])


if __name__ == "__main__":
    main()

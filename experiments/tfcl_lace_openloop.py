"""Open-loop analysis of a LACE-style loss-ratio rule on single-package shadow traces.

LACE (Tathe, 2026) expands when the training loss exceeds tau times the mean of
the previous W losses (published tau = 2.5, W = 50, K = 1, cooldown 60, warmup 100
steps of batch 64). Our streams are shorter (16-example batches), so we apply the
published tau and K = 1 with W = 20, our common warmup of 8 batches and a cooldown
of 8 batches. We report, per stream: number of firings, fraction within 2 batches
after a batch containing a never-seen label, and fraction within 2 batches after a
true segment change. Traces come from ``--policy shadow`` runs (one package).
"""
import argparse
import glob
import json
from pathlib import Path


def fires(recs, tau=2.5, window=20, warmup=8, cooldown=8):
    hist, out, last = [], [], -10 ** 9
    for i, r in enumerate(recs):
        loss = r.get("loss")
        if loss is None:
            continue
        if len(hist) >= 5 and i >= warmup and i - last > cooldown:
            base = sum(hist[-window:]) / len(hist[-window:])
            if loss > tau * base:
                out.append(i)
                last = i
        hist.append(loss)
    return out


def analyse(recs, **kw):
    f = fires(recs, **kw)
    seen, novel, change = set(), set(), set()
    for i, r in enumerate(recs):
        if i > 0 and r["seg"] != recs[i - 1]["seg"]:
            change.add(i)
    return f, change


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", nargs="+", required=True, help="shadow JSON files")
    ap.add_argument("--tau", type=float, default=2.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from src.tfcl.data import make_stream
    rows = {}
    for pat in a.glob:
        for fn in sorted(glob.glob(pat)):
            r = json.loads(Path(fn).read_text())
            s, seed = r["args"]["stream"], r["args"]["seed"]
            if s.endswith("_iid"):
                continue
            recs = r["trigger_records"]
            st = make_stream(s, seed)[0] if not s.startswith("cifar") else None
            if st is None:
                from src.tfcl.vision import make_stream_v
                st = make_stream_v(s, seed)[0]
            seen, novel = set(), set()
            for i, b in enumerate(st):
                if any(y not in seen for y in b["y"]):
                    novel.add(i)
                seen |= set(b["y"])
            f, change = analyse(recs, tau=a.tau)
            near = lambda e, S: any((e - k) in S for k in range(3))
            d = rows.setdefault(s, {"fires": 0, "near_novel": 0, "near_change": 0, "runs": 0, "changes": 0})
            d["runs"] += 1
            d["changes"] += len(change)
            d["fires"] += len(f)
            d["near_novel"] += sum(near(e, novel - {0}) for e in f)
            d["near_change"] += sum(near(e, change) for e in f)
    print(json.dumps(rows, indent=1))
    Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()

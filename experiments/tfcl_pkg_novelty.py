"""EXPLORATORY (post hoc, after H1 failed on banking_rec): spawn attribution to package-level label novelty."""
import argparse
import json
from pathlib import Path

from src.tfcl.data import make_stream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/tfcl/confirm")
    ap.add_argument("--streams", nargs="+", default=["banking_rec", "banking_cil", "clinc_cil", "news_cil"])
    ap.add_argument("--q", default="0.99")
    a = ap.parse_args()
    out = {}
    for s in a.streams:
        for t in ["loss_z", "fresh_util", "label_novel"]:
            hits = tot = stream_hits = 0
            for seed in range(2027, 2037):
                f = Path(a.dir) / s / f"trigger-{t}-q{a.q}_s{seed}.json"
                if not f.exists():
                    continue
                r = json.loads(f.read_text())
                st = make_stream(s, seed)[0]
                spawns = sorted(e["step"] for e in r["events"])
                pkg_seen, glob_seen, pnov, gnov = set(), set(), set(), set()
                for i, b in enumerate(st):
                    if i in spawns:
                        pkg_seen = set()
                    if any(y not in pkg_seen for y in b["y"]):
                        pnov.add(i)
                    if any(y not in glob_seen for y in b["y"]):
                        gnov.add(i)
                    pkg_seen |= set(b["y"]); glob_seen |= set(b["y"])
                # Attribute with novelty status before the spawn decision, i.e. relative to the deciding package.
                pkg_seen, pnov_dec = set(), set()
                sp = set(spawns)
                for i, b in enumerate(st):
                    if any(y not in pkg_seen for y in b["y"]):
                        pnov_dec.add(i)
                    if i in sp:
                        pkg_seen = set()
                    pkg_seen |= set(b["y"])
                for e in spawns:
                    tot += 1
                    hits += any((e - k) in pnov_dec for k in range(3))
                    stream_hits += any((e - k) in gnov for k in range(3))
            out[f"{s}/{t}"] = {"spawns": tot, "stream_level": stream_hits / tot if tot else None,
                               "package_level": hits / tot if tot else None}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

"""Wall-clock cost per run by policy (mean over streams and seeds), relative to single."""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["results/tfcl/confirm"])
    ap.add_argument("--out", default="paper/tables/cost.tex")
    a = ap.parse_args()
    secs = defaultdict(dict)
    for d in a.dirs:
        for f in glob.glob(f"{d}/*/*.json"):
            s = f.split("/")[-2]
            pol, seed = os.path.basename(f)[:-5].rsplit("_s", 1)
            secs[pol][(s, seed)] = json.load(open(f))["seconds"]
    base = secs.get("single", {})
    rows = []
    for pol, v in sorted(secs.items()):
        ratio = [v[k] / base[k] for k in v if k in base]
        rows.append((pol, np.mean(list(v.values())), np.median(ratio) if ratio else float("nan"), len(v)))
    lines = ["\\begin{tabular}{lrrr}\\toprule", "Policy & mean s/run & median ratio to single & runs\\\\\\midrule"]
    for pol, m, r, n in rows:
        lines.append(f"{pol.replace('_', chr(92) + '_')} & {m:.0f} & {r:.2f} & {n}\\\\")
    lines.append("\\bottomrule\\end{tabular}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write("\n".join(lines) + "\n")
    for pol, m, r, n in rows:
        print(f"{pol:36s} {m:7.0f}s  x{r:5.2f}  n={n}")


if __name__ == "__main__":
    main()

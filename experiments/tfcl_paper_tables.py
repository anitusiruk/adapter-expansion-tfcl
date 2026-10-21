"""LaTeX tables for the paper from saved run JSONs (proto_mean unless --metric)."""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

GROUPS = [("CIL", ["banking_rec", "banking_cil", "clinc_cil", "news_cil"]),
          ("DIL", ["amazon_rec", "amazon_dil", "senti_dil"]),
          ("Drift", ["amazon_conflict", "amazon_dilconf", "senti_conf"])]
HELD = {"news_cil", "senti_dil", "senti_conf"}
COLS = [("frozen", "Frozen"), ("single", "Single"), ("firstseg", "First-seg."), ("oracle", "Oracle"),
        ("random", "Random"), ("trigger-label_novel-q0.99", "LabelNovel"), ("trigger-loss_z-q0.99", "LossZ"),
        ("trigger-repr_z-q0.99", "ReprZ"), ("trigger-interf_logit-q0.99", "InterfLogit"),
        ("trigger-fresh_util-q0.99", "FreshUtil"), ("trigger-label_surprise-q0.99", "\\textsc{LS}")]


def load(dirs, metric):
    acc, pk = defaultdict(dict), defaultdict(dict)
    for d in dirs:
        for f in glob.glob(f"{d}/*/*.json"):
            s = f.split("/")[-2]
            pol, seed = os.path.basename(f)[:-5].rsplit("_s", 1)
            r = json.load(open(f))
            acc[(s, pol)][seed] = r["final_mean"][metric]
            pk[(s, pol)][seed] = r["packages"]
    return acc, pk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["results/tfcl/confirm", "results/tfcl/ext_recon"])
    ap.add_argument("--metric", default="proto_mean")
    ap.add_argument("--groups", default=None, help="json override of GROUPS")
    ap.add_argument("--caption", default="")
    ap.add_argument("--label", default="tab:main")
    ap.add_argument("--out", default="paper/tables/main_table.tex")
    a = ap.parse_args()
    groups = json.loads(a.groups) if a.groups else GROUPS
    acc, pk = load(a.dirs, a.metric)
    cols = [c for c in COLS if any((s, c[0]) in acc for _, ss in groups for s in ss)]
    L = ["\\begin{table}[t]\\centering\\small", f"\\caption{{{a.caption}}}\\label{{{a.label}}}",
         "\\resizebox{\\linewidth}{!}{%", "\\begin{tabular}{l" + "r" * len(cols) + "}\\toprule",
         "Stream & " + " & ".join(n for _, n in cols) + "\\\\\\midrule"]
    for g, ss in groups:
        L.append(f"\\multicolumn{{{len(cols) + 1}}}{{l}}{{\\emph{{{g}}}}}\\\\")
        for s in ss:
            if not any((s, c) in acc for c, _ in cols):
                continue
            base = acc.get((s, "single"), {})
            cells = []
            for c, _ in cols:
                v = acc.get((s, c))
                if not v:
                    cells.append("--")
                    continue
                m = 100 * np.mean(list(v.values()))
                p = np.mean(list(pk[(s, c)].values()))
                cell = f"{m:.1f}"
                if c not in ("frozen", "single", "firstseg"):
                    cell += f"\\,{{\\scriptsize({p:.1f})}}"
                cells.append(cell)
            name = s.replace("_", "\\_") + ("$^\\dagger$" if s in HELD else "")
            L.append(name + " & " + " & ".join(cells) + "\\\\")
    L += ["\\bottomrule\\end{tabular}}", "\\end{table}"]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

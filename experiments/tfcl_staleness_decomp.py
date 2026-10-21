"""Staleness decomposition of the matched controls: V_ms = V_mf + (S_single - S_oracle), with S = A_mf - A_ms."""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

STREAMS = ["banking_rec", "banking_cil", "clinc_cil", "news_cil", "cifar_cil", "imnr_cil"]
LABEL = {"banking_rec": "BANKING-REC", "banking_cil": "BANKING-CIL", "clinc_cil": "CLINC",
         "news_cil": "20NG$^\\dagger$", "cifar_cil": "CIFAR-100", "imnr_cil": "ImageNet-R"}
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#222222", "#8a8a85"


def ci(x):
    x = np.asarray(x)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), h


def load(root, s):
    rows = []
    for f in sorted(glob.glob(f"{root}/{s}/single_s*.json")):
        o = f.replace("single_s", "oracle_s")
        if not os.path.exists(o):
            continue
        g, q = (json.load(open(p))["final_mean"] for p in (f, o))
        rows.append({"S_single": 100 * (g["proto_mf"] - g["proto_ms"]), "S_oracle": 100 * (q["proto_mf"] - q["proto_ms"]),
                     "V_ms": 100 * (q["proto_ms"] - g["proto_ms"]), "V_mf": 100 * (q["proto_mf"] - g["proto_mf"])})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/tfcl/ext6/matched")
    ap.add_argument("--out", default="figures/staleness.pdf")
    a = ap.parse_args()
    res = {}
    for s in STREAMS:
        rows = load(a.root, s)
        res[s] = {k: ci([r[k] for r in rows]) for k in rows[0]}
        res[s]["n"] = len(rows)
        res[s]["all_seeds_S_single_gt_S_oracle"] = all(r["S_single"] > r["S_oracle"] for r in rows)
        res[s]["shield"] = ci([r["S_single"] - r["S_oracle"] for r in rows])
        print(f"{s:12s} n={len(rows):2d} " + " ".join(f"{k}={m:+6.1f}±{h:4.1f}" for k, (m, h) in
              ((k, res[s][k]) for k in ("S_single", "S_oracle", "shield", "V_mf", "V_ms"))) +
              f" all_seeds={res[s]['all_seeds_S_single_gt_S_oracle']}")
    json.dump(res, open(a.out.replace(".pdf", ".json"), "w"), indent=1)

    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
    y = np.arange(len(STREAMS))[::-1]
    ax = axes[0]
    for k, (key, col, name) in enumerate([("S_single", ORANGE, "Single"), ("S_oracle", BLUE, "Oracle")]):
        m = [res[s][key][0] for s in STREAMS]; h = [res[s][key][1] for s in STREAMS]
        ax.barh(y + (0.18 if k == 0 else -0.18), m, height=0.34, xerr=h, color=col, label=name,
                error_kw={"ecolor": INK, "lw": 0.8})
    ax.set_xlabel("staleness cost $S$ (points)")
    ax.set_title("(a) accuracy lost to prototype feature age", fontsize=8.5, color=INK)
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    ax = axes[1]
    sh = [res[s]["shield"][0] for s in STREAMS]; vf = [res[s]["V_mf"][0] for s in STREAMS]
    ax.barh(y, sh, height=0.5, color=BLUE, label="staleness shield $S_{\\mathrm{Single}}-S_{\\mathrm{Oracle}}$")
    ax.barh(y, vf, height=0.5, color=MUTED, label="fresh-feature value $V_{\\mathrm{mf}}$")
    ax.plot([res[s]["V_ms"][0] for s in STREAMS], y, "D", color=INK, ms=4, mec="white", mew=0.8,
            label="total $V_{\\mathrm{ms}}$")
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Oracle $-$ Single (points)")
    ax.set_title("(b) decomposition of the value of expansion", fontsize=8.5, color=INK)
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.45, -0.28), ncol=3, columnspacing=1.0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([LABEL[s] for s in STREAMS])
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()

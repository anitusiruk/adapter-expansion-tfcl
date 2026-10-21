"""Regime-map figure: oracle expansion vs. single and first-segment, exemplar-based and exemplar-free."""
import argparse
import glob
import json
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROWS = [("CIL", ["banking_rec", "banking_cil", "clinc_cil", "news_cil", "cifar_cil", "imnr_cil"]),
        ("DIL", ["amazon_dil", "senti_dil", "cifar_dil"]),
        ("Drift", ["amazon_dilconf", "senti_conf", "cifar_conf"])]
HELD = {"news_cil", "senti_dil", "senti_conf"}
LABEL = {"banking_rec": "BANKING-REC", "banking_cil": "BANKING-CIL", "clinc_cil": "CLINC",
         "news_cil": "20News", "cifar_cil": "CIFAR-100 (ViT)", "imnr_cil": "ImageNet-R (ViT)",
         "amazon_dil": "Amazon-DIL", "senti_dil": "Sentiment-DIL", "cifar_dil": "CIFAR-sub (ViT)",
         "amazon_dilconf": "Amazon-DILCONF", "senti_conf": "Sentiment-CONF", "cifar_conf": "CIFAR-swap (ViT)"}
BLUE, ORANGE = "#2a78d6", "#eb6834"


def load(dirs, metric):
    v = defaultdict(dict)
    for d in dirs:
        for f in glob.glob(f"{d}/*/*.json"):
            m = re.match(r"(.+)_s(\d+)\.json", f.split("/")[-1])
            if not m:
                continue
            rec = json.load(open(f))
            if metric in rec.get("final_mean", {}):
                v[(f.split("/")[-2], m.group(1))][int(m.group(2))] = rec["final_mean"][metric]
    return v


def diff(v, s, a, b):
    ks = sorted(set(v.get((s, a), {})) & set(v.get((s, b), {})))
    if len(ks) < 3:
        return None
    d = np.array([v[(s, a)][k] - v[(s, b)][k] for k in ks]) * 100
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), h, len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/figures/regime.pdf")
    a = ap.parse_args()
    exb = load(["results/tfcl/confirm", "results/tfcl/ext_recon", "results/tfcl/ext_vision",
                "results/tfcl/ext3_imnr/exemplar"], "proto_mean")
    exf = load(["results/tfcl/ext2_ef", "results/tfcl/ext3_imnr/ef"], "proto_ef")
    streams = [(g, s) for g, ss in ROWS for s in ss]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 0.36 * len(streams) + 1.2), sharey=True)
    summary = {}
    for ax, ref, title in [(axes[0], "single", "(a) oracle expansion $-$ single package"),
                           (axes[1], "firstseg", "(b) oracle expansion $-$ first-segment adaptation")]:
        for y, (g, s) in enumerate(streams):
            for off, v, col, name in [(-0.15, exb, BLUE, "exemplar-based"), (0.15, exf, ORANGE, "exemplar-free")]:
                r = diff(v, s, "oracle", ref)
                if r is None:
                    continue
                summary[f"{ref}/{name}/{s}"] = r
                ax.errorbar(r[0], y + off, xerr=r[1], fmt="o", ms=5, color=col, ecolor=col, elinewidth=2,
                            capsize=0, mec="white", mew=1, label=name if y == 0 else None)
        ax.axvline(0, color="#555", lw=1.2, zorder=1.8)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("final accuracy difference (points)")
        ax.grid(axis="x", color="#e5e5e5", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        start = 0
        for g, ss in ROWS[:-1]:
            start += len(ss)
            ax.axhline(start - 0.5, color="#ccc", lw=0.6)
    axes[0].set_yticks(range(len(streams)))
    axes[0].set_yticklabels([LABEL[s] + ("$^\\dagger$" if s in HELD else "") + f"  [{g}]" for g, s in streams], fontsize=8)
    axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    json.dump({k: list(map(float, v)) for k, v in summary.items()}, open(a.out.replace(".pdf", "_data.json"), "w"), indent=1)
    print(f"wrote {a.out}; {len(summary)} cells")


if __name__ == "__main__":
    main()

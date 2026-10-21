"""Heatmap of paired accuracy difference vs. the single package (policy x stream)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

CMAP = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
NAMES = {"oracle": "Oracle", "random": "Random", "frozen": "Frozen NCM", "firstseg": "First-segment",
         "label_novel": "LabelNovel", "loss_z": "LossZ", "repr_z": "ReprZ", "interf_logit": "InterfLogit",
         "fresh_util": "FreshUtil", "interf_proto": "InterfProto", "conflict_z": "ConflictZ",
         "label_surprise": "LabelSurprise"}


def short(pol):
    if pol.startswith("trigger-"):
        pol = pol.split("-")[1]
    return NAMES.get(pol, pol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True, help="report.json:key pairs")
    ap.add_argument("--streams", nargs="+", required=True)
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--groups", nargs="+", default=None, help="stream group labels, e.g. CIL:4 DIL:3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vmax", type=float, default=15.0)
    a = ap.parse_args()
    rows = []
    for spec in a.rows:
        f, k = spec.split(":")
        rows += json.loads(Path(f).read_text())[k]
    M = np.full((len(a.policies), len(a.streams)), np.nan)
    sig = np.zeros_like(M, dtype=bool)
    for r in rows:
        if r["stream"] in a.streams and r["a"] in a.policies and r["b"] == "single":
            i, j = a.policies.index(r["a"]), a.streams.index(r["stream"])
            M[i, j] = 100 * r["mean"]
            sig[i, j] = r["lo"] > 0 or r["hi"] < 0
    fig, ax = plt.subplots(figsize=(0.62 * len(a.streams) + 1.8, 0.36 * len(a.policies) + 1.2))
    ax.imshow(M, cmap=CMAP, vmin=-a.vmax, vmax=a.vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.text(j, i, "–", ha="center", va="center", fontsize=7, color="#8a8a86")
                continue
            ax.text(j, i, f"{M[i, j]:+.1f}" + ("•" if sig[i, j] else ""), ha="center", va="center",
                    fontsize=6.5, color="#1a1a19")
    ax.set_xticks(range(len(a.streams)), [s.replace("_", "\n", 1) for s in a.streams], fontsize=6.5)
    ax.set_yticks(range(len(a.policies)), [short(p) for p in a.policies], fontsize=7)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(a.streams)), minor=True)
    ax.set_yticks(np.arange(-.5, len(a.policies)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    if a.groups:
        x = -0.5
        for g in a.groups:
            name, n = g.split(":")
            n = int(n)
            ax.text(x + n / 2, -0.9, name, ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                    color="#3d3d3a")
            if x > -0.5:
                ax.axvline(x, color="#1a1a19", linewidth=1)
            x += n
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()

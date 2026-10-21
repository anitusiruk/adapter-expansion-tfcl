"""Figure 1: per-batch trigger statistics on one CIL and one drift stream (development seed 2026)."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#222222", "#8a8a85"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
d = "results/tfcl/shadow"
rep = json.load(open(f"{d}/report_seed2026.json"))
streams = [("banking_cil", "BANKING77, class-incremental"), ("amazon_conflict", "Amazon, concept drift (B flipped)")]
fig, axes = plt.subplots(3, 2, figsize=(7.0, 5.0), sharex="col")
for c, (s, title) in enumerate(streams):
    R = json.load(open(f"{d}/{s}_s2026.json"))["trigger_records"]
    t = np.arange(len(R))
    seg = [r["seg"] for r in R]
    bounds = [i for i in range(1, len(R)) if seg[i] != seg[i - 1]]
    g = np.array([r["loss_group_term"] for r in R]); w = np.array([r["loss_within_term"] for r in R])
    lz = np.array([r["loss_z"] for r in R], float); ls = np.array([r["label_surprise"] for r in R], float)
    el = np.array([r["eligible"] for r in R])
    tau = rep[s]["null_tau_q0.99"]
    ax = axes[0, c]
    ax.bar(t, w, width=0.9, color=BLUE, label="within-group term", linewidth=0)
    ax.bar(t, g, bottom=w, width=0.9, color=ORANGE, label="label-group term", linewidth=0)
    ax.set_title(title, fontsize=9, color=INK)
    if c == 0:
        ax.set_ylabel("incoming loss")
        ax.legend(frameon=False, fontsize=7, loc="upper right")
    for k, (v, name, lab) in enumerate([(lz, "loss_z", "loss-spike z"), (ls, "label_surprise", "label surprise")]):
        ax = axes[k + 1, c]
        m = el & np.isfinite(v)
        ax.plot(t[m], v[m], color=INK, lw=1.2, marker="o", ms=2.2)
        ax.axhline(tau[name], color=ORANGE if k == 0 else BLUE, lw=1, ls="--")
        ax.text(len(R) - 1, tau[name], " null q.99", fontsize=6.5, color=MUTED, va="bottom", ha="right")
        if c == 0:
            ax.set_ylabel(lab)
    for ax in axes[:, c]:
        for b in bounds:
            ax.axvline(b - 0.5, color=MUTED, lw=0.6, alpha=0.6)
    axes[2, c].set_xlabel("batch (grey lines: hidden segment changes)")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"paper/figures/mechanism.{ext}", dpi=200)
print("ok")

"""Pre-registered analysis of extensions 3 (ImageNet-R) and 4 (SDC refresh);
notes/tmlr_extension34_prereg.md. Decision rule: the 95% paired t-interval over seeds lies
entirely on the predicted side of zero (n = 5-6 seeds; Wilcoxon p reported, not used)."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from experiments.tfcl_confirm_report import load

CIL4 = ["banking_rec", "banking_cil", "clinc_cil", "news_cil", "cifar_cil"]


def contrast(d, stream, label, sign):
    ks = sorted(d)
    x = np.array([d[k] for k in ks])
    if len(x) < 3:
        return {"stream": stream, "contrast": label, "n": len(x), "supported": None}
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    h = stats.t.ppf(0.975, len(x) - 1) * se
    p = stats.wilcoxon(x).pvalue if np.any(x != 0) else 1.0
    ok = (m - h > 0) if sign > 0 else (m + h < 0)
    return {"stream": stream, "contrast": label, "n": len(x), "mean": float(m), "lo": float(m - h),
            "hi": float(m + h), "p_wilcoxon": float(p), "wins": int((x > 0).sum()), "supported": bool(ok)}


def pair(acc, s, a, b):
    A, B = acc.get((s, a), {}), acc.get((s, b), {})
    return {k: A[k] - B[k] for k in set(A) & set(B)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdc", default="results/tfcl/ext4_sdc")
    ap.add_argument("--imnr", default="results/tfcl/ext3_imnr")
    ap.add_argument("--out", default="results/tfcl/extension34_report.json")
    a = ap.parse_args()
    rep = {"S1": [], "S2": [], "S_desc": [], "I": [], "I_desc": []}
    ef, _, _ = load(a.sdc, metric="proto_ef")
    sdc, _, _ = load(a.sdc, metric="proto_ef_sdc")
    for s in CIL4:
        # S1: SDC refreshes the single package's exemplar-free prototypes
        rep["S1"].append(contrast(pair({**{(s, "x"): sdc.get((s, "single"), {})}, **{(s, "y"): ef.get((s, "single"), {})}}, s, "x", "y"),
                                  s, "single: proto_ef_sdc - proto_ef", +1))
        # S2: refresh shrinks the exemplar-free value of expansion
        g_ef, g_sdc = pair(ef, s, "oracle", "single"), pair(sdc, s, "oracle", "single")
        rep["S2"].append(contrast({k: g_sdc[k] - g_ef[k] for k in set(g_ef) & set(g_sdc)}, s,
                                  "(oracle-single|sdc) - (oracle-single|ef)", -1))
        for m, acc in [("proto_ef", ef), ("proto_ef_sdc", sdc)]:
            for p, q in [("oracle", "single"), ("firstseg", "single"), ("oracle", "firstseg")]:
                rep["S_desc"].append(contrast(pair(acc, s, p, q), s, f"{m}: {p} - {q}", +1))
    for m in ["proto_ef_sdc02", "proto_ef_sdcinf"]:
        acc, _, _ = load(a.sdc, metric=m)
        for s in CIL4:
            rep["S_desc"].append(contrast(pair(acc, s, "oracle", "single"), s, f"{m}: oracle - single", +1))
    exb, _, _ = load(f"{a.imnr}/exemplar", metric="proto_mean")
    exf, _, _ = load(f"{a.imnr}/ef", metric="proto_ef")
    exf_mean, _, _ = load(f"{a.imnr}/ef", metric="proto_mean")
    exf_sdc, _, _ = load(f"{a.imnr}/ef", metric="proto_ef_sdc")
    s = "imnr_cil"
    rep["I"].append(contrast(pair(exb, s, "oracle", "single"), s, "I1 exemplar: oracle - single", -1))
    rep["I"].append(contrast(pair(exb, s, "oracle", "firstseg"), s, "I2 exemplar: oracle - firstseg", +1))
    rep["I"].append(contrast(pair(exf, s, "oracle", "single"), s, "I3 exemplar-free: oracle - single", +1))
    g1, g2 = pair(exf, s, "oracle", "single"), pair(exf_mean, s, "oracle", "single")
    rep["I"].append(contrast({k: g1[k] - g2[k] for k in set(g1) & set(g2)}, s, "I4 DiD (proto_ef - proto_mean, same runs)", +1))
    for lab, acc in [("exemplar proto_mean", exb), ("ef proto_ef", exf), ("ef proto_ef_sdc", exf_sdc)]:
        for p, q in [("firstseg", "single"), ("oracle", "firstseg"), ("oracle", "single")]:
            rep["I_desc"].append(contrast(pair(acc, s, p, q), s, f"{lab}: {p} - {q}", +1))
    Path(a.out).write_text(json.dumps(rep, indent=1))
    for k, rows in rep.items():
        print(k)
        for r in rows:
            if r.get("mean") is None:
                print(f"  {r['stream']:12s} {r['contrast']:45s} n={r['n']}")
                continue
            print(f"  {r['stream']:12s} {r['contrast']:45s} {100*r['mean']:+6.2f} [{100*r['lo']:+6.2f},{100*r['hi']:+6.2f}]"
                  f" n={r['n']} W={r['wins']} p={r['p_wilcoxon']:.3f} supported={r['supported']}")


if __name__ == "__main__":
    main()

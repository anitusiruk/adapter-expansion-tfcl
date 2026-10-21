"""Pre-registered analysis of the extension study (notes/tmlr_extension_prereg.md).

XV: vision replication (results/tfcl/ext_vision), XR: baseline dependence of the value
of expansion (results/tfcl/ext_recon + primary confirmation dirs), XL: open-loop
LACE-style loss-ratio rule on shadow traces (results/tfcl/ext_shadow).
Paired seed-level contrasts as in tfcl_confirm_report (t-interval, Wilcoxon, Holm).
"""
import argparse
import json
from pathlib import Path

from experiments.tfcl_confirm_report import holm, load, paired

VCIL, VDIL, VDRIFT = ["cifar_cil"], ["cifar_dil"], ["cifar_conf"]
TCIL = ["banking_rec", "banking_cil", "clinc_cil", "news_cil"]
TDIL = ["amazon_rec", "amazon_dil", "senti_dil"]
TDRIFT = ["amazon_conflict", "amazon_dilconf", "senti_conf"]


def fmt(r):
    return (f"  {r['stream']:16s} {r['a']:30s} vs {r['b']:18s} {100*r['mean']:+6.2f} "
            f"[{100*r['lo']:+6.2f},{100*r['hi']:+6.2f}] W/L {r['wins']}/{r['losses']} p={r['p_wilcoxon']:.4f}"
            + (f" holm={r['holm_reject']}" if "holm_reject" in r else "")
            + (f" NI={r['noninferior_margin_0.01']}" if "noninferior_margin_0.01" in r else ""))


def merge(*dicts):
    out = {}
    for d in dicts:
        for k, v in d.items():
            out.setdefault(k, {}).update(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision", default="results/tfcl/ext_vision")
    ap.add_argument("--recon", default="results/tfcl/ext_recon")
    ap.add_argument("--recon-noreplay", default="results/tfcl/ext_recon_noreplay")
    ap.add_argument("--vision-noreplay", default="results/tfcl/ext_vision_noreplay")
    ap.add_argument("--confirm", default="results/tfcl/confirm")
    ap.add_argument("--out", default="results/tfcl/extension_report.json")
    a = ap.parse_args()
    T = lambda t: f"trigger-{t}-q0.99"
    rep = {}

    # ---- XV: vision replication
    va, vpk, vraw = load(a.vision)
    xv2 = [r for s in VCIL for p in ["oracle", T("loss_z"), T("fresh_util"), T("label_novel")]
           if (r := paired(va, s, p, "single"))]
    xv3 = [r for s in VDRIFT for p in ["oracle", T("label_surprise")] if (r := paired(va, s, p, "single"))]
    xv4 = []
    for s in VCIL + VDIL:
        r = paired(va, s, T("label_surprise"), "single")
        if r:
            r["noninferior_margin_0.01"] = bool(r["lo"] > -0.010)
            xv4.append(r)
    rep["XV2"], rep["XV3"], rep["XV4"] = holm(xv2), holm(xv3), xv4
    from src.tfcl.vision import make_stream_v
    xv1 = {}
    for s in VCIL:
        for t in ["loss_z", "fresh_util"]:
            hits = tot = 0
            for seed in range(2027, 2037):
                r = vraw.get((s, T(t), seed))
                if r is None:
                    continue
                st = make_stream_v(s, seed)[0]
                seen, nov = set(), set()
                for i, b in enumerate(st):
                    if any(y not in seen for y in b["y"]):
                        nov.add(i)
                    seen |= set(b["y"])
                for e in r["events"]:
                    tot += 1
                    hits += any((e["step"] - k) in nov for k in range(3))
            xv1[f"{s}/{t}"] = {"spawns": tot, "within2_of_novel": hits, "fraction": hits / tot if tot else None}
    rep["XV1"] = xv1
    rep["XV_label_novel_spawns_dil_drift"] = {
        s: int(sum(v - 1 for v in vpk.get((s, T("label_novel")), {}).values())) for s in VDIL + VDRIFT}
    rep["XV_all_vs_single"] = [r for s in VCIL + VDIL + VDRIFT
                               for p in ["frozen", "oracle", "random", "firstseg"] +
                               [T(t) for t in ["label_novel", "loss_z", "repr_z", "fresh_util", "label_surprise"]]
                               if (r := paired(va, s, p, "single"))]

    # ---- XR: baseline dependence (oracle vs firstseg; with and without replay)
    ca, _, _ = load(a.confirm)
    ra, _, _ = load(a.recon)
    rna, _, _ = load(a.recon_noreplay)
    vna, _, _ = load(a.vision_noreplay)
    acc = merge(ca, ra, va)
    nore = merge(rna, vna)
    xr1 = [r for s in TCIL + VCIL if (r := paired(acc, s, "oracle", "firstseg"))]
    rep["XR1_oracle_vs_firstseg_CIL"] = holm(xr1)
    rep["XR_all_oracle_vs_firstseg"] = [r for s in TCIL + TDIL + TDRIFT + VCIL + VDIL + VDRIFT
                                        if (r := paired(acc, s, "oracle", "firstseg"))]
    rep["XR_single_vs_firstseg"] = [r for s in TCIL + TDIL + TDRIFT + VCIL + VDIL + VDRIFT
                                    if (r := paired(acc, s, "single", "firstseg"))]
    rep["XR2_noreplay_oracle_vs_single"] = [r for s in TCIL + TDIL + TDRIFT + VCIL + VDIL + VDRIFT
                                            if (r := paired(nore, s, "oracle", "single"))]
    # difference-in-differences: (oracle-single | no replay) - (oracle-single | replay)
    did = []
    for s in TCIL + TDIL + TDRIFT + VCIL + VDIL + VDRIFT:
        d_r = {k: acc.get((s, "oracle"), {}).get(k, None) for k in range(2027, 2037)}
        both = {}
        for k in range(2027, 2037):
            try:
                both[k] = ((nore[(s, "oracle")][k] - nore[(s, "single")][k])
                           - (acc[(s, "oracle")][k] - acc[(s, "single")][k]))
            except KeyError:
                pass
        if len(both) >= 3:
            did.append(paired({(s, "x"): both, (s, "z"): {k: 0.0 for k in both}}, s, "x", "z"))
            did[-1]["a"], did[-1]["b"] = "DiD(noreplay-replay)", "0"
    rep["XR2_did"] = did

    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    for k in ["XV2", "XV3", "XV4", "XV_all_vs_single", "XR1_oracle_vs_firstseg_CIL", "XR_all_oracle_vs_firstseg",
              "XR_single_vs_firstseg", "XR2_noreplay_oracle_vs_single", "XR2_did"]:
        print(k)
        for r in rep[k]:
            print(fmt(r))
    print("XV1", json.dumps(rep["XV1"]))
    print("label_novel spawns vision DIL/DRIFT", rep["XV_label_novel_spawns_dil_drift"])


if __name__ == "__main__":
    main()

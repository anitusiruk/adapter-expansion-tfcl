"""Pre-registered analysis of extension 2 (notes/tmlr_extension2_prereg.md): exemplar-free regime."""
import argparse
import json
from pathlib import Path

from experiments.tfcl_confirm_report import holm, load, paired
from experiments.tfcl_extension_report import fmt

CIL = ["banking_rec", "banking_cil", "clinc_cil", "news_cil", "cifar_cil"]
OTHER = ["amazon_dil", "senti_dil", "amazon_dilconf", "senti_conf", "cifar_dil", "cifar_conf"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/tfcl/ext2_ef")
    ap.add_argument("--out", default="results/tfcl/extension2_report.json")
    a = ap.parse_args()
    ef, pk, _ = load(a.dir, metric="proto_ef")
    ex, _, _ = load(a.dir, metric="proto_mean")
    T = lambda t: f"trigger-{t}-q0.99"
    rep = {}
    rep["R1"] = holm([r for s in CIL if (r := paired(ef, s, "oracle", "single"))])
    did = []
    for s in CIL:
        d = {}
        for k in set(ef.get((s, "oracle"), {})) & set(ef.get((s, "single"), {})):
            d[k] = (ef[(s, "oracle")][k] - ef[(s, "single")][k]) - (ex[(s, "oracle")][k] - ex[(s, "single")][k])
        if len(d) >= 3:
            r = paired({(s, "did"): d, (s, "zero"): {k: 0.0 for k in d}}, s, "did", "zero")
            r["a"], r["b"] = "DiD(ef - exemplar)", "0"
            did.append(r)
    rep["R2"] = holm(did)
    rep["R3"] = holm([r for s in CIL if (r := paired(ef, s, T("loss_z"), T("label_surprise")))])
    rep["R4"] = holm([r for s in CIL if (r := paired(ef, s, "oracle", "random"))])
    rep["all_vs_single_ef"] = [r for s in CIL + OTHER
                               for p in ["oracle", "random", "firstseg", T("label_novel"), T("loss_z"), T("label_surprise")]
                               if (r := paired(ef, s, p, "single"))]
    rep["all_vs_single_exemplar_same_runs"] = [r for s in CIL + OTHER
                                               for p in ["oracle", "random", "firstseg", T("loss_z"), T("label_surprise")]
                                               if (r := paired(ex, s, p, "single"))]
    rep["packages"] = {f"{s}/{p}": sum(v.values()) / len(v) for (s, p), v in pk.items()}
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    for k in ["R1", "R2", "R3", "R4", "all_vs_single_ef", "all_vs_single_exemplar_same_runs"]:
        print(k)
        for r in rep[k]:
            print(fmt(r))


if __name__ == "__main__":
    main()

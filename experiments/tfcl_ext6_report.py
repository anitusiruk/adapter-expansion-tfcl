"""Analysis of extension 6 (notes/tmlr_extension6_prereg.md). Committed before any extension-6
confirmation result. Also contains two re-analyses of saved results requested by the audit
(A4: refreshed value of expansion within the exemplar-free runs; A6: SEMA NCM-vs-NCM interaction).

Usage: python -m experiments.tfcl_ext6_report  -> results/tfcl/ext6_report.json (+ printed summary)
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

D = "results/tfcl/ext6"
TXT = ["banking_rec", "banking_cil", "clinc_cil", "news_cil"]
CIL = TXT + ["cifar_cil", "imnr_cil"]
ALLTXT = TXT + ["amazon_rec", "amazon_dil", "amazon_conflict", "amazon_dilconf", "senti_dil", "senti_conf"]
SEEDS = {**{s: range(2027, 2037) for s in ALLTXT}, "cifar_cil": range(2027, 2035), "imnr_cil": range(2027, 2033)}
HOLM = [s for s in CIL if s != "imnr_cil"]  # >= 8 seeds: Wilcoxon + Holm; imnr: t-interval rule


def ef_path(s, p, seed):
    if s == "imnr_cil":
        return f"results/tfcl/ext3_imnr/ef/imnr_cil/{p}_s{seed}.json"
    return f"results/tfcl/ext2_ef/{s}/{p}_s{seed}.json"


def ex_path(s, p, seed):
    if s == "imnr_cil":
        return f"results/tfcl/ext3_imnr/exemplar/imnr_cil/{p}_s{seed}.json"
    if s.startswith("cifar"):
        return f"results/tfcl/ext_vision/{s}/{p}_s{seed}.json"
    return f"results/tfcl/confirm/{s}/{p}_s{seed}.json"


def fm(path):
    return json.load(open(path))["final_mean"] if os.path.exists(path) else None


def summ(d, sign=None):
    """d: seed -> paired difference (accuracy fraction). Returns points (x100)."""
    x = np.array([d[k] for k in sorted(d)]) * 100
    n = len(x)
    if n < 2:
        return {"n": n, "values": x.tolist()}
    m, h = x.mean(), stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    try:
        p = float(stats.wilcoxon(x).pvalue) if np.any(x != 0) else 1.0
    except ValueError:
        p = 1.0
    r = {"n": n, "mean": float(m), "lo": float(m - h), "hi": float(m + h), "p_wilcoxon": p,
         "n_pos": int((x > 0).sum()), "n_neg": int((x < 0).sum()), "values": [round(v, 3) for v in x]}
    if sign is not None:
        r["predicted_sign"] = sign
        r["ci_excludes_zero_predicted"] = bool((m - h > 0) if sign > 0 else (m + h < 0))
    return r


def holm(res, keys):
    ps = sorted((res[k]["p_wilcoxon"], k) for k in keys if "p_wilcoxon" in res[k])
    m, prev = len(ps), 0.0
    for i, (p, k) in enumerate(ps):
        adj = min(1.0, max(prev, (m - i) * p))
        prev = adj
        res[k]["p_holm"] = adj


def decide(res, sign):
    for s, r in res.items():
        if "mean" not in r:
            r["supported"] = None
            continue
        ok_sign = (r["mean"] > 0) if sign > 0 else (r["mean"] < 0)
        if s in HOLM:
            r["supported"] = bool(ok_sign and r.get("p_holm", 1) < 0.05)
        else:
            r["supported"] = bool(r["ci_excludes_zero_predicted"])


def paired(a, b):
    return {k: a[k] - b[k] for k in set(a) & set(b)}


# --------------------------------------------------------------------------- block M
def block_m():
    rules = ["proto_ef", "proto_ms", "proto_mf", "proto_mfa", "proto_mean"]
    acc = defaultdict(dict)  # (s, p, rule) -> seed -> acc
    integ = {"checked": 0, "mismatch": []}
    for s in CIL:
        for p in ["single", "oracle"]:
            for seed in SEEDS[s]:
                r = fm(f"{D}/matched/{s}/{p}_s{seed}.json")
                if r is None:
                    continue
                for k in rules:
                    acc[(s, p, k)][seed] = r[k]
                old = fm(ef_path(s, p, seed))
                if old is not None:
                    integ["checked"] += 1
                    for k in ("proto_ef", "proto_mean"):
                        if abs(old[k] - r[k]) > 1e-9:
                            integ["mismatch"].append([s, p, seed, k, old[k], r[k]])
    val = {k: {s: paired(acc[(s, "oracle", k)], acc[(s, "single", k)]) for s in CIL} for k in rules}
    out = {"integrity": integ, "M1": {}, "M2": {}, "value_by_rule": {}, "decomposition": {}, "means": {}}
    for s in CIL:
        out["M1"][s] = summ(paired(acc[(s, "single", "proto_mf")], acc[(s, "single", "proto_ms")]), +1)
        out["M2"][s] = summ(paired(val["proto_ms"][s], val["proto_mf"][s]), +1)
        out["value_by_rule"][s] = {k: summ(val[k][s]) for k in rules}
        steps = {"sample (ef->ms)": ("proto_ef", "proto_ms"), "feature age (ms->mf)": ("proto_ms", "proto_mf"),
                 "coverage+aggregation (mf->mfa)": ("proto_mf", "proto_mfa"), "check (mfa->mean)": ("proto_mfa", "proto_mean"),
                 "total (ef->mean)": ("proto_ef", "proto_mean")}
        out["decomposition"][s] = {name: summ(paired(val[a][s], val[b][s])) for name, (a, b) in steps.items()}
        out["means"][s] = {f"{p}/{k}": float(np.mean(list(acc[(s, p, k)].values())) * 100)
                           for p in ["single", "oracle"] for k in rules if acc[(s, p, k)]}
    holm(out["M1"], HOLM); decide(out["M1"], +1)
    holm(out["M2"], HOLM); decide(out["M2"], +1)
    return out


# --------------------------------------------------------------------------- block G
def block_g():
    out = {"integrity": {}, "G1": {}, "G2": {}, "desc": {}}
    r = fm(f"{D}/global_replay/cifar_cil/single-global_s2027.json")
    o = fm(ex_path("cifar_cil", "single", 2027))
    if r and o:
        out["integrity"]["cifar_single_global_equals_saved"] = abs(r["proto_mean"] - o["proto_mean"]) < 1e-9
    for s in ALLTXT + ["cifar_cil", "imnr_cil"]:
        A = defaultdict(dict)
        for seed in SEEDS[s]:
            g = fm(f"{D}/global_replay/{s}/oracle-global_s{seed}.json")
            si, ow = fm(ex_path(s, "single", seed)), fm(ex_path(s, "oracle", seed))
            if g is None or si is None or ow is None:
                continue
            A["g"][seed], A["s"][seed], A["o"][seed] = g["proto_mean"], si["proto_mean"], ow["proto_mean"]
        rec = {"oracle_global - single": summ(paired(A["g"], A["s"]), -1),
               "oracle_global - oracle_owned": summ(paired(A["g"], A["o"])),
               "oracle_owned - single": summ(paired(A["o"], A["s"])),
               "means": {k: float(np.mean(list(v.values())) * 100) for k, v in A.items()}}
        if A["g"]:
            o_s = np.array([A["o"][k] - A["s"][k] for k in sorted(A["g"])])
            g_o = np.array([A["g"][k] - A["o"][k] for k in sorted(A["g"])])
            rec["share_of_harm_removed_by_global_replay"] = float(g_o.mean() / -o_s.mean()) if o_s.mean() < 0 else None
        out["desc"][s] = rec
        if s in CIL:
            out["G1"][s] = rec["oracle_global - single"]
    holm(out["G1"], HOLM); decide(out["G1"], -1)
    return out


# --------------------------------------------------------------------------- block F
def block_f():
    out = {}
    for d, tag in ((f"{D}/null_closed", "exemplar"), (f"{D}/null_closed_ef", "exemplar_free")):
        for f in sorted(glob.glob(f"{d}/*/*.json")):
            s = f.split("/")[-2]
            trig = f.split("/")[-1].split("-q0.99")[0].replace("trigger-", "")
            r = json.load(open(f))
            recs = r["trigger_records"]
            elig = sum(1 for x in recs if x.get("eligible"))
            tau = json.loads(r["args"]["trigger_args"])["tau"]
            vals = [x.get(trig) for x in recs if x.get("eligible")]
            vals = [v for v in vals if v is not None and v == v]
            exc = float(np.mean([v > tau for v in vals])) if vals else None
            e = out.setdefault(tag, {}).setdefault(s, {}).setdefault(trig, {"spawns": [], "eligible": [], "exceed": []})
            e["spawns"].append(r["packages"] - 1); e["eligible"].append(elig); e["exceed"].append(exc)
    summary = {}
    for tag, ss in out.items():
        for s, tt in ss.items():
            for t, e in tt.items():
                sp, el = np.array(e["spawns"]), np.array(e["eligible"])
                ex = [x for x in e["exceed"] if x is not None]
                summary.setdefault(tag, {}).setdefault(t, {})[s] = {
                    "runs": len(sp), "mean_spawns": float(sp.mean()), "runs_with_spawn": int((sp > 0).sum()),
                    "spawns_per_100_eligible": float(100 * sp.sum() / max(el.sum(), 1)),
                    "marginal_exceedance": float(np.mean(ex)) if ex else None}
    pooled = {}
    for tag, tt in summary.items():
        for t, ss in tt.items():
            sp = sum(v["mean_spawns"] * v["runs"] for v in ss.values())
            runs = sum(v["runs"] for v in ss.values())
            pooled.setdefault(tag, {})[t] = {"runs": runs, "mean_spawns_per_run": sp / max(runs, 1),
                                            "frac_runs_with_spawn": sum(v["runs_with_spawn"] for v in ss.values()) / max(runs, 1),
                                            "spawns_per_100_eligible_range": [min(v["spawns_per_100_eligible"] for v in ss.values()),
                                                                              max(v["spawns_per_100_eligible"] for v in ss.values())]}
    return {"per_stream": summary, "pooled": pooled}


# --------------------------------------------------------------------------- re-analyses
def reanalysis_a4():
    """Audit point 4: oracle - single under the refreshed rule within the exemplar-free runs."""
    out = {}
    for s in CIL:
        A = defaultdict(dict)
        for seed in SEEDS[s]:
            o, si = fm(ef_path(s, "oracle", seed)), fm(ef_path(s, "single", seed))
            if o and si:
                for k in ("proto_ef", "proto_mean"):
                    A[k][seed] = o[k] - si[k]
        out[s] = {"stored (proto_ef)": summ(A["proto_ef"]), "refreshed (proto_mean)": summ(A["proto_mean"])}
    return out


def reanalysis_a6():
    """Audit point 6: SEMA interaction with NCM on both sides (ncm_ef vs ncm_ex20)."""
    v = defaultdict(dict)
    for f in glob.glob("results/sema_x/*_s*.json"):
        name = os.path.basename(f)[:-5]
        var, seed = name.rsplit("_s", 1)
        r = json.load(open(f))
        for k in ("linear", "ncm_ef", "ncm_ex20", "ncm_full"):
            if k in r.get("final", {}):
                v[(var, k)][int(seed)] = r["final"][k] / (100 if r["final"][k] > 1.5 else 1)
    g = {k: paired(v[("sema", k)], v[("single", k)]) for k in ("linear", "ncm_ef", "ncm_ex20")}
    return {"sema - single | linear": summ(g["linear"]), "sema - single | ncm_ef": summ(g["ncm_ef"]),
            "sema - single | ncm_ex20": summ(g["ncm_ex20"]),
            "NCM interaction (ncm_ef - ncm_ex20)": summ(paired(g["ncm_ef"], g["ncm_ex20"])),
            "sema - noexp | linear": summ(paired(v[("sema", "linear")], v[("noexp", "linear")]))}


def main():
    rep = {"M": block_m(), "G": block_g(), "F": block_f(), "A4": reanalysis_a4(), "A6": reanalysis_a6()}
    json.dump(rep, open("results/tfcl/ext6_report.json", "w"), indent=1)
    f = lambda r: (f"{r['mean']:+.2f} [{r['lo']:+.2f},{r['hi']:+.2f}] {r['n_pos']}+/{r['n_neg']}- n={r['n']}"
                   + (f" pH={r['p_holm']:.3g}" if "p_holm" in r else "")
                   + (f" SUPPORTED={r['supported']}" if "supported" in r else "")) if "mean" in r else f"n={r['n']}"
    print("M integrity:", rep["M"]["integrity"]["checked"], "checked; mismatches:", rep["M"]["integrity"]["mismatch"][:5])
    for h in ("M1", "M2"):
        for s, r in rep["M"][h].items():
            print(h, s, f(r))
    for s in CIL:
        print("value", s, {k: round(v.get("mean", float("nan")), 2) for k, v in rep["M"]["value_by_rule"][s].items()})
        print("decomp", s, {k: round(v.get("mean", float("nan")), 2) for k, v in rep["M"]["decomposition"][s].items()})
    print("G integrity", rep["G"]["integrity"])
    for s, r in rep["G"]["desc"].items():
        print("G", s, "glob-single", f(r["oracle_global - single"]), "| glob-owned", f(r["oracle_global - oracle_owned"]),
              "| owned-single", f(r["oracle_owned - single"]), "| share", r.get("share_of_harm_removed_by_global_replay"))
    for tag, tt in rep["F"]["pooled"].items():
        for t, r in tt.items():
            print("F", tag, t, r)
    for s, r in rep["A4"].items():
        print("A4", s, "stored", f(r["stored (proto_ef)"]), "| refreshed", f(r["refreshed (proto_mean)"]))
    for k, r in rep["A6"].items():
        print("A6", k, f(r))


if __name__ == "__main__":
    main()

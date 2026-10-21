"""Analysis of extension 5 (notes/tmlr_extension5_prereg.md)."""
import glob
import json
import re
from collections import defaultdict

import numpy as np
from scipy import stats

RULES = ["linear", "ncm_ef", "ncm_ex20", "ncm_full"]


def load():
    v = defaultdict(dict)  # (variant, rule) -> seed -> final acc
    pk = defaultdict(dict)
    for f in glob.glob("results/sema_x/*_s*.json"):
        m = re.match(r"(\w+)_s(\d+)\.json", f.split("/")[-1])
        r = json.load(open(f))
        for k in RULES:
            if k in r["final"]:
                v[(m.group(1), k)][int(m.group(2))] = r["final"][k]
        v[(m.group(1), "avg_linear")][int(m.group(2))] = r["avg_inc"]["linear"]
        pk[m.group(1)][int(m.group(2))] = r["tasks"][-1]["adapters"]
    return v, pk


def test(d, sign, label):
    x = np.array([d[k] for k in sorted(d)])
    if len(x) < 2:
        return {"contrast": label, "n": len(x), "values": x.tolist()}
    m, h = x.mean(), stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    consistent = bool(np.all(np.sign(x) == sign))
    ci = (m - h > 0) if sign > 0 else (m + h < 0)
    return {"contrast": label, "n": len(x), "mean": float(m), "lo": float(m - h), "hi": float(m + h),
            "values": x.tolist(), "directionally_consistent": consistent, "supported": bool(consistent and ci)}


def diff(v, a, b, rule):
    A, B = v.get((a, rule), {}), v.get((b, rule), {})
    return {k: A[k] - B[k] for k in set(A) & set(B)}


def main():
    v, pk = load()
    rep = {"E1": test(diff(v, "sema", "single", "linear"), +1, "linear: sema - single"),
           "E2": None, "E3": test(diff(v, "single", "sema", "ncm_ex20"), +1, "ncm_ex20: single - sema")}
    g1, g2 = diff(v, "sema", "single", "linear"), diff(v, "sema", "single", "ncm_ex20")
    rep["E2"] = test({k: g1[k] - g2[k] for k in set(g1) & set(g2)}, +1, "(sema-single|linear) - (sema-single|ncm_ex20)")
    rep["desc"] = [test(diff(v, a, b, r), +1, f"{r}: {a} - {b}") for r in RULES + ["avg_linear"]
                   for a, b in [("sema", "single"), ("sema", "noexp"), ("single", "noexp")]]
    rep["means"] = {f"{a}/{r}": float(np.mean(list(s.values()))) for (a, r), s in v.items()}
    rep["adapters"] = pk
    json.dump(rep, open("results/sema_x/report.json", "w"), indent=1)
    for k in ["E1", "E2", "E3"] + list(range(len(rep["desc"]))):
        r = rep[k] if isinstance(k, str) else rep["desc"][k]
        print(json.dumps(r))
    print(json.dumps(rep["means"], indent=1))


if __name__ == "__main__":
    main()

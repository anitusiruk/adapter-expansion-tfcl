"""EXPLORATORY (session 4, not pre-registered): head rows of returning labels in the package that meets them."""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

import experiments.tfcl_run as R
from src.tfcl.data import make_stream
from src.tfcl.learner import Learner
from src.tfcl.model import Encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="banking_rec")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(2027, 2037)))
    ap.add_argument("--replay", type=int, default=16)
    ap.add_argument("--out", default="results/tfcl/return_rows.json")
    a = ap.parse_args()
    enc = None
    res = {}
    for seed in a.seeds:
        R.seed_all(seed)
        stream, dev, n_labels = make_stream(a.stream, seed)
        enc = enc or Encoder("distilbert-base-uncased", max_length=64)
        L = Learner(enc, n_labels, seed, replay=a.replay)
        trained = set()  # labels the active package has trained with positive examples
        events = []
        for i, b in enumerate(stream):
            ys = b["y"]
            ret = [y for y in set(ys) if y not in trained and bool(L.seen[y])]
            if ret and trained:
                p = L.active
                with torch.no_grad():
                    f = enc.features(b["x"], p.lora)
                    z = (f @ p.w.t() + p.b)
                    O = torch.tensor(sorted(trained), device=z.device)
                    Nset = sorted(y for y in set(ys) if y not in trained)
                    N = torch.tensor(Nset, device=z.device)
                    Rr = torch.tensor(sorted(ret), device=z.device)
                    gap = float((z[:, Rr].mean(1) - z[:, O].mean(1)).mean())
                    lse_O, lse_N = torch.logsumexp(z[:, O], 1), torch.logsumexp(z[:, N], 1)
                    grp_r = F.softplus(lse_O - lse_N)            # restricted to O u N (old)
                    grp0_r = F.softplus(lse_O - np.log(len(Nset)))
                    E = L.seen.clone()                           # eligible labels in the loss
                    E[torch.tensor(list(ys), device=z.device)] = True
                    lse_E = torch.logsumexp(z.masked_fill(~E, float("-inf")), 1)
                    grp = lse_E - lse_N                          # -log P(N | x), full classifier
                    z0 = z.clone()
                    z0[:, N] = 0.0                               # N rows (and biases) zero
                    grp0 = torch.logsumexp(z0.masked_fill(~E, float("-inf")), 1) - np.log(len(Nset))
                    mask = torch.tensor([y in Nset for y in ys], device=z.device)
                    events.append({"step": i, "seg": b["seg"], "n_returning": len(ret),
                                   "logit_gap_returning_minus_trained": gap,
                                   "n_eligible": int(E.sum()), "n_other_observed": int(E.sum()) - len(trained) - len(Nset),
                                   "group_term_full": float(grp[mask].mean()),
                                   "group_term_full_if_zero_rows": float(grp0[mask].mean()),
                                   "group_term_restricted": float(grp_r[mask].mean()),
                                   "group_term_restricted_if_zero_rows": float(grp0_r[mask].mean()),
                                   "row_norm_returning": float(p.w[Rr].norm(dim=1).mean()),
                                   "row_norm_trained": float(p.w[O].norm(dim=1).mean())})
            if i > 0 and b["seg"] != stream[i - 1]["seg"]:  # oracle spawn AFTER measuring the deciding package
                L.spawn()
                trained = set()
            L.observe(b["x"], ys)
            trained |= set(ys)
        ev = L.evaluate(dev, sorted(set(s["seg"] for s in stream)))
        final = float(np.mean([v["proto_mean"] for v in ev.values()]))
        saved = json.load(open(f"results/tfcl/confirm/{a.stream}/oracle_s{seed}.json"))["final_mean"]["proto_mean"]
        res[seed] = {"final_proto_mean": final, "matches_saved_oracle": abs(final - saved) < 1e-9,
                     "events": events}
        print(seed, "match", abs(final - saved) < 1e-9, "n_events", len(events),
              "gap", np.mean([e["logit_gap_returning_minus_trained"] for e in events]).round(3),
              "grp_full", np.mean([e["group_term_full"] for e in events]).round(3),
              "grp0_full", np.mean([e["group_term_full_if_zero_rows"] for e in events]).round(3),
              "grp_restr", np.mean([e["group_term_restricted"] for e in events]).round(3), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

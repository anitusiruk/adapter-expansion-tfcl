"""Run one learner/policy on one stream and save a JSON record.

Policies:
  frozen        no training; prototypes from memory (and from the full stream)
  single        one package trained throughout
  oracle        spawn at every true segment change (uses segment ids; reference)
  periodic:K    spawn every K batches
  trigger:NAME  task-free trigger from src.tfcl.triggers (see that module)
  shadow        single package, but every trigger statistic is recorded per batch
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.tfcl.data import make_stream
from src.tfcl.learner import Learner
from src.tfcl.model import Encoder


def seed_all(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def frozen_full_eval(enc, stream, dev, segs, n_labels):
    """SimpleCIL-style NCM: prototypes = means over every streamed example (frozen features)."""
    xs = [x for b in stream for x in b["x"]]
    ys = torch.tensor([y for b in stream for y in b["y"]], device=enc.device)
    f = torch.nn.functional.normalize(enc.features_batched(xs), dim=-1)
    protos = torch.zeros(n_labels, f.shape[1], device=f.device).index_add_(0, ys, f)
    has = torch.bincount(ys, minlength=n_labels) > 0
    out = {}
    for s in segs:
        g = enc.features_batched(dev[s]["x"])
        sc = (torch.nn.functional.normalize(g, dim=-1) @ torch.nn.functional.normalize(protos, dim=-1).t())
        pred = sc.masked_fill(~has, -2).argmax(1).cpu()
        out[s] = float((pred == torch.tensor(dev[s]["y"])).float().mean())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--replay", type=int, default=16)
    ap.add_argument("--mem", type=int, default=512)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--head-init", default="fresh")
    ap.add_argument("--lora-init", default="fresh")
    ap.add_argument("--no-lora", action="store_true")
    ap.add_argument("--trigger-args", default="{}")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    seed_all(a.seed)
    t0 = time.time()
    stream, dev, n_labels = make_stream(a.stream, a.seed)
    enc = Encoder(a.model, max_length=a.max_length)
    L = Learner(enc, n_labels, a.seed, lr=a.lr, steps=a.steps, replay=a.replay, mem_size=a.mem,
                train_lora=not a.no_lora, head_init=a.head_init, lora_init=a.lora_init)
    trig = None
    if a.policy.startswith("trigger:") or a.policy == "shadow":
        from src.tfcl.triggers import make_trigger
        name = a.policy.split(":", 1)[1] if a.policy != "shadow" else "shadow"
        trig = make_trigger(name, L, **json.loads(a.trigger_args))
    # random: as many spawns as true segment changes (= oracle count), at uniformly random
    # steps >= 8 (the trigger warmup), drawn from a dedicated generator.
    n_changes = sum(1 for i in range(1, len(stream)) if stream[i]["seg"] != stream[i - 1]["seg"])
    random_steps = set(np.random.default_rng(a.seed + 555).choice(
        np.arange(8, len(stream)), size=n_changes, replace=False).tolist())
    seg_order, checkpoints, events, records = [], [], [], []
    for i, b in enumerate(stream):
        prev = stream[i - 1]["seg"] if i else None
        if b["seg"] not in seg_order:
            seg_order.append(b["seg"])
        spawn = False
        if a.policy == "oracle" and i > 0 and b["seg"] != prev:
            spawn = True
        elif a.policy.startswith("periodic:") and i > 0 and i % int(a.policy.split(":")[1]) == 0:
            spawn = True
        elif a.policy == "random" and i in random_steps:
            spawn = True
        elif trig is not None:
            rec = trig.decide(b["x"], b["y"])
            rec.update(step=i, seg=b["seg"])
            records.append(rec)
            spawn = bool(rec.get("spawn", False))
        if spawn:
            L.spawn()
            events.append({"step": i, "seg": b["seg"], "boundary": bool(i > 0 and b["seg"] != prev)})
        if a.policy == "frozen":
            for y in b["y"]:
                L.seen[y] = True
            for x, y in zip(b["x"], b["y"]):
                L.mem.add(x, y, 0)
            L.t += 1
        else:
            L.observe(b["x"], b["y"])
        if trig is not None:
            trig.after_update(b["x"], b["y"])
        end_of_seg = i == len(stream) - 1 or stream[i + 1]["seg"] != b["seg"]
        if end_of_seg:
            ev = L.evaluate(dev, sorted(dev) if a.stream.endswith('_iid') else sorted(set(seg_order)))
            checkpoints.append({"step": i, "seg": b["seg"], "seen": sorted(set(seg_order)), "acc": ev})
            if a.stream.endswith('_iid'):
                checkpoints[-1]["seen"] = sorted(dev)
            print(f"[{a.stream} {a.policy} s{a.seed}] step {i} seg {b['seg']} packages {len(L.packages)} "
                  + " ".join(f"{s}:{v['proto_mean']:.3f}" for s, v in ev.items()), flush=True)
    segs = sorted(dev) if a.stream.endswith('_iid') else sorted(set(seg_order))
    final = checkpoints[-1]["acc"]
    summary = {k: float(np.mean([final[s][k] for s in segs])) for k in final[segs[0]]}
    if a.policy == "frozen":
        ff = frozen_full_eval(enc, stream, dev, segs, n_labels)
        summary["frozen_full_ncm"] = float(np.mean(list(ff.values())))
    rec = {"args": vars(a), "n_batches": len(stream), "packages": len(L.packages), "events": events,
           "checkpoints": checkpoints, "final_mean": summary, "trigger_records": records,
           "seconds": time.time() - t0,
           "trainable_params_per_package": int(sum(p.numel() for p in L.packages[0].params()))}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    print("FINAL", a.stream, a.policy, a.seed, json.dumps(summary), f"{rec['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()

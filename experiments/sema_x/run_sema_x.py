"""External-system check (extension 5): the value of expansion inside SEMA's official code.

SEMA (Wang et al., CVPR 2025; github.com/huiyiwang01/SEMA-CL, commit 5a45512) is run with its
published CIFAR-100 configuration (exps/sema_cifar.json) and its own training loop. Only the
expansion decision and the freezing of functional adapters are varied:

  sema    official (representation-descriptor z-score > exp_threshold triggers expansion)
  noexp   exp_threshold = +inf: never expands; adapters frozen after task 1 (first-session
          adaptation + linear head; the reference of SEMA's own "w/o expansion" ablation)
  single  never expands, and functional adapters are NOT frozen at the end of a task: one
          adapter set keeps training on every task (the "single package" of our study)
  expall  exp_threshold = -inf: expands at every task (oracle expansion at true boundaries)

Every variant is scored after every task under four inference rules on the official test split:
  linear     SEMA's linear head (exemplar-free; old-class rows are never refreshed) -- the
             metric SEMA reports
  ncm_ef     cosine nearest-class-mean with class means computed from all training data of a
             task at the end of that task and never recomputed (APER/EASE-style exemplar-free)
  ncm_ex20   cosine NCM with class means recomputed under the current network from 20 stored
             exemplars per class (same exemplar indices for all variants of a seed)
  ncm_full   cosine NCM with all seen classes' means recomputed from all their training data
             (refresh upper bound; final task only)

Per-epoch test evaluation inside SEMA's training loop is disabled (it only feeds a progress
bar), and TF32 matrix multiplication is enabled for speed (identically for every variant);
nothing else in training is changed.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

SEMA_DIR = os.environ.get("SEMA_DIR", "/workspace/ext_src/SEMA-CL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["sema", "noexp", "single", "expall"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", default="exps/sema_cifar.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-ex", type=int, default=20)
    ap.add_argument("--max-tasks", type=int, default=0, help="debug: stop after this many tasks")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    if os.path.exists(out):
        raise SystemExit(f"refusing to overwrite {out}")
    os.chdir(SEMA_DIR)
    sys.path.insert(0, SEMA_DIR)
    import trainer as T
    from backbone.sema_block import SEMAModules
    from models.sema import Learner as SEMALearner
    from torch.utils.data import DataLoader
    from utils.data_manager import DataManager

    args = json.load(open(a.config))
    args["seed"] = a.seed
    if a.variant in ("noexp", "single"):
        args["exp_threshold"] = float("inf")
    elif a.variant == "expall":
        args["exp_threshold"] = float("-inf")

    if a.variant == "single":
        def end_of_task_training(self):  # freeze descriptors only; functional adapters keep training
            self.freeze_rd()
            self.reset_newly_added_status()
            self.added_for_task = False
        SEMAModules.end_of_task_training = end_of_task_training

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    def _detect_outlier(self, detect_loader, train_loader, test_loader, added):
        # SEMA's own logic verbatim, except that the detection forward runs under no_grad
        # (its outputs only decide whether to expand; this avoids a ~14 GB autograd graph).
        is_added = False
        for i, (_, inputs, targets) in enumerate(detect_loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            with torch.no_grad():
                model_outcome = self._network(inputs)
            added_record = model_outcome["added_record"]
            if sum(added_record) > 0:
                added += 1
                is_added = True
                for module in self._network.backbone.modules():
                    if isinstance(module, SEMAModules):
                        module.detecting_outlier = False
                self._train_new(train_loader, test_loader)
                for module in self._network.backbone.modules():
                    if isinstance(module, SEMAModules):
                        module.detecting_outlier = True
                for module in self._network.backbone.modules():
                    if isinstance(module, SEMAModules):
                        module.freeze_functional()
                        module.freeze_rd()
                        module.reset_newly_added_status()
        if is_added:
            return self._detect_outlier(detect_loader, train_loader, test_loader, added)
        return added
    SEMALearner._detect_outlier = _detect_outlier

    T._set_random(a.seed)
    T._set_device(args)
    dm = DataManager(args["dataset"], args["shuffle"], args["seed"], args["init_cls"], args["increment"], args)
    args["nb_classes"], args["nb_tasks"] = dm.nb_classes, dm.nb_tasks
    model = SEMALearner(args)
    model._compute_accuracy = lambda *_a, **_k: 0.0  # progress-bar only
    dev = model._device

    @torch.no_grad()
    def feats(loader):
        model._network.eval()
        fs, ys = [], []
        for _, x, y in loader:
            fs.append(F.normalize(model._network(x.to(dev))["features"].float(), dim=-1).cpu())
            ys.append(y)
        return torch.cat(fs), torch.cat(ys)

    def loader(classes, source, mode, idx=None):
        ds = dm.get_dataset(np.asarray(classes), source=source, mode=mode)
        if idx is not None:
            ds = torch.utils.data.Subset(ds, idx)
        return DataLoader(ds, batch_size=128, shuffle=False, num_workers=8)

    def means(f, y, n):
        m = torch.zeros(n, f.shape[1]).index_add_(0, y, f)
        return F.normalize(m, dim=-1)

    def ncm_acc(m, f, y, seen):
        s = f @ m.t()
        s[:, seen:] = -2
        return float((s.argmax(1) == y).float().mean() * 100)

    rng = np.random.RandomState(a.seed + 7)
    ef_means = torch.zeros(dm.nb_classes, 768)
    ex_idx = {}
    rec = {"args": vars(a), "sema_args": {k: (str(v) if isinstance(v, float) and not np.isfinite(v) else v)
                                           for k, v in args.items()}, "tasks": []}
    t0 = time.time()
    n_tasks = dm.nb_tasks if not a.max_tasks else a.max_tasks
    for task in range(n_tasks):
        model.incremental_train(dm)
        lo, hi = model._known_classes, model._total_classes
        # exemplar-free class means of the new classes, computed now and never recomputed
        tr_ds = dm.get_dataset(np.arange(lo, hi), source="train", mode="test")
        f_new, y_new = feats(DataLoader(tr_ds, batch_size=128, shuffle=False, num_workers=8))
        ef_means[lo:hi] = means(f_new, y_new, dm.nb_classes)[lo:hi]
        # choose exemplar indices for the new classes (positions within tr_ds)
        y_np = y_new.numpy()
        pos = []
        for c in range(lo, hi):
            cand = np.where(y_np == c)[0]
            pos += rng.choice(cand, size=min(a.n_ex, len(cand)), replace=False).tolist()
        ex_idx[task] = (lo, hi, pos)
        # evaluation
        f_te, y_te = feats(model.test_loader)
        cnn_accy, _ = model.eval_task()
        ex_f, ex_y = [], []
        for tk, (l2, h2, p2) in ex_idx.items():
            f2, y2 = feats(loader(np.arange(l2, h2), "train", "test", p2))
            ex_f.append(f2); ex_y.append(y2)
        ex_m = means(torch.cat(ex_f), torch.cat(ex_y), dm.nb_classes)
        r = {"task": task, "seen": hi, "linear": float(cnn_accy["top1"]),
             "ncm_ef": ncm_acc(ef_means, f_te, y_te, hi), "ncm_ex20": ncm_acc(ex_m, f_te, y_te, hi),
             "adapters": [m.num_adapters for m in model._network.backbone.modules() if isinstance(m, SEMAModules)],
             "seconds": time.time() - t0}
        if task == n_tasks - 1:
            f_all, y_all = feats(loader(np.arange(0, hi), "train", "test"))
            r["ncm_full"] = ncm_acc(means(f_all, y_all, dm.nb_classes), f_te, y_te, hi)
        rec["tasks"].append(r)
        print(json.dumps(r), flush=True)
        model.after_task()
    last = rec["tasks"][-1]
    rec["final"] = {k: last[k] for k in ("linear", "ncm_ef", "ncm_ex20", "ncm_full") if k in last}
    rec["avg_inc"] = {k: float(np.mean([t[k] for t in rec["tasks"]])) for k in ("linear", "ncm_ef", "ncm_ex20")}
    rec["seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rec, open(out, "w"), indent=1, default=str)
    print("FINAL", a.variant, a.seed, json.dumps(rec["final"]), f"{rec['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()

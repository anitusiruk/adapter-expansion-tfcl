"""Task-free expansion triggers, computed on the newest package before it trains.

Every trigger reduces to a scalar statistic per incoming batch plus a common
decision rule: after ``warmup`` updates of the newest package, spawn when the
statistic exceeds ``tau`` on ``confirm`` consecutive eligible batches.
Thresholds are calibrated on a no-shift (i.i.d.-shuffled) version of the
stream; see experiments/tfcl_calibrate.py.

Statistics (stylised versions of published signals; not full reproductions):
  label_novel    fraction of batch labels never observed before (trivial).
  loss_z         z-score of the incoming loss against a sliding window
                 (loss-spike/plateau triggers: LACE, Online-LoRA).
  repr_z         z-score of a label-free diagonal-Mahalanobis feature
                 distance (distribution-shift descriptors: SEMA).
  interf_logit   prospective increase of protected-memory cross-entropy after
                 a virtual update (lookahead forgetting: CABLE, AdamW-I1).
  fresh_util     cross-fitted query-loss advantage of a fresh package over
                 updating the current one (counterfactual utility: CAU).
  conflict_z     z-score of the incoming loss restricted to examples whose label
                 is *established*: first observed >= MATURE (= warmup) batches
                 earlier (label-novelty-corrected loss spike; proposed).
                 Undefined (no fire) with < 4 such examples.
  label_surprise mean over the batch of [loss(x, y) - baseline_y], where baseline_y
                 is the mean of the last WINDOW batch-mean losses of label y under
                 the current package (needs >= 3 prior occurrences of y; batch
                 needs >= 4 such examples, otherwise undefined = no fire).
                 Label-conditional, so newly learned labels (falling loss) do not
                 register; known labels whose mapping changes do (proposed v3).
  interf_proto   prospective increase of protected-memory *prototype* loss
                 after a virtual update; prototypes are recomputed from the
                 updated features, so the statistic is invariant to the
                 linear head (head-equalized; proposed).
All statistics are recorded together in ``shadow`` mode.
"""
import math
import random
from collections import deque

import torch
import torch.nn.functional as F

from src.tfcl.learner import Package

ALL = ["label_novel", "loss_z", "repr_z", "interf_logit", "fresh_util", "interf_proto", "conflict_z",
       "label_surprise"]
PROBE = 64
PROTO_TEMP = 0.05
MATURE = 8  # equal to the default warmup; not tuned


class Stats:
    def __init__(self, L, window=20):
        self.L = L
        self.win = {"loss": deque(maxlen=window), "repr": deque(maxlen=window),
                    "conflict": deque(maxlen=window)}
        self.window = window
        self.label_hist = {}
        self.mu = None
        self.var = None
        self.n = 0

    # ---------------------------------------------------------------- helpers
    def seen_plus(self, ys):
        s = self.L.seen.clone()
        s[torch.tensor(ys, device=s.device)] = True
        return s

    def probe(self, p):
        own = self.L.mem.owned(self.L.pid(p))
        rng = self.L.probe_rng
        if len(own) > PROBE:
            own = rng.sample(own, PROBE)
        return [it[0] for it in own], [it[1] for it in own]

    @torch.no_grad()
    def head_loss(self, p, xs, ys, seen, reduction="mean"):
        return self.L.loss(p, xs, ys, seen=seen, reduction=reduction)

    @torch.no_grad()
    def proto_loss(self, p, xs, ys):
        """Leave-one-out cosine-prototype log-loss on (xs, ys) under package p."""
        f = F.normalize(self.L.enc.features_batched(xs, p.lora), dim=-1)
        y = torch.tensor(ys, device=f.device)
        n_lab = self.L.n_labels
        sums = torch.zeros(n_lab, f.shape[1], device=f.device).index_add_(0, y, f)
        cnt = torch.bincount(y, minlength=n_lab).float()
        # leave-one-out own-class prototype
        own = (sums[y] - f)
        own_cnt = cnt[y] - 1
        valid = own_cnt > 0
        protos_norm = F.normalize(sums, dim=-1)
        logits = f @ protos_norm.t()
        own_sim = (f * F.normalize(own, dim=-1)).sum(-1)
        logits[torch.arange(len(y)), y] = own_sim
        logits = logits.masked_fill(cnt[None, :] == 0, float("-inf")) / PROTO_TEMP
        lp = -F.log_softmax(logits, -1)[torch.arange(len(y)), y]
        if valid.sum() == 0:
            return float("nan")
        return float(lp[valid].mean())

    def virtual(self, p, xs, ys, seen, fn):
        """Run fn() after a virtual update of p on (xs, ys); restore exactly."""
        snap = p.snapshot()
        rng_state = self.L.probe_rng.getstate()
        try:
            self.L.step(p, xs, ys, rng=self.L.probe_rng, seen=seen)
            return fn()
        finally:
            p.restore(snap)
            self.L.probe_rng.setstate(rng_state)

    @staticmethod
    def z(window, v):
        if len(window) < 5:
            return float("nan")
        m = sum(window) / len(window)
        sd = math.sqrt(sum((w - m) ** 2 for w in window) / (len(window) - 1)) + 1e-8
        return (v - m) / sd

    # ---------------------------------------------------------------- statistics
    def compute(self, xs, ys, which):
        L, p = self.L, self.L.active
        out = {}
        seen = self.seen_plus(ys)
        novel = [not bool(L.seen[y]) for y in ys]
        if "label_novel" in which:
            out["label_novel"] = sum(novel) / len(ys)
        if any(k in which for k in ("loss_z", "conflict_z", "decomp")):
            with torch.no_grad():
                f = L.enc.features(xs, p.lora)
                logits = (f @ p.w.t() + p.b).masked_fill(~seen, float("-inf"))
                lsm = F.log_softmax(logits, -1)
                y = torch.tensor(ys, device=f.device)
                nll = -lsm[torch.arange(len(ys)), y]
                old_mask = L.seen.clone()
                # label-group decomposition: group = {previously seen} vs {new in this batch}
                g_new = seen & ~old_mask
                lse_all = torch.logsumexp(logits, -1)
                in_new = torch.tensor(novel, device=f.device)
                grp = torch.where(in_new[:, None], g_new[None, :], old_mask[None, :])
                lse_grp = torch.logsumexp(logits.masked_fill(~grp, float("-inf")), -1)
                group_term = (lse_all - lse_grp)          # -log P(group(y))
                within = nll - group_term                  # -log p(y | group(y))
            loss = float(nll.mean())
            out["loss"] = loss
            out["loss_group_term"] = float(group_term.mean())
            out["loss_within_term"] = float(within.mean())
            out["loss_z"] = self.z(self.win["loss"], loss)
            self._last_loss = loss
            known = torch.tensor([L.first_seen[y] >= 0 and L.t - L.first_seen[y] >= MATURE for y in ys],
                                 device=f.device)
            if int(known.sum()) >= 4:
                c = float(nll[known].mean())
                out["conflict_loss"] = c
                out["conflict_z"] = self.z(self.win["conflict"], c)
                self._last_conflict = c
            else:
                out["conflict_loss"] = out["conflict_z"] = float("nan")
            # label-conditional surprise
            per_label = {}
            for i, yy in enumerate(ys):
                per_label.setdefault(yy, []).append(float(nll[i]))
            sur = []
            for yy, vals in per_label.items():
                h = self.label_hist.get(yy)
                if h is not None and len(h) >= 3:
                    base = sum(h) / len(h)
                    sur += [v - base for v in vals]
            out["label_surprise"] = sum(sur) / len(sur) if len(sur) >= 4 else float("nan")
            self._last_label_means = {yy: sum(v) / len(v) for yy, v in per_label.items()}
        if "repr_z" in which:
            with torch.no_grad():
                f = L.enc.features(xs, p.lora)
            if self.mu is None:
                d = float("nan")
            else:
                d = float(((f - self.mu) ** 2 / (self.var + 1e-6)).mean())
            out["repr_dist"] = d
            out["repr_z"] = self.z(self.win["repr"], d) if not math.isnan(d) else float("nan")
            self._last_repr = d
        need_probe = any(k in which for k in ("interf_logit", "interf_proto"))
        if need_probe:
            px, py = self.probe(p)
            if len(px) >= 8:
                pseen = seen
                b_logit = float(self.head_loss(p, px, py, pseen))
                b_proto = self.proto_loss(p, px, py)
                a_logit, a_proto = self.virtual(
                    p, xs, ys, seen,
                    lambda: (float(self.head_loss(p, px, py, pseen)), self.proto_loss(p, px, py)))
                out["interf_logit"] = a_logit - b_logit
                out["interf_proto"] = a_proto - b_proto
            else:
                out["interf_logit"] = out["interf_proto"] = float("nan")
        if "fresh_util" in which:
            out["fresh_util"] = self.fresh_util(p, xs, ys, seen)
        return out

    def fresh_util(self, p, xs, ys, seen):
        L = self.L
        if len(xs) < 4:
            return float("nan")
        gen = torch.Generator().manual_seed(L.probe_rng.randrange(2 ** 31))
        fresh = Package(L.enc, L.n_labels, L.lr, gen, L.t)
        if not L.train_lora:
            for t in fresh.lora:
                t.requires_grad_(False)
            fresh.opt = torch.optim.AdamW([fresh.w, fresh.b], lr=L.lr, weight_decay=0.0)
        us = []
        idx = list(range(len(xs)))
        for sup_par in (0, 1):
            sup = [i for i in idx if i % 2 == sup_par]
            qry = [i for i in idx if i % 2 != sup_par]
            sx, sy = [xs[i] for i in sup], [ys[i] for i in sup]
            qx, qy = [xs[i] for i in qry], [ys[i] for i in qry]
            reuse_after = self.virtual(p, sx, sy, seen, lambda: float(self.head_loss(p, qx, qy, seen)))
            # fresh package has no memory of its own -> its virtual step uses no replay
            fresh_snap = fresh.snapshot()
            st = L.probe_rng.getstate()
            try:
                L.step(fresh, sx, sy, rng=L.probe_rng, seen=seen)
                fresh_after = float(self.head_loss(fresh, qx, qy, seen))
            finally:
                fresh.restore(fresh_snap)
                L.probe_rng.setstate(st)
            us.append(reuse_after - fresh_after)
        return sum(us) / len(us)

    def update(self, xs, ys, alarmed=()):
        """Update sliding windows/feature moments after the real update. Values of a
        statistic that is currently in alarm are not added to its own baseline window
        (standard change-detection practice; otherwise a spike masks its confirmation)."""
        L, p = self.L, self.L.active
        for key, attr, stat in (("loss", "_last_loss", "loss_z"), ("conflict", "_last_conflict", "conflict_z"),
                                ("repr", "_last_repr", "repr_z")):
            if hasattr(self, attr):
                v = getattr(self, attr)
                if not math.isnan(v) and stat not in alarmed:
                    self.win[key].append(v)
                delattr(self, attr)
        if hasattr(self, "_last_label_means"):
            if "label_surprise" not in alarmed:
                for yy, v in self._last_label_means.items():
                    self.label_hist.setdefault(yy, deque(maxlen=self.window)).append(v)
            del self._last_label_means
        with torch.no_grad():
            f = L.enc.features(xs, p.lora)
        m, v = f.mean(0), f.var(0, unbiased=False)
        if self.mu is None:
            self.mu, self.var = m, v
        else:
            a = 0.1
            self.var = (1 - a) * (self.var + a * (m - self.mu) ** 2) + a * v
            self.mu = (1 - a) * self.mu + a * m

    def reset(self):
        for w in self.win.values():
            w.clear()
        self.label_hist = {}
        self.mu = self.var = None


class Trigger:
    def __init__(self, L, name, tau=None, confirm=2, warmup=8, record=None):
        self.L, self.name, self.tau, self.confirm, self.warmup = L, name, tau, confirm, warmup
        self.stats = Stats(L)
        self.record = record or ([name] if name != "shadow" else ALL)
        self.which = set(self.record) | {"loss_z", "repr_z", "conflict_z", "label_surprise"}  # windows always maintained
        self.streak = 0

    def decide(self, xs, ys):
        rec = self.stats.compute(xs, ys, self.which)
        p = self.L.active
        eligible = p.updates >= self.warmup and len(self.L.mem.owned(self.L.pid(p))) >= 8
        rec["eligible"] = eligible
        rec["spawn"] = False
        if self.name == "shadow" or self.tau is None:
            return rec
        v = rec.get(self.name, float("nan"))
        fire = eligible and not math.isnan(v) and v > self.tau
        self._alarm = (self.name,) if fire else ()
        self.streak = self.streak + 1 if fire else 0
        if fire and self.streak >= self.confirm:
            rec["spawn"] = True
            self.streak = 0
            self._spawned = True
        return rec

    def after_update(self, xs, ys):
        if getattr(self, "_spawned", False):
            self.stats.reset()
            self._spawned = False
        self.stats.update(xs, ys, alarmed=getattr(self, "_alarm", ()))


def make_trigger(name, L, **kw):
    return Trigger(L, name, **kw)

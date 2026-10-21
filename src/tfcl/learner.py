"""Task-free expand-and-freeze adapter pool with a fixed global replay budget.

Protocol (shared by every learner in the study):
  * Only the newest package trains; older packages are frozen (as in
    EASE/SEMA-style expansion). A package = LoRA factors + linear head.
  * Each incoming batch receives ``steps`` AdamW steps on [batch + replay
    sample of the trained package's own memory items].
  * One global reservoir of ``mem_size`` items (fixed total budget, independent
    of the number of packages). Items are tagged with the package that saw them.
  * Training loss: cross-entropy over labels the learner has observed so far.
  * Inference: cosine prototype classifier. Prototypes are class means of
    memory features under each package; the score of class c is the mean over
    packages of cos(f_p(x), proto_p(c)) (equivalent to NCM on the normalized
    concatenation of package features, as in EASE). A max-over-packages
    variant is reported as secondary.
"""
import copy
import random

import numpy as np
import torch
import torch.nn.functional as F

from src.tfcl.model import cosine_scores


class Package:
    def __init__(self, enc, n_labels, lr, gen, created, head_init=None, lora_init=None):
        self.lora = lora_init if lora_init is not None else enc.new_lora(gen)
        if head_init is None:
            self.w = torch.zeros(n_labels, enc.hidden, device=enc.device, requires_grad=True)
            self.b = torch.zeros(n_labels, device=enc.device, requires_grad=True)
        else:
            self.w = head_init[0].detach().clone().requires_grad_()
            self.b = head_init[1].detach().clone().requires_grad_()
        self.opt = torch.optim.AdamW(self.params(), lr=lr, weight_decay=0.0)
        self.created = created
        self.updates = 0

    def params(self):
        return list(self.lora) + [self.w, self.b]

    def snapshot(self):
        return ([p.detach().clone() for p in self.params()], copy.deepcopy(self.opt.state_dict()),
                self.updates)

    def restore(self, snap):
        vals, opt_state, upd = snap
        with torch.no_grad():
            for p, v in zip(self.params(), vals):
                p.copy_(v)
        self.opt.load_state_dict(opt_state)
        self.updates = upd


class Reservoir:
    def __init__(self, size, seed):
        self.size, self.items, self.seen = size, [], 0
        self.rng = random.Random(seed)

    def add(self, x, y, owner):
        self.seen += 1
        if len(self.items) < self.size:
            self.items.append((x, y, owner))
        else:
            j = self.rng.randrange(self.seen)
            if j < self.size:
                self.items[j] = (x, y, owner)

    def owned(self, owner):
        return [it for it in self.items if it[2] == owner]


class Learner:
    def __init__(self, enc, n_labels, seed, lr=1e-3, steps=2, replay=16, mem_size=512,
                 train_lora=True, head_init="fresh", lora_init="fresh"):
        self.enc, self.n_labels, self.lr, self.steps, self.replay = enc, n_labels, lr, steps, replay
        self.train_lora, self.head_init, self.lora_init = train_lora, head_init, lora_init
        self.gen = torch.Generator().manual_seed(seed)
        self.replay_rng = random.Random(seed + 1)
        self.probe_rng = random.Random(seed + 2)  # used only by counterfactual probes
        self.mem = Reservoir(mem_size, seed + 3)
        self.seen = torch.zeros(n_labels, dtype=torch.bool, device=enc.device)
        self.first_seen = [-1] * n_labels  # stream step at which each label first arrived
        self.packages = []
        self.t = 0
        self.spawn()

    # ------------------------------------------------------------------ structure
    def spawn(self):
        prev = self.packages[-1] if self.packages else None
        head = (prev.w, prev.b) if (prev is not None and self.head_init == "inherit") else None
        lora = ([t.detach().clone().requires_grad_(t.requires_grad) for t in prev.lora]
                if (prev is not None and self.lora_init == "inherit") else None)
        p = Package(self.enc, self.n_labels, self.lr, self.gen, self.t, head_init=head, lora_init=lora)
        if not self.train_lora:
            for t in p.lora:
                t.requires_grad_(False)
            p.opt = torch.optim.AdamW([p.w, p.b], lr=self.lr, weight_decay=0.0)
        self.packages.append(p)
        return p

    @property
    def active(self):
        return self.packages[-1]

    def pid(self, p):
        return self.packages.index(p)

    # ------------------------------------------------------------------ training
    def loss(self, p, xs, ys, seen=None, reduction="mean"):
        seen = self.seen if seen is None else seen
        f = self.enc.features(xs, p.lora, grad=True)
        logits = (f @ p.w.t() + p.b).masked_fill(~seen, float("-inf"))
        y = torch.tensor(ys, device=self.enc.device)
        return F.cross_entropy(logits, y, reduction=reduction)

    def _replay_sample(self, p, rng, k):
        if k == 0 or p not in self.packages:  # a probe-only fresh package owns no memory
            return [], []
        own = self.mem.owned(self.pid(p))
        if not own:
            return [], []
        pick = [own[rng.randrange(len(own))] for _ in range(k)]
        return [it[0] for it in pick], [it[1] for it in pick]

    def step(self, p, xs, ys, rng=None, seen=None):
        """``steps`` AdamW updates of package p on xs (+ replay of p's memory)."""
        rng = rng or self.replay_rng
        for _ in range(self.steps):
            rx, ry = self._replay_sample(p, rng, self.replay)
            p.opt.zero_grad(set_to_none=True)
            loss = self.loss(p, list(xs) + rx, list(ys) + ry, seen=seen)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(p.params(), 1.0)
            p.opt.step()
        p.updates += 1

    def observe(self, xs, ys):
        """Real learning on an incoming batch with the newest package."""
        for y in ys:
            self.seen[y] = True
            if self.first_seen[y] < 0:
                self.first_seen[y] = self.t
        p = self.active
        self.step(p, xs, ys)
        for x, y in zip(xs, ys):
            self.mem.add(x, y, self.pid(p))
        self.t += 1

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def prototypes(self, p, items=None):
        items = self.mem.items if items is None else items
        xs = [it[0] for it in items]
        ys = torch.tensor([it[1] for it in items], device=self.enc.device)
        f = self.enc.features_batched(xs, p.lora)
        protos = torch.zeros(self.n_labels, f.shape[1], device=f.device)
        protos.index_add_(0, ys, F.normalize(f, dim=-1))
        has = torch.bincount(ys, minlength=self.n_labels) > 0
        return protos, has

    @torch.no_grad()
    def predict(self, xs, packages=None):
        """proto_mean: mean cosine over packages, prototypes from all memory (EASE-like).
        proto_max: max over packages of the same scores.
        proto_own: each package scores only classes it owns in memory, using prototypes from
        its own items; prediction = best (package, class) pair (routing by nearest prototype).
        linear_last: newest package's linear head (diagnostic)."""
        packages = self.packages if packages is None else packages
        mean_s = max_s = own_s = None
        for p in packages:
            protos, has = self.prototypes(p)
            own_items = self.mem.owned(self.pid(p))
            f = self.enc.features_batched(xs, p.lora)
            s = cosine_scores(f, protos).masked_fill(~has, -2.0)
            mean_s = s if mean_s is None else mean_s + s
            max_s = s if max_s is None else torch.maximum(max_s, s)
            if own_items:
                op, ohas = self.prototypes(p, own_items)
                so = cosine_scores(f, op).masked_fill(~ohas, -2.0)
                own_s = so if own_s is None else torch.maximum(own_s, so)
        lin_logits = (f @ packages[-1].w.t() + packages[-1].b).masked_fill(~self.seen, float("-inf"))
        return {"proto_mean": mean_s.argmax(1).cpu(), "proto_max": max_s.argmax(1).cpu(),
                "proto_own": own_s.argmax(1).cpu(), "linear_last": lin_logits.argmax(1).cpu()}

    def evaluate(self, dev, segs):
        xs, ys, tags = [], [], []
        for s in segs:
            xs += dev[s]["x"]; ys += dev[s]["y"]; tags += [s] * len(dev[s]["y"])
        preds = self.predict(xs)
        y = torch.tensor(ys)
        tags = np.array(tags)
        out = {}
        for s in segs:
            m = torch.from_numpy(tags == s)
            out[s] = {k: float((v[m] == y[m]).float().mean()) for k, v in preds.items()}
        return out

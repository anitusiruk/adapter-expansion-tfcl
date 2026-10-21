"""Exemplar-free prototype refresh by semantic drift compensation (extension 4).

Extension 2 attributes the benefit of expansion in the exemplar-free regime to stale class
prototypes: a package that keeps training moves its features, while prototypes recorded at
observation time do not move with them. Semantic drift compensation (SDC; Yu et al., CVPR
2020) refreshes old prototypes *without exemplars* by interpolating the drift of current
data at each prototype:

    delta_i  = z_i^new - z_i^old                      (current data, before/after the update)
    w_ic     = exp(-||z_i^old - mu_c||^2 / (2 sigma^2))
    mu_c    += sum_i w_ic delta_i / sum_i w_ic

with L2-normalised embeddings and sigma = 0.3 (the paper's default; 0.2 for its
from-scratch CIFAR-100 runs). In a task-free stream there are no tasks, so the compensation
is applied after every update of the active package, using the incoming batch as the
current data (drift vectors are additive, so per-batch compensation accumulates exactly the
per-step drift that SDC sums over tasks). Frozen packages do not move and need no
compensation.

The mixin sits on top of ExemplarFreeMixin and changes no training computation: it adds one
no-grad forward pass of the incoming batch under the active package before its update and
reports extra inference rules computed from separately maintained prototype sums:

  proto_ef_sdc     sigma = 0.3   (primary)
  proto_ef_sdc02   sigma = 0.2   (sensitivity)
  proto_ef_sdcinf  sigma = inf   (uniform mean drift: global translation; sensitivity)

Scores are computed exactly as proto_ef (mean cosine over packages that hold a prototype).
"""
import math

import torch
import torch.nn.functional as F

from src.tfcl.model import cosine_scores

SIGMAS = {"proto_ef_sdc": 0.3, "proto_ef_sdc02": 0.2, "proto_ef_sdcinf": math.inf}


class SDCMixin:
    def _sdc_sync(self):
        if not hasattr(self, "sdc_sum"):
            self.sdc_sum = {k: [] for k in SIGMAS}
            self.sdc_cnt = []
            self.sdc_stats = {"updates": 0, "low_weight": 0, "proto_updates": 0}
        while len(self.sdc_cnt) < len(self.packages):
            for k in SIGMAS:
                self.sdc_sum[k].append(torch.zeros(self.n_labels, self.enc.hidden, device=self.enc.device))
            self.sdc_cnt.append(torch.zeros(self.n_labels, device=self.enc.device))

    def observe(self, xs, ys):
        self._sdc_sync()
        i = len(self.packages) - 1
        train = self._sdc_will_train() if hasattr(self, "_sdc_will_train") else True
        with torch.no_grad():
            z_old = F.normalize(self.enc.features(xs, self.active.lora), dim=-1) if train else None
        super().observe(xs, ys)  # trains the active package, then records proto_ef sums
        self._sdc_sync()
        y = torch.tensor(ys, device=self.enc.device)
        with torch.no_grad():
            z_new = F.normalize(self.enc.features(xs, self.packages[i].lora), dim=-1)
            has = self.sdc_cnt[i] > 0
            if train and has.any():
                delta = z_new - z_old                                         # (B, H)
                for k, sigma in SIGMAS.items():
                    mu = self.sdc_sum[k][i][has] / self.sdc_cnt[i][has, None]  # (C, H)
                    if math.isinf(sigma):
                        w = torch.ones(len(xs), mu.shape[0], device=mu.device)
                    else:
                        d2 = torch.cdist(z_old, mu).pow(2)                     # (B, C)
                        w = torch.exp(-d2 / (2 * sigma ** 2))
                    wsum = w.sum(0)                                            # (C,)
                    shift = (w.t() @ delta) / wsum.clamp(min=1e-12)[:, None]
                    shift = torch.where(wsum[:, None] > 1e-12, shift, torch.zeros_like(shift))
                    self.sdc_sum[k][i][has] += shift * self.sdc_cnt[i][has, None]
                    if k == "proto_ef_sdc":
                        self.sdc_stats["proto_updates"] += int(has.sum())
                        self.sdc_stats["low_weight"] += int((wsum < 1e-3).sum())
                self.sdc_stats["updates"] += 1
            # record the new examples under every package (as proto_ef does)
            for j, p in enumerate(self.packages):
                f = z_new if j == i else F.normalize(self.enc.features(xs, p.lora), dim=-1)
                for k in SIGMAS:
                    self.sdc_sum[k][j].index_add_(0, y, f)
                self.sdc_cnt[j].index_add_(0, y, torch.ones(len(ys), device=self.enc.device))

    @torch.no_grad()
    def predict(self, xs, packages=None):
        out = super().predict(xs, packages)
        self._sdc_sync()
        feats = {}
        for k in SIGMAS:
            tot = num = None
            for i, p in enumerate(self.packages):
                has = self.sdc_cnt[i] > 0
                if not has.any():
                    continue
                if i not in feats:
                    feats[i] = self.enc.features_batched(xs, p.lora)
                s = cosine_scores(feats[i], self.sdc_sum[k][i]) * has.float()
                tot = s if tot is None else tot + s
                num = has.float()[None, :].expand_as(s) if num is None else num + has.float()[None, :]
            if tot is None:
                out[k] = out["proto_mean"]
                continue
            out[k] = ((tot / num.clamp(min=1)).masked_fill(num == 0, -2.0)).argmax(1).cpu()
        return out

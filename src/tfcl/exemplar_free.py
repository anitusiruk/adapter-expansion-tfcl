"""Exemplar-free prototype inference (extension 2; notes/tmlr_extension2_prereg.md).

Rehearsal-free pre-trained-model learners (APER/ADAM, EASE, SEMA) store no exemplars:
class prototypes are feature means computed when the class's data are observed, and are
never recomputed. Under a package that keeps training, prototypes of earlier classes
become stale; frozen packages keep valid prototypes. This mixin records, for every
package, running sums of L2-normalised features of each observed example under that
package *at observation time* (the active package after its update on the batch) and
adds an inference rule:

  proto_ef: score(c) = mean over packages that hold a prototype for c of
            cos(f_p(x), proto_p(c)); argmax over classes with >= 1 prototype.

A package spawned after a class was observed holds no prototype for it (no exemplars to
recompute from). All other rules are unchanged. Use with --replay 0 for a fully
rehearsal-free learner (memory then only feeds the exemplar-based diagnostic rules).
"""
import torch
import torch.nn.functional as F

from src.tfcl.model import cosine_scores


class ExemplarFreeMixin:
    def _ef_init(self):
        if not hasattr(self, "ef_sum"):
            self.ef_sum, self.ef_cnt = [], []

    def _ef_sync(self):
        self._ef_init()
        while len(self.ef_sum) < len(self.packages):
            self.ef_sum.append(torch.zeros(self.n_labels, self.enc.hidden, device=self.enc.device))
            self.ef_cnt.append(torch.zeros(self.n_labels, device=self.enc.device))

    @torch.no_grad()
    def _ef_record(self, xs, ys):
        self._ef_sync()
        y = torch.tensor(ys, device=self.enc.device)
        for i, p in enumerate(self.packages):
            f = F.normalize(self.enc.features(xs, p.lora), dim=-1)
            self.ef_sum[i].index_add_(0, y, f)
            self.ef_cnt[i].index_add_(0, y, torch.ones(len(ys), device=self.enc.device))

    def observe(self, xs, ys):
        super().observe(xs, ys)
        self._ef_record(xs, ys)

    @torch.no_grad()
    def predict(self, xs, packages=None):
        out = super().predict(xs, packages)
        self._ef_sync()
        tot = num = None
        for i, p in enumerate(self.packages):
            has = self.ef_cnt[i] > 0
            if not has.any():
                continue
            f = self.enc.features_batched(xs, p.lora)
            s = cosine_scores(f, self.ef_sum[i]) * has.float()
            tot = s if tot is None else tot + s
            num = has.float()[None, :].expand_as(s) if num is None else num + has.float()[None, :]
        if tot is None:  # policy 'frozen' never calls observe(); nothing recorded
            out["proto_ef"] = out["proto_mean"]
            return out
        score = (tot / num.clamp(min=1)).masked_fill(num == 0, -2.0)
        out["proto_ef"] = score.argmax(1).cpu()
        return out

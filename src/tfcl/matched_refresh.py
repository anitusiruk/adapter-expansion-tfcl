"""Matched stale-versus-refreshed prototype diagnostic (extension 6; notes/tmlr_extension6_prereg.md).

Audit finding: proto_ef (stored, exemplar-free) and proto_mean (recomputed from memory) differ
in four ways at once: (i) sample (all observed examples vs the 512-item reservoir), (ii) feature
age (features at observation time vs under current parameters), (iii) class-package coverage
(a package spawned after a class was observed holds no stored prototype for it, but recomputes
one from memory), and (iv) aggregation (mean over packages holding a prototype vs mean over all
packages). This mixin adds rules that change these one at a time on the identical trajectory:

  proto_ms   matched STALE:   reservoir items only; for package p, an item contributes iff p
             existed when the item was observed; its feature is the one recorded at observation
             time (after the active package's update on that batch, as in proto_ef);
             score = mean over packages holding a prototype for the class (as proto_ef).
  proto_mf   matched FRESH:   identical item set, identical item->package coverage, identical
             aggregation; features recomputed under the current parameters. proto_ms vs proto_mf
             differs ONLY in when the features were computed.
  proto_mfa  fresh, full coverage: identical items, every package scores every class in memory
             (mean over holders = mean over all packages). Same argmax as proto_mean up to
             floating-point differences; reported as a check of the decomposition.

Decomposition of proto_ef -> proto_mean:
  ef -> ms (sample)  ms -> mf (feature age)  mf -> mfa (coverage + aggregation)  [mfa == mean]

Training is unchanged (no-grad forwards only, no random-number use), so proto_ef and proto_mean
in a run of this mixin equal those of the corresponding extension-2 run.
"""
import torch
import torch.nn.functional as F

from src.tfcl.model import cosine_scores


class MatchedRefreshMixin:
    def _mr_init(self):
        if not hasattr(self, "mr_store"):
            # id(item) -> (item, number of packages at observation, stale features (k, H))
            self.mr_store = {}

    def observe(self, xs, ys):
        super().observe(xs, ys)  # trains (with gradients); everything below is no-grad
        with torch.no_grad():
            self._mr_record()

    def _mr_record(self):
        self._mr_init()
        live = {id(it) for it in self.mem.items}
        for key in [k for k in self.mr_store if k not in live]:
            del self.mr_store[key]
        new = [it for it in self.mem.items if id(it) not in self.mr_store]
        if not new:
            return
        # new reservoir items all come from this batch; record their features now, i.e. at
        # observation time (after the active package's update), under every existing package
        xn = [it[0] for it in new]
        feats = torch.stack([F.normalize(self.enc.features(xn, p.lora), dim=-1) for p in self.packages], 1)
        for j, it in enumerate(new):
            self.mr_store[id(it)] = (it, len(self.packages), feats[j])

    @torch.no_grad()
    def predict(self, xs, packages=None):
        out = super().predict(xs, packages)
        self._mr_init()
        items = [v for v in self.mr_store.values()]
        if not items:  # policy 'frozen' never calls observe()
            for k in ("proto_ms", "proto_mf", "proto_mfa"):
                out[k] = out["proto_mean"]
            return out
        dev = self.enc.device
        ys = torch.tensor([it[0][1] for it in items], device=dev)
        kk = torch.tensor([it[1] for it in items], device=dev)
        xm = [it[0][0] for it in items]
        H = self.enc.hidden
        acc = {k: [None, None] for k in ("proto_ms", "proto_mf", "proto_mfa")}
        for i, p in enumerate(self.packages):
            cov = kk > i                                   # items observed while package i existed
            fq = self.enc.features_batched(xs, p.lora)
            fm = F.normalize(self.enc.features_batched(xm, p.lora), dim=-1)
            stale = torch.zeros(len(items), H, device=dev)
            for j in torch.nonzero(cov).flatten().tolist():
                stale[j] = items[j][2][i]
            for k, feat, mask in (("proto_ms", stale, cov), ("proto_mf", fm, cov),
                                  ("proto_mfa", fm, torch.ones_like(cov))):
                w = mask.float()
                sums = torch.zeros(self.n_labels, H, device=dev).index_add_(0, ys, feat * w[:, None])
                has = torch.zeros(self.n_labels, device=dev).index_add_(0, ys, w) > 0
                if not has.any():
                    continue
                s = cosine_scores(fq, sums) * has.float()
                n = has.float()[None, :].expand_as(s)
                acc[k][0] = s if acc[k][0] is None else acc[k][0] + s
                acc[k][1] = n if acc[k][1] is None else acc[k][1] + n
        for k, (tot, num) in acc.items():
            out[k] = (tot / num.clamp(min=1)).masked_fill(num == 0, -2.0).argmax(1).cpu()
        return out

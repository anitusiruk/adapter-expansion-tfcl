"""Frozen transformer encoder with a bank of selectable LoRA adapters.

Each adapter ("package") owns LoRA factors on the query/value projections of
every layer plus its own linear classifier over the global label space. The
backbone is frozen. Adapter parameters live in plain tensors so that a
package's complete trainable state (weights + AdamW moments) can be
snapshotted and restored exactly for counterfactual probes.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

TARGETS = {
    "distilbert": ("attention.q_lin", "attention.v_lin"),
    "bert": ("attention.self.query", "attention.self.value"),
    "roberta": ("attention.self.query", "attention.self.value"),
}


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear; adds B@A of the currently active adapter."""

    def __init__(self, base, scale):
        super().__init__()
        self.base = base
        self.scale = scale
        self.A = None  # set by the encoder before each forward
        self.B = None

    def forward(self, x):
        out = self.base(x)
        if self.A is not None:
            out = out + (x @ self.A.t() @ self.B.t()) * self.scale
        return out


class Encoder(nn.Module):
    def __init__(self, name, rank=8, alpha=16, max_length=64, device="cuda"):
        super().__init__()
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        self.backbone = AutoModel.from_pretrained(name).to(device).eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.rank, self.scale = rank, alpha / rank
        self.max_length = max_length
        self.device = device
        family = next(k for k in TARGETS if k in name)
        self.wrapped = []
        for mod_name, mod in list(self.backbone.named_modules()):
            for suffix in TARGETS[family]:
                if mod_name.endswith(suffix) and isinstance(mod, nn.Linear):
                    parent_name, attr = mod_name.rsplit(".", 1)
                    parent = self.backbone.get_submodule(parent_name)
                    w = LoRALinear(mod, self.scale)
                    setattr(parent, attr, w)
                    self.wrapped.append(w)
        self.hidden = self.backbone.config.hidden_size

    def new_lora(self, generator):
        """Standard LoRA init: A ~ kaiming-uniform, B = 0 (zero output)."""
        params = []
        for w in self.wrapped:
            fan_in = w.base.in_features
            bound = 1.0 / math.sqrt(fan_in)
            A = (torch.rand(self.rank, fan_in, generator=generator) * 2 - 1) * bound
            B = torch.zeros(w.base.out_features, self.rank)
            params += [A.to(self.device).requires_grad_(), B.to(self.device).requires_grad_()]
        return params

    def set_lora(self, params):
        if params is None:
            for w in self.wrapped:
                w.A = w.B = None
            return
        for i, w in enumerate(self.wrapped):
            w.A, w.B = params[2 * i], params[2 * i + 1]

    def encode(self, xs):
        if isinstance(xs[0], (list, tuple)):
            return self.tok([a for a, _ in xs], [b for _, b in xs], padding=True,
                            truncation=True, max_length=self.max_length,
                            return_tensors="pt").to(self.device)
        return self.tok(list(xs), padding=True, truncation=True,
                        max_length=self.max_length, return_tensors="pt").to(self.device)

    def features(self, xs, lora=None, grad=False):
        """Mean-pooled last hidden state under the given adapter (None = base)."""
        self.set_lora(lora)
        enc = self.encode(xs)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            h = self.backbone(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            f = (h * m).sum(1) / m.sum(1)
        self.set_lora(None)
        return f

    def features_batched(self, xs, lora=None, bs=128):
        return torch.cat([self.features(xs[i:i + bs], lora) for i in range(0, len(xs), bs)])


def masked_logits(head_w, head_b, feats, seen_mask):
    logits = feats @ head_w.t() + head_b
    return logits.masked_fill(~seen_mask, float("-inf"))


def cosine_scores(feats, protos):
    return F.normalize(feats, dim=-1) @ F.normalize(protos, dim=-1).t()

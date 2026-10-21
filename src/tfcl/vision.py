"""Vision extension (declared in notes/tmlr_extension_prereg.md).

Adds a frozen ViT-B/16 encoder and CIFAR-100 streams to the frozen text testbed
WITHOUT modifying any pre-registered file: ``VisionEncoder`` exposes the same
interface as ``src.tfcl.model.Encoder`` (``features``, ``features_batched``,
``new_lora``, ``set_lora``, ``hidden``, ``device``) and examples are integer ids
into an image tensor, so learner, triggers and runner are reused unchanged.

Streams (training split only; the official test split is never loaded):
  cifar_cil   10 disjoint groups of 10 fine classes (fixed permutation), 40
              examples per class (CIL; 250 batches).
  cifar_dil   20 coarse labels; domain d in A..E contains the d-th fine subclass
              of every superclass (BREEDS-style same-label domain shift),
              40 examples per (domain, coarse label) (DIL; 250 batches).
  cifar_conf  cifar_dil with domains B and D relabelled by the fixed involution
              c -> c xor 1 (pairs of superclasses swap labels): concept drift.
Dev sets: 30 held-out training images per class (CIL) / per (domain, label) (DIL).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel

from src.tfcl.data import _batches, _class_incremental
from src.tfcl.model import Encoder, LoRALinear

VIT = "google/vit-base-patch16-224-in21k"
_CACHE = {}


def cifar100():
    if "cifar" not in _CACHE:
        ds = load_dataset("uoft-cs/cifar100", split="train")
        imgs = np.stack([np.asarray(im.convert("RGB"), dtype=np.uint8) for im in ds["img"]])
        _CACHE["cifar"] = (imgs, np.asarray(ds["fine_label"]), np.asarray(ds["coarse_label"]))
    return _CACHE["cifar"]


class VisionEncoder(Encoder):
    """Frozen ViT with LoRA on q/v projections; x = integer image id. Forward passes use
    bf16 autocast (1.8x faster); LoRA/head parameters and optimiser state stay fp32."""

    def __init__(self, name=VIT, rank=8, alpha=16, device="cuda", images=None, **_):
        nn.Module.__init__(self)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.name = name
        self.backbone = AutoModel.from_pretrained(name).to(device).eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.rank, self.scale = rank, alpha / rank
        self.device = device
        self.wrapped = []
        for mod_name, mod in list(self.backbone.named_modules()):
            if mod_name.endswith(("attention.q_proj", "attention.v_proj")) and isinstance(mod, nn.Linear):
                parent_name, attr = mod_name.rsplit(".", 1)
                w = LoRALinear(mod, self.scale)
                setattr(self.backbone.get_submodule(parent_name), attr, w)
                self.wrapped.append(w)
        assert len(self.wrapped) == 24, len(self.wrapped)
        self.hidden = self.backbone.config.hidden_size
        self.images = torch.from_numpy(images).to(device)  # N,32,32,3 uint8
        self.res = self.backbone.config.image_size

    def pixels(self, xs):
        idx = torch.as_tensor(list(xs), device=self.device, dtype=torch.long)
        x = self.images[idx].permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(self.res, self.res), mode="bilinear", align_corners=False)
        return (x - 0.5) / 0.5  # ViT-in21k normalisation

    def features(self, xs, lora=None, grad=False):
        self.set_lora(lora)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
            h = self.backbone(pixel_values=self.pixels(xs)).last_hidden_state
            f = h.float().mean(1)
        self.set_lora(None)
        return f


def _dev_train_split(labels, n_dev, seed=424242):
    rng = np.random.default_rng(seed)
    dev_by, train_by = {}, {}
    for c in np.unique(labels):
        ids = rng.permutation(np.flatnonzero(labels == c))
        dev_by[int(c)] = ids[:n_dev].tolist()
        train_by[int(c)] = ids[n_dev:].tolist()
    return train_by, dev_by


def cifar(kind, seed, bs=16):
    _, fine, coarse = cifar100()
    if kind == "cifar_cil":
        train_by, dev_by = _dev_train_split(fine, 30)
        perm = np.random.default_rng(777).permutation(100)
        groups = [sorted(perm[i * 10:(i + 1) * 10].tolist()) for i in range(10)]
        return _class_incremental(train_by, dev_by, groups, 40, bs, seed) + (100,)
    # DIL / drift: domain d = d-th fine subclass (sorted) of each superclass
    subs = {c: sorted(np.unique(fine[coarse == c]).tolist()) for c in range(20)}
    train_by, dev_by = _dev_train_split(fine, 30)
    flip = {1, 3} if kind == "cifar_conf" else set()
    rng = np.random.default_rng(seed + 9000)
    names = [chr(ord("A") + i) for i in range(5)]
    stream, dev = [], {}
    for d, name in enumerate(names):
        rows, xs, ys = [], [], []
        for c in range(20):
            f = subs[c][d]
            lab = (c ^ 1) if d in flip else c
            pool = train_by[f]
            pick = rng.permutation(len(pool))[:40]
            rows += [(pool[i], lab) for i in pick]
            xs += dev_by[f]
            ys += [lab] * len(dev_by[f])
        stream += _batches(rows, bs, name, rng)
        dev[name] = {"x": xs, "y": ys}
    return stream, dev, 20


VISION_KINDS = ("cifar_cil", "cifar_dil", "cifar_conf")


def make_stream_v(kind, seed, bs=16):
    """Vision streams; ``<kind>_iid`` uses the same null construction as src.tfcl.data."""
    if kind.endswith("_iid"):
        stream, dev, n = make_stream_v(kind[:-4], seed, bs)
        rows = [(x, y) for b in stream for x, y in zip(b["x"], b["y"])]
        return _batches(rows, bs, "iid", np.random.default_rng(seed + 31337)), dev, n
    if kind in VISION_KINDS:
        return cifar(kind, seed, bs)
    raise ValueError(kind)

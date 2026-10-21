"""ImageNet-R class-incremental stream (extension 3; notes/tmlr_extension3_prereg.md).

ImageNet-R (Hendrycks et al., 2021; 200 classes, 30k renditions) is the standard
class-incremental benchmark for pre-trained-model continual learning (e.g. L2P, CODA-P,
EASE, SEMA). Preprocessed by experiments/prep_imagenet_r.py to a 224x224 uint8 memmap
(/workspace/imnr). There is no official train/test split; we hold out 20% of each class
(fixed seed) as the development set and stream 40 examples per class from the rest.

  imnr_cil  10 groups of 20 classes (fixed permutation), 500 batches.
"""
import numpy as np
import torch

from src.tfcl.data import _batches, _class_incremental
from src.tfcl.vision import VisionEncoder

ROOT = "/workspace/imnr"


def imnr_arrays():
    return np.load(f"{ROOT}/images.npy", mmap_mode="r"), np.load(f"{ROOT}/labels.npy")


class MemmapVisionEncoder(VisionEncoder):
    """Images stay in a CPU memmap (4.5 GB); each batch is copied to the GPU."""

    def __init__(self, images, **kw):
        super().__init__(images=np.zeros((1, 2, 2, 3), np.uint8), **kw)
        self.images_cpu = images

    def pixels(self, xs):
        idx = np.asarray(list(xs), dtype=np.int64)
        order = np.argsort(idx)
        arr = np.empty((len(idx),) + self.images_cpu.shape[1:], np.uint8)
        arr[order] = self.images_cpu[idx[order]]
        x = torch.from_numpy(arr).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        return (x - 0.5) / 0.5


def imnr(kind, seed, bs=16):
    _, labels = imnr_arrays()
    rng = np.random.default_rng(424242)
    train_by, dev_by = {}, {}
    for c in range(200):
        ids = rng.permutation(np.flatnonzero(labels == c))
        n_dev = int(round(0.2 * len(ids)))
        dev_by[c] = ids[:n_dev].tolist()
        train_by[c] = ids[n_dev:].tolist()
    perm = np.random.default_rng(777).permutation(200)
    groups = [sorted(perm[i * 20:(i + 1) * 20].tolist()) for i in range(10)]
    return _class_incremental(train_by, dev_by, groups, 40, bs, seed) + (200,)


def make_stream_imnr(kind, seed, bs=16):
    if kind.endswith("_iid"):
        stream, dev, n = make_stream_imnr(kind[:-4], seed, bs)
        rows = [(x, y) for b in stream for x, y in zip(b["x"], b["y"])]
        return _batches(rows, bs, "iid", np.random.default_rng(seed + 31337)), dev, n
    if kind == "imnr_cil":
        return imnr(kind, seed, bs)
    raise ValueError(kind)

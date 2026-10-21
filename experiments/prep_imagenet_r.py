"""Preprocess ImageNet-R to a 224x224 uint8 memmap plus labels for src/tfcl/vision_imnr.py."""
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from datasets import load_dataset
from PIL import Image

OUT = sys.argv[1] if len(sys.argv) > 1 else "/workspace/imnr"
R = 224


def prep(img):
    img = img.convert("RGB")
    w, h = img.size
    s = R / min(w, h)
    img = img.resize((max(R, round(w * s)), max(R, round(h * s))), Image.BICUBIC)
    w, h = img.size
    l, t = (w - R) // 2, (h - R) // 2
    return np.asarray(img.crop((l, t, l + R, t + R)), dtype=np.uint8)


def chunk(args):
    lo, hi = args
    ds = load_dataset("axiong/imagenet-r", split="test")
    mm = np.lib.format.open_memmap(f"{OUT}/images.npy", mode="r+")
    for i in range(lo, hi):
        mm[i] = prep(ds[i]["image"])
    mm.flush()
    return hi - lo


def main():
    ds = load_dataset("axiong/imagenet-r", split="test")
    n = len(ds)
    wnids = ds["wnid"]
    classes = sorted(set(wnids))
    labels = np.array([classes.index(w) for w in wnids], dtype=np.int64)
    np.save(f"{OUT}/labels.npy", labels)
    json.dump({"classes": classes, "names": {w: c for w, c in zip(wnids, ds["class_name"])}}, open(f"{OUT}/meta.json", "w"))
    np.lib.format.open_memmap(f"{OUT}/images.npy", mode="w+", dtype=np.uint8, shape=(n, R, R, 3)).flush()
    step = 1000
    with ProcessPoolExecutor(24) as ex:
        done = sum(ex.map(chunk, [(i, min(n, i + step)) for i in range(0, n, step)]))
    cnt = np.bincount(labels)
    print("images", done, "classes", len(classes), "per-class min/median/max", cnt.min(), int(np.median(cnt)), cnt.max())


if __name__ == "__main__":
    main()

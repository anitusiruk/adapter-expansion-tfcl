"""Extension-3 runner: tfcl_run_x2 plus the ImageNet-R stream (src/tfcl/vision_imnr.py)."""
import experiments.tfcl_run_x as X
import experiments.tfcl_run_x2 as X2
from src.tfcl import vision_imnr as V

_orig_stream, _orig_enc = X.make_stream, X.make_encoder
_IS = {"imnr": False}


def make_stream(kind, seed, bs=16):
    if kind.split("_iid")[0] == "imnr_cil":
        _IS["imnr"] = True
        out = V.make_stream_imnr(kind, seed, bs)
        st = out[0]
        X.STATE["first_change"] = next((i for i in range(1, len(st)) if st[i]["seg"] != st[i - 1]["seg"]), len(st))
        return out
    return _orig_stream(kind, seed, bs)


def make_encoder(name, **kw):
    if _IS["imnr"]:
        return V.MemmapVisionEncoder(images=V.imnr_arrays()[0], **kw)
    return _orig_enc(name, **kw)


X.make_stream, X.make_encoder = make_stream, make_encoder

if __name__ == "__main__":
    X2.main()

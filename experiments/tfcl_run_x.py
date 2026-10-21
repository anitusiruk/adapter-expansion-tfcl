"""Extension runner (notes/tmlr_extension_prereg.md).

Runs the frozen experiments/tfcl_run.py logic unchanged, adding only:
  * vision streams (src/tfcl/vision.py) with a frozen ViT-B/16 encoder;
  * natural concept-drift text stream tweet_conc (src/tfcl/data_x.py);
  * policy ``firstseg``: one package trained on the first segment only, then
    frozen (memory and seen labels keep updating, so prototype inference still
    covers every class). This is the first-session-adaptation baseline that
    SEMA/APER-style ablations compare expansion against;
  * ``--replay 0`` (already a frozen-runner option) gives rehearsal-free training;
    memory is then used only for prototypes.
"""
import sys

import experiments.tfcl_run as R
from src.tfcl import data as text_data
from src.tfcl.learner import Learner
from src.tfcl.model import Encoder
from src.tfcl import vision

STATE = {"first_change": None, "vision": False}


def make_stream(kind, seed, bs=16):
    if kind.split("_iid")[0] in vision.VISION_KINDS:
        STATE["vision"] = True
        out = vision.make_stream_v(kind, seed, bs)
    elif kind.split("_iid")[0] in ("tweet_conc",):
        from src.tfcl.data_x import make_stream_t
        out = make_stream_t(kind, seed, bs)
    else:
        out = text_data.make_stream(kind, seed, bs)
    stream = out[0]
    STATE["first_change"] = next((i for i in range(1, len(stream)) if stream[i]["seg"] != stream[i - 1]["seg"]),
                                 len(stream))
    return out


def make_encoder(name, **kw):
    if STATE["vision"]:
        return vision.VisionEncoder(images=vision.cifar100()[0], **kw)
    return Encoder(name, **kw)


class FirstSegLearner(Learner):
    def observe(self, xs, ys):
        if self.t < STATE["first_change"]:
            return super().observe(xs, ys)
        for y in ys:
            self.seen[y] = True
            if self.first_seen[y] < 0:
                self.first_seen[y] = self.t
        for x, y in zip(xs, ys):
            self.mem.add(x, y, self.pid(self.active))
        self.t += 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    pol = argv[argv.index("--policy") + 1]
    if pol == "firstseg":
        argv[argv.index("--policy") + 1] = "single"
        R.Learner = FirstSegLearner
    R.make_stream = make_stream
    R.Encoder = make_encoder
    if pol == "firstseg":
        argv += ["--policy-label", "firstseg"]
    R.main(argv) if "--policy-label" not in argv else _main_with_label(argv)


def _main_with_label(argv):
    i = argv.index("--policy-label")
    label = argv[i + 1]
    del argv[i:i + 2]
    R.main(argv)
    # record the true policy in the saved JSON
    import json
    from pathlib import Path
    out = Path(argv[argv.index("--out") + 1])
    rec = json.loads(out.read_text())
    rec["args"]["policy"] = label
    out.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()

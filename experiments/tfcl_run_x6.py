"""Extension-6 runner (notes/tmlr_extension6_prereg.md): tfcl_run_x3 (all streams incl. ImageNet-R)
plus two audit controls. Frozen files are not modified.

  --mr             exemplar-free learner (requires --ef semantics: implies the proto_ef rule) that
                   also reports the matched stale/fresh prototype rules proto_ms, proto_mf, proto_mfa
                   (src/tfcl/matched_refresh.py). Training identical to the extension-2 run.
  --global-replay  replay draws from the WHOLE reservoir instead of the items owned by the active
                   package. For one package (single) this is identical to owned replay, because the
                   one package owns every item; for expansion it gives a new package rehearsal access
                   to examples observed under earlier packages.
"""
import json
import sys
from pathlib import Path

import experiments.tfcl_run as R
import experiments.tfcl_run_x as X
import experiments.tfcl_run_x3  # noqa: F401  (registers the ImageNet-R stream on X)
from src.tfcl.exemplar_free import ExemplarFreeMixin
from src.tfcl.learner import Learner
from src.tfcl.matched_refresh import MatchedRefreshMixin


class GlobalReplayMixin:
    def _replay_sample(self, p, rng, k):
        if k == 0 or p not in self.packages:
            return [], []
        items = self.mem.items
        if not items:
            return [], []
        pick = [items[rng.randrange(len(items))] for _ in range(k)]
        return [it[0] for it in pick], [it[1] for it in pick]


def build(mr, glob, firstseg):
    bases = []
    if mr:
        bases += [MatchedRefreshMixin, ExemplarFreeMixin]
    if glob:
        bases += [GlobalReplayMixin]
    bases += [X.FirstSegLearner if firstseg else Learner]
    return type("X6Learner", tuple(bases), {})


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    mr, glob = "--mr" in argv, "--global-replay" in argv
    for f in ("--mr", "--global-replay"):
        if f in argv:
            argv.remove(f)
    firstseg = argv[argv.index("--policy") + 1] == "firstseg"
    cls = build(mr, glob, firstseg)
    real_main = R.main

    def r_main(av):
        R.Learner = cls
        return real_main(av)
    R.main = r_main
    X.main(argv)
    out = Path(argv[argv.index("--out") + 1])
    rec = json.loads(out.read_text())
    rec["args"].update(mr=mr, global_replay=glob, ef=mr)
    out.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()

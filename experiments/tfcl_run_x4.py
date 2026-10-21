"""Extension-4 runner: tfcl_run_x3 (all streams, incl. ImageNet-R) with ``--ef`` learners that
additionally carry SDC-refreshed exemplar-free prototypes (src/tfcl/drift_comp.py).

Training is unchanged: proto_ef in a run of this runner equals proto_ef of the same
extension-2 run (checked by experiments/tfcl_ext4_integrity.py). Requires --ef.
"""
import json
import sys
from pathlib import Path

import experiments.tfcl_run as R
import experiments.tfcl_run_x as X
import experiments.tfcl_run_x3  # noqa: F401  (registers the ImageNet-R stream on X)
from src.tfcl.drift_comp import SDCMixin
from src.tfcl.exemplar_free import ExemplarFreeMixin
from src.tfcl.learner import Learner

LIVE = {}


class _Track:
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        LIVE["learner"] = self


class SDCLearner(_Track, SDCMixin, ExemplarFreeMixin, Learner):
    pass


class SDCFirstSegLearner(_Track, SDCMixin, ExemplarFreeMixin, X.FirstSegLearner):
    def _sdc_will_train(self):
        return self.t < X.STATE["first_change"]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--ef" not in argv:
        raise SystemExit("tfcl_run_x4 requires --ef")
    argv.remove("--ef")
    firstseg = argv[argv.index("--policy") + 1] == "firstseg"
    real_main = R.main

    def r_main(av):
        R.Learner = SDCFirstSegLearner if firstseg else SDCLearner
        return real_main(av)
    R.main = r_main
    X.main(argv)
    out = Path(argv[argv.index("--out") + 1])
    rec = json.loads(out.read_text())
    rec["args"]["ef"] = True
    rec["sdc_stats"] = getattr(LIVE.get("learner"), "sdc_stats", None)
    out.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()

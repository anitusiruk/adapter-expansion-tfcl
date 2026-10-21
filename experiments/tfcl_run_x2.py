"""Extension-2 runner: tfcl_run_x plus ``--ef`` (exemplar-free prototype rule proto_ef,
src/tfcl/exemplar_free.py). Without --ef it is identical to tfcl_run_x."""
import sys

import experiments.tfcl_run as R
import experiments.tfcl_run_x as X
from src.tfcl.exemplar_free import ExemplarFreeMixin
from src.tfcl.learner import Learner


class EFLearner(ExemplarFreeMixin, Learner):
    pass


class EFFirstSegLearner(ExemplarFreeMixin, X.FirstSegLearner):
    pass


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--ef" in argv:
        argv.remove("--ef")
        firstseg = argv[argv.index("--policy") + 1] == "firstseg"
        orig = X.main

        def patched(av):
            return orig(av)
        # X.main sets R.Learner = FirstSegLearner for firstseg; override after by wrapping R.main
        real_main = R.main

        def r_main(av):
            R.Learner = EFFirstSegLearner if firstseg else EFLearner
            return real_main(av)
        R.main = r_main
    X.main(argv)


if __name__ == "__main__":
    main()

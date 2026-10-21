"""Extension grid (tfcl_run_x): run a grid of jobs with a few parallel GPU workers.

Never overwrites an existing output; failed jobs keep their logs in
<outdir>/logs and are reported at the end (re-running retries only them).
"""
import argparse
import itertools
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

STREAMS = ["banking_rec", "banking_cil", "clinc_cil", "amazon_rec", "amazon_dil",
           "amazon_conflict", "amazon_dilconf"]
TRIGGERS = ["label_novel", "loss_z", "repr_z", "interf_logit", "fresh_util", "interf_proto", "conflict_z", "label_surprise"]


def jobs(a, taus):
    for stream, seed in itertools.product(a.streams, a.seeds):
        for pol in a.policies:
            extra = []
            if pol.startswith("trigger:"):
                name = pol.split(":", 1)[1]
                tau = taus[stream][f"null_tau_q{a.q}"][name]
                extra = ["--trigger-args", json.dumps({"tau": tau})]
            tag = pol.replace(":", "-") + (f"-q{a.q}" if pol.startswith("trigger:") else "")
            out = Path(a.outdir) / stream / f"{tag}_s{seed}.json"
            cmd = [sys.executable, "-m", "experiments.tfcl_run_x", "--stream", stream, "--policy", pol,
                   "--seed", str(seed), "--model", a.model, "--max-length", str(a.max_length),
                   "--out", str(out)] + extra + a.extra
            yield out, cmd


def run(job, logdir):
    out, cmd = job
    if out.exists():
        return out, "skip"
    log = logdir / (out.parent.name + "__" + out.stem + ".log")
    with open(log, "w") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
    return out, "ok" if rc == 0 else f"fail({rc})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--streams", nargs="+", default=STREAMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(2027, 2037)))
    ap.add_argument("--policies", nargs="+",
                    default=["frozen", "single", "oracle"] + [f"trigger:{t}" for t in TRIGGERS])
    ap.add_argument("--taus", default="results/tfcl/shadow/report_seed2026.json")
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--q", type=float, default=0.99, help="null quantile used as threshold")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    a = ap.parse_args()
    taus = json.loads(Path(a.taus).read_text())
    logdir = Path(a.outdir) / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    js = list(jobs(a, taus))
    print(f"{len(js)} jobs", flush=True)
    fails = []
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (out, st) in enumerate(ex.map(lambda j: run(j, logdir), js)):
            if st.startswith("fail"):
                fails.append(str(out))
            if st != "skip":
                print(f"[{i + 1}/{len(js)}] {st} {out}", flush=True)
    print("FAILED:", fails if fails else "none", flush=True)


if __name__ == "__main__":
    main()

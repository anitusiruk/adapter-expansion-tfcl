"""Scheduling helper (no effect on results): run_jobs.py with a configurable launch delay and
GPU-memory gate, for many short jobs. Usage: run_jobs_fast.py <jobs.jsonl> <workers> <module>
<need_mb> <delay_s>"""
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import experiments.run_jobs as R

_LOCK = threading.Lock()


def run(args, module, need_mb, delay):
    out = Path(args[args.index("--out") + 1])
    if out.exists():
        return None
    with _LOCK:
        while R.gpu_free_mb() < need_mb:
            time.sleep(10)
        time.sleep(delay)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_suffix(".log")
    with open(log, "w") as fh:
        rc = subprocess.call([sys.executable, "-m", module] + args, stdout=fh, stderr=subprocess.STDOUT)
    if rc == 0:
        log.unlink()
        return None
    return f"FAIL {' '.join(args)}"


def main():
    jobfile, workers, module, need, delay = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), float(sys.argv[5])
    jobs = [json.loads(l) for l in open(jobfile) if l.strip()]
    with ThreadPoolExecutor(workers) as ex:
        for r in ex.map(lambda j: run(j, module, need, delay), jobs):
            if r:
                print(r, flush=True)
    print("JOBS_DONE", flush=True)


if __name__ == "__main__":
    main()

"""Run a list of jobs (one JSON argv list per line) with N parallel workers; skip jobs whose
--out exists; print FAIL lines for failed jobs. No shell quoting involved."""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def gpu_free_mb():
    try:
        return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]).split()[0])
    except Exception:
        return 10 ** 6


_LAUNCH = __import__("threading").Lock()


def run(args, module, need_mb=6000):
    out = Path(args[args.index("--out") + 1])
    if out.exists():
        return None
    import time
    with _LAUNCH:  # wait for GPU headroom, then give the job time to allocate before the next launch
        while gpu_free_mb() < need_mb:
            time.sleep(20)
        time.sleep(30)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_suffix(".log")
    with open(log, "w") as fh:
        rc = subprocess.call([sys.executable, "-m", module] + args, stdout=fh, stderr=subprocess.STDOUT)
    if rc == 0:
        log.unlink()
        return None
    return f"FAIL {' '.join(args)}"


def main():
    jobfile, workers, module = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    jobs = [json.loads(l) for l in open(jobfile) if l.strip()]
    with ThreadPoolExecutor(workers) as ex:
        for r in ex.map(lambda j: run(j, module), jobs):
            if r:
                print(r, flush=True)
    print("JOBS_DONE", flush=True)


if __name__ == "__main__":
    main()

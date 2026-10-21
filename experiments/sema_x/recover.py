"""Rebuild results/sema_x/<variant>_s<seed>.json from per-task log lines when the final json.dump failed."""
import glob
import json
import re

import numpy as np

for log in glob.glob("results/sema_x/*_s*.log"):
    v, s = re.match(r".*/(\w+)_s(\d+)\.log", log).groups()
    out = log[:-4] + ".json"
    try:
        json.load(open(out))
        continue
    except Exception:
        pass
    tasks = [json.loads(l) for l in open(log, errors="ignore") if l.startswith('{"task"')]
    if len(tasks) < 10 or "ncm_full" not in tasks[-1]:
        continue
    rec = {"args": {"variant": v, "seed": int(s), "recovered_from_log": True}, "tasks": tasks,
           "final": {k: tasks[-1][k] for k in ("linear", "ncm_ef", "ncm_ex20", "ncm_full")},
           "avg_inc": {k: float(np.mean([t[k] for t in tasks])) for k in ("linear", "ncm_ef", "ncm_ex20")},
           "seconds": tasks[-1]["seconds"]}
    json.dump(rec, open(out, "w"), indent=1)
    print("recovered", out, rec["final"])

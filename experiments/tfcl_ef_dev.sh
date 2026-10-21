#!/bin/bash
# Extension 2 DEVELOPMENT (seed 2026 only): exemplar-free, rehearsal-free regime (--replay 0 --ef).
set -u
export PYTHONPATH=/workspace/mitosis-interference TRANSFORMERS_VERBOSITY=error HF_DATASETS_TRUST_REMOTE_CODE=1
cd /workspace/mitosis-interference
python - > /tmp/ef_dev_jobs.jsonl <<'PY'
import json
D = "results/tfcl/ef_dev"
for s in ["banking_rec", "banking_cil", "clinc_cil", "amazon_dil", "amazon_dilconf", "cifar_cil", "cifar_conf"]:
    T = json.load(open("results/tfcl/vision_dev/shadow/report_seed2026.json" if s.startswith("cifar") else "results/tfcl/shadow/report_seed2026.json"))
    for p in ["single", "oracle", "random", "firstseg"]:
        print(json.dumps(["--stream", s, "--policy", p, "--replay", "0", "--ef", "--seed", "2026", "--out", f"{D}/{s}/{p}_s2026.json"]))
    for t in ["loss_z", "label_surprise", "label_novel"]:
        tau = T[s]["null_tau_q0.99"][t]
        print(json.dumps(["--stream", s, "--policy", f"trigger:{t}", "--trigger-args", json.dumps({"tau": tau}), "--replay", "0", "--ef",
                          "--seed", "2026", "--out", f"{D}/{s}/trigger-{t}-q0.99_s2026.json"]))
PY
python experiments/run_jobs.py /tmp/ef_dev_jobs.jsonl 2 experiments.tfcl_run_x2

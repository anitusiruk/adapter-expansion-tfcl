#!/bin/bash
# Extension 2 pipeline exactly as declared in notes/tmlr_extension2_prereg.md.
set -u
export PYTHONPATH=/workspace/mitosis-interference TRANSFORMERS_VERBOSITY=error HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/mitosis-interference
python - > /tmp/ext2_jobs.jsonl <<'PY'
import json
T = json.load(open("results/tfcl/ef_dev/shadow/report_seed2026.json"))
D = "results/tfcl/ext2_ef"
text = ["banking_rec", "banking_cil", "clinc_cil", "news_cil", "amazon_dil", "senti_dil", "amazon_dilconf", "senti_conf"]
vis = ["cifar_cil", "cifar_dil", "cifar_conf"]
jobs = []
for streams, seeds in ((text, range(2027, 2037)), (vis, range(2027, 2035))):
    for seed in seeds:
        for s in streams:
            for p in ["single", "oracle", "random", "firstseg"]:
                jobs.append(["--stream", s, "--policy", p, "--replay", "0", "--ef", "--seed", str(seed),
                             "--out", f"{D}/{s}/{p}_s{seed}.json"])
            for t in ["label_novel", "loss_z", "label_surprise"]:
                tau = T[s]["null_tau_q0.99"][t]
                jobs.append(["--stream", s, "--policy", f"trigger:{t}", "--trigger-args", json.dumps({"tau": tau}),
                             "--replay", "0", "--ef", "--seed", str(seed), "--out", f"{D}/{s}/trigger-{t}-q0.99_s{seed}.json"])
for j in jobs:
    print(json.dumps(j))
PY
python experiments/run_jobs.py /tmp/ext2_jobs.jsonl ${EXT2_WORKERS:-3} experiments.tfcl_run_x2
echo EXTENSION2_DONE

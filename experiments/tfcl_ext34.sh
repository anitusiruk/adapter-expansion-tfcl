#!/bin/bash
# Extensions 3 + 4 exactly as declared in notes/tmlr_extension34_prereg.md.
set -u
export PYTHONPATH=/workspace/mitosis-interference TRANSFORMERS_VERBOSITY=error HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd /workspace/mitosis-interference
python - > /tmp/ext4_jobs.jsonl <<'PY'
import json
for s in ["banking_rec", "banking_cil", "clinc_cil", "news_cil", "cifar_cil"]:
    for seed in range(2027, 2032):
        for p in ["single", "oracle", "firstseg"]:
            print(json.dumps(["--stream", s, "--policy", p, "--replay", "0", "--ef", "--seed", str(seed),
                              "--out", f"results/tfcl/ext4_sdc/{s}/{p}_s{seed}.json"]))
for seed in range(2027, 2033):
    for p in ["single", "oracle", "firstseg"]:
        print(json.dumps(["--stream", "imnr_cil", "--policy", p, "--replay", "0", "--ef", "--seed", str(seed),
                          "--out", f"results/tfcl/ext3_imnr/ef/imnr_cil/{p}_s{seed}.json"]))
PY
python - > /tmp/ext3_jobs.jsonl <<'PY'
import json
for seed in range(2027, 2033):
    for p in ["single", "oracle", "firstseg"]:
        print(json.dumps(["--stream", "imnr_cil", "--policy", p, "--seed", str(seed),
                          "--out", f"results/tfcl/ext3_imnr/exemplar/imnr_cil/{p}_s{seed}.json"]))
PY
python experiments/run_jobs.py /tmp/ext4_jobs.jsonl ${EXT4_WORKERS:-3} experiments.tfcl_run_x4
python experiments/run_jobs.py /tmp/ext3_jobs.jsonl ${EXT4_WORKERS:-3} experiments.tfcl_run_x3
echo EXT34_DONE

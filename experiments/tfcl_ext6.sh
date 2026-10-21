#!/bin/bash
# Extension 6 (audit controls) exactly as declared in notes/tmlr_extension6_prereg.md.
# Two queues: TEXT (short jobs, EXT6_TWORKERS) and VISION (EXT6_VWORKERS). Skips finished outputs.
set -u
export PYTHONPATH=/workspace/mitosis-interference TRANSFORMERS_VERBOSITY=error HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/mitosis-interference
J=${EXT6_JOBDIR:-/tmp/ext6}; mkdir -p $J
python - "$J" <<'PY'
import json, sys
J = sys.argv[1]
D = "results/tfcl/ext6"
TXT = ["banking_rec", "banking_cil", "clinc_cil", "news_cil"]
ALLTXT = TXT + ["amazon_rec", "amazon_dil", "amazon_conflict", "amazon_dilconf", "senti_dil", "senti_conf"]
TS, VS, IS = range(2027, 2037), range(2027, 2035), range(2027, 2033)
text, vis = [], []
# Block M: matched stale/fresh prototypes, exemplar-free (single, oracle)
for s in TXT:
    for seed in TS:
        for p in ["single", "oracle"]:
            text.append(["--stream", s, "--policy", p, "--replay", "0", "--mr", "--seed", str(seed),
                         "--out", f"{D}/matched/{s}/{p}_s{seed}.json"])
for s, seeds in (("cifar_cil", VS), ("imnr_cil", IS)):
    for seed in seeds:
        for p in ["single", "oracle"]:
            vis.append(["--stream", s, "--policy", p, "--replay", "0", "--mr", "--seed", str(seed),
                        "--out", f"{D}/matched/{s}/{p}_s{seed}.json"])
# Block G: oracle expansion with global replay access, exemplar-based (replay 16)
for s in ALLTXT:
    for seed in TS:
        text.append(["--stream", s, "--policy", "oracle", "--global-replay", "--seed", str(seed),
                     "--out", f"{D}/global_replay/{s}/oracle-global_s{seed}.json"])
for s, seeds in (("cifar_cil", VS), ("imnr_cil", IS)):
    for seed in seeds:
        vis.append(["--stream", s, "--policy", "oracle", "--global-replay", "--seed", str(seed),
                    "--out", f"{D}/global_replay/{s}/oracle-global_s{seed}.json"])
# integrity: single with global replay must equal saved single (vision path)
vis.append(["--stream", "cifar_cil", "--policy", "single", "--global-replay", "--seed", "2027",
            "--out", f"{D}/global_replay/cifar_cil/single-global_s2027.json"])
# Block F: closed-loop false spawns on independent shuffled null streams
T = json.load(open("results/tfcl/shadow/report_seed2026.json"))
TV = json.load(open("results/tfcl/vision_dev/shadow/report_seed2026.json"))
TE = json.load(open("results/tfcl/ef_dev/shadow/report_seed2026.json"))
for s in ALLTXT:
    for seed in range(2027, 2032):
        for t in ["label_novel", "loss_z", "repr_z", "interf_logit", "fresh_util", "label_surprise"]:
            text.append(["--stream", f"{s}_iid", "--policy", f"trigger:{t}", "--trigger-args",
                         json.dumps({"tau": T[s]["null_tau_q0.99"][t]}), "--seed", str(seed),
                         "--out", f"{D}/null_closed/{s}/trigger-{t}-q0.99_s{seed}.json"])
for s in TXT:
    for seed in range(2027, 2032):
        for t in ["loss_z", "label_surprise"]:
            text.append(["--stream", f"{s}_iid", "--policy", f"trigger:{t}", "--trigger-args",
                         json.dumps({"tau": TE[s]["null_tau_q0.99"][t]}), "--replay", "0", "--seed", str(seed),
                         "--out", f"{D}/null_closed_ef/{s}/trigger-{t}-q0.99_s{seed}.json"])
for s in ["cifar_cil", "cifar_dil", "cifar_conf"]:
    for seed in range(2027, 2030):
        for t in ["label_novel", "loss_z", "repr_z", "fresh_util", "label_surprise"]:
            vis.append(["--stream", f"{s}_iid", "--policy", f"trigger:{t}", "--trigger-args",
                        json.dumps({"tau": TV[s]["null_tau_q0.99"][t]}), "--seed", str(seed),
                        "--out", f"{D}/null_closed/{s}/trigger-{t}-q0.99_s{seed}.json"])
open(f"{J}/text.jsonl", "w").write("".join(json.dumps(j) + "\n" for j in text))
open(f"{J}/vis.jsonl", "w").write("".join(json.dumps(j) + "\n" for j in vis))
print(len(text), "text jobs;", len(vis), "vision jobs", flush=True)
PY
python experiments/run_jobs_fast.py $J/text.jsonl ${EXT6_TWORKERS:-6} experiments.tfcl_run_x6 3000 3 &
python experiments/run_jobs.py $J/vis.jsonl ${EXT6_VWORKERS:-3} experiments.tfcl_run_x6 &
wait
echo EXT6_DONE

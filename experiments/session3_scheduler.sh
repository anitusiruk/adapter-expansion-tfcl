#!/bin/bash
# Scheduling only (no effect on results): start extra workers when capacity frees up.
cd /workspace/mitosis-interference
export PYTHONPATH=$PWD TRANSFORMERS_VERBOSITY=error HF_DATASETS_TRUST_REMOTE_CODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
n() { find "$1" -name '*.json' 2>/dev/null | wc -l; }
# 1) when extension-4 SDC runs are done: second extension-1 vision worker, reverse order
until [ $(n results/tfcl/ext4_sdc) -ge 75 ]; do sleep 60; done
echo "$(date) start reverse vision worker"
python -m experiments.tfcl_grid_x --outdir results/tfcl/ext_vision --streams cifar_conf cifar_dil cifar_cil \
  --seeds 2034 2033 2032 2031 2030 2029 2028 2027 --workers 1 --q 0.99 \
  --taus results/tfcl/vision_dev/shadow/report_seed2026.json \
  --policies trigger:label_surprise trigger:fresh_util trigger:repr_z trigger:loss_z trigger:label_novel random frozen firstseg oracle single \
  > results/tfcl/ext_vision_reverse.log 2>&1 &
# 2) when extension-2 text runs are done: BERT subset
until [ $(ls results/tfcl/ext2_ef/{banking_rec,banking_cil,clinc_cil,news_cil,amazon_dil,senti_dil,amazon_dilconf,senti_conf}/*.json 2>/dev/null | wc -l) -ge 560 ]; do sleep 60; done
echo "$(date) start BERT subset"
BERT_WORKERS=2 experiments/retry_until_done.sh experiments/tfcl_bert_reduced.sh results/tfcl/bert_reduced_pipeline.log
wait
echo "$(date) scheduler done"

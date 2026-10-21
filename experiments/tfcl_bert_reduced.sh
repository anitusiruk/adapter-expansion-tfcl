#!/bin/bash
# Session-3 compute deviation: strict subset of the pre-registered BERT grid (notes/tmlr_session3_plan.md).
set -u
export PYTHONPATH=/workspace/mitosis-interference TRANSFORMERS_VERBOSITY=error HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/mitosis-interference
ALL="banking_rec banking_cil clinc_cil amazon_rec amazon_dil amazon_conflict amazon_dilconf news_cil senti_dil senti_conf"
python -m experiments.tfcl_grid --outdir results/tfcl/confirm_bert --streams $ALL --workers ${BERT_WORKERS:-3} --q 0.99 \
  --seeds 2027 2028 2029 2030 2031 --model bert-base-uncased --taus results/tfcl/shadow_bert/report_seed2026.json \
  --policies single oracle trigger:loss_z trigger:label_surprise
echo BERT_REDUCED_DONE

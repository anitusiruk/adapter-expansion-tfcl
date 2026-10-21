#!/bin/bash
# Confirmation pipeline exactly as pre-registered in notes/tmlr_confirmation_prereg.md
set -u
export PYTHONPATH=/workspace/mitosis-interference TRANSFORMERS_VERBOSITY=error HF_DATASETS_TRUST_REMOTE_CODE=1
cd /workspace/mitosis-interference
ALL="banking_rec banking_cil clinc_cil amazon_rec amazon_dil amazon_conflict amazon_dilconf news_cil senti_dil senti_conf"
TR="trigger:label_novel trigger:loss_z trigger:repr_z trigger:interf_logit trigger:fresh_util trigger:interf_proto trigger:conflict_z trigger:label_surprise"
python -m experiments.tfcl_grid --outdir results/tfcl/confirm --streams $ALL --workers 5 --q 0.99 --policies frozen single oracle random $TR
python -m experiments.tfcl_grid --outdir results/tfcl/confirm --streams $ALL --workers 5 --q 0.95 --policies $TR
python -m experiments.tfcl_grid --outdir results/tfcl/confirm_bert --streams $ALL --workers 6 --q 0.99 --seeds 2027 2028 2029 2030 2031 --model bert-base-uncased --taus results/tfcl/shadow_bert/report_seed2026.json --policies single oracle random $TR
for nl in "" "--no-lora"; do
  tag=$([ -z "$nl" ] && echo lora || echo nolora)
  for s in banking_rec banking_cil clinc_cil news_cil; do for seed in 2027 2028 2029 2030 2031; do
    out=results/tfcl/attribution/${tag}/${s}_s${seed}.json
    [ -f $out ] || echo "python -m experiments.tfcl_run --stream $s --policy shadow --seed $seed $nl --out $out"
  done; done
done | xargs -P 6 -I{} sh -c '{} > /dev/null 2>&1'
echo CONFIRM_PIPELINE_DONE

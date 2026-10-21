#!/bin/bash
set -u
export PYTHONPATH=/workspace/mitosis-interference TRANSFORMERS_VERBOSITY=error HF_DATASETS_TRUST_REMOTE_CODE=1
cd /workspace/mitosis-interference
until grep -q CONFIRM_PIPELINE_DONE results/tfcl/confirm_pipeline.log; do sleep 60; done
{
for s in banking_cil clinc_cil amazon_dil amazon_conflict; do for seed in 2027 2028 2029 2030 2031; do
  o=results/tfcl/explore_design/$s/single_s$seed.json; [ -f $o ] || echo "--stream $s --policy single --seed $seed --out $o"
  for li in fresh inherit; do for hi in fresh inherit; do
    o=results/tfcl/explore_design/$s/oracle-L$li-H${hi}_s$seed.json
    [ -f $o ] || echo "--stream $s --policy oracle --lora-init $li --head-init $hi --seed $seed --out $o"
  done; done
done; done
for s in banking_cil amazon_conflict; do for seed in 2027 2028 2029 2030 2031; do
  for hp in "5e-4 2" "2e-3 2" "1e-3 1"; do set -- $hp; for p in single oracle; do
    o=results/tfcl/explore_hparams/$s/$p-lr$1-st$2_s$seed.json
    [ -f $o ] || echo "--stream $s --policy $p --lr $1 --steps $2 --seed $seed --out $o"
  done; done
done; done
} | xargs -P 6 -I{} sh -c 'python -m experiments.tfcl_run {} > /dev/null 2>&1 || echo FAIL {}'
echo EXPLORE_DONE

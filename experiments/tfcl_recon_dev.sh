#!/bin/bash
# Reconciliation factorial, DEVELOPMENT seed 2026 only (text streams).
set -u
export PYTHONPATH=/workspace/mitosis-interference TRANSFORMERS_VERBOSITY=error HF_DATASETS_TRUST_REMOTE_CODE=1
cd /workspace/mitosis-interference
D=results/tfcl/recon_dev
{
for s in banking_rec banking_cil clinc_cil amazon_rec amazon_dil amazon_conflict amazon_dilconf; do
  o=$D/$s/firstseg_s2026.json; [ -f $o ] || echo "--stream $s --policy firstseg --seed 2026 --out $o"
  for p in single oracle; do o=$D/$s/${p}-noreplay_s2026.json; [ -f $o ] || echo "--stream $s --policy $p --replay 0 --seed 2026 --out $o"; done
done
} | xargs -P 1 -I{} sh -c 'python -m experiments.tfcl_run_x {} > /dev/null 2>&1 || echo FAIL {}'
echo RECON_DEV_DONE

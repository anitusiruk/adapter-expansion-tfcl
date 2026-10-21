#!/bin/bash
# Vision extension, DEVELOPMENT seed 2026 only: shadow traces and reference policies.
set -u
export PYTHONPATH=/workspace/mitosis-interference TRANSFORMERS_VERBOSITY=error HF_DATASETS_TRUST_REMOTE_CODE=1
cd /workspace/mitosis-interference
D=results/tfcl/vision_dev
{
for s in cifar_cil cifar_dil cifar_conf; do
  for k in $s ${s}_iid; do o=$D/shadow/${k}_s2026.json; [ -f $o ] || echo "--stream $k --policy shadow --seed 2026 --out $o"; done
  for p in frozen single oracle firstseg; do o=$D/$s/${p}_s2026.json; [ -f $o ] || echo "--stream $s --policy $p --seed 2026 --out $o"; done
  for p in single oracle; do o=$D/$s/${p}-noreplay_s2026.json; [ -f $o ] || echo "--stream $s --policy $p --replay 0 --seed 2026 --out $o"; done
done
} | xargs -P 1 -I{} sh -c 'python -m experiments.tfcl_run_x {} > /dev/null 2>&1 || echo FAIL {}'
echo VISION_DEV_DONE

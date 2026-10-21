#!/bin/bash
# tweet_conc DEVELOPMENT seed 2026 only (shadow + null for thresholds, reference policies)
set -u
export PYTHONPATH=/workspace/mitosis-interference TRANSFORMERS_VERBOSITY=error HF_DATASETS_TRUST_REMOTE_CODE=1
cd /workspace/mitosis-interference
D=results/tfcl/tweet_dev
{
for k in tweet_conc tweet_conc_iid; do o=$D/shadow/${k}_s2026.json; [ -f $o ] || echo "--stream $k --policy shadow --seed 2026 --out $o"; done
for p in frozen single oracle firstseg; do o=$D/tweet_conc/${p}_s2026.json; [ -f $o ] || echo "--stream tweet_conc --policy $p --seed 2026 --out $o"; done
} | xargs -P 2 -I{} sh -c 'python -m experiments.tfcl_run_x {} > /dev/null 2>&1 || echo FAIL {}'
echo TWEET_DEV_DONE

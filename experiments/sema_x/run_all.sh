#!/bin/bash
# Extension 5 (SEMA official code): variants x seeds, two queues. Skips finished outputs.
cd /workspace/mitosis-interference
export HF_HUB_ENABLE_HF_TRANSFER=0
q() { for job in "$@"; do v=${job%:*}; s=${job#*:}; o=results/sema_x/${v}_s${s}.json
  [ -f $o ] || /workspace/sema_venv/bin/python experiments/sema_x/run_sema_x.py --variant $v --seed $s --out $o > results/sema_x/${v}_s${s}.log 2>&1; done; }
q single:1993 noexp:1993 sema:1994 single:1995 noexp:1995 &
q sema:1993 single:1994 noexp:1994 sema:1995 &
wait; echo SEMA_X_DONE

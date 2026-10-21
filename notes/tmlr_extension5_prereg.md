# Extension 5 — the value of expansion inside a published system (SEMA official code)

Frozen 2026-10-11 ~04:45 UTC (session 3), after launching but before any extension-5 run had
completed its second task (only a smoke test's task-1 record, identical across variants since
no variant expands at task 1, had been seen: linear 98.6, ncm_ef 98.1, ncm_ex20 98.4).

## Why
Our triggers are stylised and our learner is minimal. SEMA (Wang et al., CVPR 2025) is a
complete exemplar-free self-expanding method with its own adapters (AdaptMLP), descriptors,
router, linear head and multi-epoch training. Its ablation compares expansion with adapters
frozen after the first task (first-session adaptation). The regime map predicts that in
SEMA's exemplar-free setting (linear head whose old-class rows are never refreshed), expansion
beats a single adapter set that keeps training, and that this advantage shrinks or reverses
when classifier statistics are refreshed from exemplars.

## Implementation
github.com/huiyiwang01/SEMA-CL commit 5a4551262d659c6d93dd6c50ff814e028554fbf3, unmodified,
config exps/sema_cifar.json (CIFAR-100, 10 tasks, ViT-B/16, 5 functional epochs, 20 descriptor
epochs). Wrapper experiments/sema_x/run_sema_x.py varies only the expansion decision and
adapter freezing (variants sema / noexp / single; see its docstring) and adds evaluation rules
linear, ncm_ef, ncm_ex20 (20 exemplars/class, identical indices across variants of a seed) and
ncm_full (final task). Deviations from the official run, identical for all variants: per-epoch
test evaluation disabled (progress bar only); TF32 matmuls; the expansion-detection forward
runs under no_grad (outputs only decide expansion; avoids a 14 GB autograd graph on a shared
GPU). Environment: separate venv with timm 0.9.8 and huggingface_hub 0.25.2 (torch 2.8).
Variant "expall" (expand at every task) was dropped before launch for compute.

## Design
Seeds 1993 (SEMA's published seed), 1994, 1995. Endpoint: final accuracy after task 10.
Decision rule (n = 3): supported if all three seed-level differences have the predicted sign
AND the 95% paired t-interval excludes zero; "directionally consistent" if only the former.

## Hypotheses
* E1 (exemplar-free value of expansion). linear: sema > single.
* E2 (refresh interaction). (sema - single | linear) - (sema - single | ncm_ex20) > 0.
* E3 (exemplar-refreshed). ncm_ex20: single > sema.
Descriptive: noexp vs sema and noexp vs single under every rule (SEMA's own ablation and
first-session adaptation); ncm_ef; ncm_full; average incremental accuracy; adapter counts.

## Frozen code (sha256)
    6213fb480ab4047bbe98373dade5a465c512af9293e7e5cde82ee0524db35e7b  experiments/sema_x/run_sema_x.py
    026b73998355a906545978dadee2ac5ecf3fbd2379bcfb402c562a17366b29f6  experiments/sema_x/run_all.sh
    5c8a73201555bec2b91788c1a30e7f98558d92e4c66f0de885c06e75de0bd780  experiments/sema_x/report.py  (analysis, committed before any extension-5 result)

## Addendum C1 — 2026-10-11 ~06:55 UTC (output bug; no effect on results)
The final json.dump in run_sema_x.py failed (SEMA's _set_device puts torch.device objects in
args). Every per-task record is printed to the run log as a JSON line before the dump, so
experiments/sema_x/recover.py rebuilds the result file from the log (marked
"recovered_from_log"); runs started after this point use json.dump(..., default=str).
Only the first completed run (single, seed 1993) had been seen: final linear 86.1, ncm_ef 87.2,
ncm_ex20 87.7, ncm_full 88.8. No hypothesis or analysis is changed.

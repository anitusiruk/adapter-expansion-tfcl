# Session 3 plan — priorities per GPU-hour (2026-10-11)

Goal: complete the analyses that matter most for the evaluation with the least GPU time; do not
cut anything that materially affects the conclusions.

## Kept (high value per GPU-hour)
| item | why it matters to reviewers | est. wall GPU |
|---|---|---|
| Extension 2 (exemplar-free regime, 724 runs, prereg) | the regime-map claim; abstract placeholder | ~6 h |
| Extension 1 remainder (vision 240, vision no-replay 48, shadows) | vision replication claimed in abstract | ~4 h |
| Extension 4: SDC refresh, CIL streams, single/oracle/firstseg | pre-empts "stale prototypes can be fixed without expansion" | ~1.5 h |
| ImageNet-R, reduced (5 seeds; single/oracle/firstseg x {exemplar, exemplar-free}) | canonical PTM-CL benchmark | ~2 h |
| SEMA external check (official code; sema/noexp/single, 3 seeds, CIFAR-100) | answers "stylised triggers / homemade learner" | measured after smoke test |
| BERT held-out backbone, reduced (single/oracle/loss_z/label_surprise, 10 streams, 5 seeds) | held-out backbone claim in abstract | ~3 h |

## Cut or deferred (low value per GPU-hour; reported as deviations)
* Primary q = 0.95 sensitivity (658 runs): the trigger conclusions rest on open-loop AUROC ~1 and
  attribution, not on the threshold; q=0.95 already exists for development seed. Not run.
* Exploratory E1/E2 (lora/head init, hparams; ~160 runs): not run.
* Slow-learner (lr 1e-4) confirmation: development (seed 2026) showed it does not reduce
  exemplar-free staleness (banking_cil proto_ef .131 vs .262 at lr 1e-3); reported as a
  development observation only.
* Attribution lora/no-lora (40 shadow runs): run only if GPU idles at the end.
* ImageNet-R triggers / shadows: not run (triggers are characterised on 15 other streams).

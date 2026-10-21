# Reframe decision — 2026-10-09

## Why the previous framing could not be submitted

1. **Learner competence.** On the exact BANKING A1 B1 A2 C1 B2 stream (seed 2027),
   a training-free frozen-DistilBERT mean-pooled nearest-class-mean classifier
   reaches 70.6% development accuracy (logistic probe 86.2%), while the CAU-v1 pools
   reported ~16% (frozen-centroid routing) and ~22% (learned router). The old
   learner took one AdamW step per 16-example batch at lr 2e-4 into a 77-way head;
   it was severely undertrained. Any accuracy comparison built on it is not
   credible to a continual-learning reviewer. (Script:
   quick sanity check, reproduced by `experiments/tfcl_run.py --policy frozen`.)
2. **Scope.** The audit findings concerned one bespoke allocator (CAU-v1). The
   central observation (allocation decisions are classifier/label-mass driven,
   not LoRA-capacity driven) is interesting only if it generalises across
   triggers and is tied to whether expansion actually helps.
3. **No positive control** where new capacity is needed by construction.

## New framing

"Capacity or classifier? What task-free expansion triggers detect."

* Clean testbed `src/tfcl/`: frozen encoder, own multi-LoRA (exact snapshot and
  restore), expand-and-freeze pool, fixed global reservoir (512 items regardless
  of pool size), replay, seen-label CE training, prototype (NCM) inference.
  A single package now reaches 86.4% on the BANKING recurrence stream (seed 2026).
* Stream families with known ground truth for "does expansion help?":
  class-incremental (BANKING rec/CIL, CLINC), same-label domain shift (Amazon),
  concept drift (Amazon with flipped domains; positive control).
* Triggers (stylised published signals): label novelty, loss spike (LACE /
  Online-LoRA), representation shift (SEMA), lookahead memory interference
  (CABLE / AdamW-I1), counterfactual fresh utility (CAU), head-equalised
  prototype interference (first proposal; failed to detect drift on pilot),
  known-label conflict loss (second proposal, derived from the loss
  decomposition: loss spike restricted to already-observed labels).
* Null calibration: thresholds = 0.99 quantile of each statistic on the
  i.i.d.-shuffled version of the stream (pilot seed 2026 only).

## Pilot observations (seed 2026; development only, not confirmatory)

* CIL: label novelty, loss spike and CAU fresh utility detect segment changes with
  AUROC 1.00 (BANKING-CIL, CLINC); 44–52% of the boundary loss is the
  label-group-mass term.
* Amazon domain shift: no statistic detects segment changes (AUROC ≈ 0.5).
* Oracle expansion at true boundaries *hurts* in CIL (BANKING-CIL 61.2–64.5% vs
  single 72.6%; CLINC 72.8% vs 83.2%) and helps slightly under domain shift
  (+1.3) and more under concept drift (+4 to +6).
* Prototype-interference statistic: AUROC 0.54 on concept drift — not a useful
  trigger. Retained and reported as a failed candidate.

The previous CAU-v1 evidence (head-only identity of allocation decisions across
10 seed/orders; ~82% of reset effect = label-group mass) becomes a case study.
All earlier results, failures and receipts remain in the repository unchanged.

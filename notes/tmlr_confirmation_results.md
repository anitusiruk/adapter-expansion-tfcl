# Confirmation study — results log (session 2)

Environment deviations (no effect on results; verified):
* New pod. Packages reinstalled at the pinned versions. Datasets re-downloaded; BANKING77 needs
  HF_DATASETS_TRUST_REMOTE_CODE=1 under datasets 3.6.0 (the loader is unchanged). A saved run
  (banking_rec single s2027) reproduced bit-for-bit (final proto_mean 0.8544746041297913),
  and the probe-restoration test passed. Frozen sha256 prefixes re-verified.
* Worker counts in experiments/tfcl_confirm.sh changed (6 -> 10 -> 8) for throughput; each
  run is an independent deterministic process, so this cannot change results. Some runs were
  restarted after interrupted grids; outputs are written only at completion.

## Interim (2026-10-10 ~03:00 UTC; development streams complete, held-out partial)

* H2 (expansion harms CIL): oracle - single = -8.2 (banking_rec), -11.4 (banking_cil),
  -10.0 (clinc_cil) points; 0 wins in every seed; loss_z / fresh_util / label_novel
  -4 to -10.5; all Holm-significant.
* H4: label_surprise never spawned in any CIL or DIL development stream -> identical to single
  (non-inferior).
* H3: amazon_dilconf oracle +1.9 [0.5, 3.3], label_surprise +2.1 [0.9, 3.3] (Holm-significant);
  amazon_conflict oracle +0.8 [-2.1, 3.8], label_surprise +1.0 [-1.9, 3.9] (NOT significant;
  development seed 2026 had shown +6). Random-timing expansion equals oracle on both drift
  streams (.615/.615, .582/.582).
* H1: fraction of loss_z / fresh_util spawns within 2 batches after a never-seen label:
  banking_cil 1.00/1.00, clinc_cil 1.00/1.00, but banking_rec 0.45/0.50 -> H1 FAILS on
  banking_rec as pre-registered.
* EXPLORATORY (post hoc, declared after seeing the H1 failure; experiments/tfcl_pkg_novelty.py):
  with *package-level* novelty (a label the newest package has never trained on; Proposition 1
  applies because its head row is zero), 100% of loss_z and fresh_util spawns fall within 2
  batches of package-level novelty in all CIL streams including banking_rec. Base rate of such
  batches among eligible batches: 0.27-0.30 (banking_rec), 0.25 (banking_cil), 0.15 (clinc_cil).
  Interpretation: in a recurrence stream, returning label groups are novel to the package
  spawned since they were last seen.
* Also notable: banking_rec random (.792) > oracle (.771): spawning exactly at label arrivals
  is worse than random timing.

## Primary q = 0.99 complete (2026-10-10 04:35 UTC) — 24 runs failed (GPU OOM from concurrent
extension jobs) and are being re-run unchanged (deterministic; results/tfcl/confirm_rerun_failed.log).
Numbers below are pre-rerun for clinc_cil (n = 8-9) and senti_conf (n = 6-8); final numbers will replace them.

* H2 holds in all four CIL streams incl. held-out news_cil (oracle -7.1, loss_z -7.3,
  fresh_util -7.3; all 10/10 seeds). label_novel on news_cil: -0.3 (n.s.; it spawned once).
* H4 holds in all 7 CIL/DIL streams (label_surprise spawns ~never; senti_dil +0.04).
* H3 FAILS on held-out senti_conf in the OPPOSITE direction: oracle -18.5 [-21.9,-15.1],
  label_surprise -17.1. Mechanism (per-segment trace): after the unflipped tweet segment,
  accuracy on the flipped Yelp segment collapses for the pool (0.16 with oracle) while the
  single package, trained with replay, fits the domain-conditional mapping (0.82 on Yelp at the
  end). The domains are distinguishable, so concept drift across them is not a capacity problem
  for one LoRA package with replay; frozen packages trained on unflipped data out-vote the
  correct one under prototype averaging.
* H1 as before: holds on banking_cil, clinc_cil, news_cil (100%); fails on banking_rec
  (45%/50%); exploratory package-level novelty explains 100% everywhere.

Implication for the paper: in exemplar-based pools the measured value of expansion is
negative in CIL, ~0 in DIL and mixed under drift (+1.9, +0.8 n.s., -18.5); LabelSurprise
detects the right *kind* of shift but detection is not value. The regime in which expansion
pays is tested by extension 2 (exemplar-free).

## Final primary q = 0.99 (all 1200 runs, after re-runs)
clinc_cil oracle -9.9 [-10.3,-9.5] 10/10; senti_conf oracle -17.5 [-20.5,-14.5], label_surprise
-16.7 [-20.2,-13.1] (0/10 wins). Other numbers unchanged from the interim entry.

## Extension 1 partial (2026-10-10 ~06:00)
XR1: oracle > firstseg banking_rec +7.2, banking_cil +5.5, clinc +6.9 (all 10/10, Holm), news +0.9 (n.s.).
XR2 (rehearsal-free oracle < single): banking_rec -4.3 (sig); banking_cil -3.3, clinc -0.6, news -0.6
(n.s.) -> mostly NOT supported. DiD (no-replay effect minus replay effect): +3.9, +8.1, +9.3, +6.8
(all sig): the harm of expansion in CIL is largely a consequence of rehearsal with exemplars.

## Session 3 deviations (2026-10-11) — compute, secondary analyses only
* q = 0.95 sensitivity grid (secondary/descriptive in the pre-registration): NOT run.
* BERT held-out backbone (secondary/descriptive): run as a strict subset of the declared grid —
  policies single, oracle, trigger:loss_z, trigger:label_surprise; all 10 streams; seeds
  2027-2031; same thresholds (experiments/tfcl_bert_reduced.sh). random and the other six
  triggers are not run.
* Attribution (--no-lora shadows) and exploratory E1/E2: deferred; run only if the GPU idles.
* Extension 2: at session start a second copy of the extension-2 pipeline was accidentally
  launched for ~2 minutes; three runs were written by the duplicate and the original then
  logged "refusing to overwrite" FAIL lines for them. Runs are deterministic (bit-exact
  reproduction verified), so outputs are unaffected.

## Deviation (session 3, 2026-10-11 ~15:30 UTC): BERT runs invalid — missing tokenizer files
The first execution of the BERT subset (200 runs, 11:12–14:50 UTC) is INVALID and is not analysed.
Cause: the bert-base-uncased snapshot (revision 86b5e09) in the re-created HF cache contained only
config.json and model.safetensors; the tokenizer files had failed to download during the HF rate
limit at 03:36 UTC. In offline mode transformers 5.18 silently built a BertTokenizer with an empty
vocabulary, mapping every word to [UNK] (accuracy at chance in every stream). Detected because the
results were at chance; confirmed by re-running the development shadow run (banking_cil_iid,
seed 2026): 0.0097 with the broken cache vs 0.7879 recorded in development. After downloading
tokenizer.json, tokenizer_config.json and vocab.txt at the pinned revision, masked-LM sanity check
returns "paris" and the development run reproduces bit-exactly (0.7878961307661874; per-segment
values identical). DistilBERT's snapshot was complete, so no other result is affected.
The invalid outputs are kept in results/tfcl/confirm_bert_INVALID_no_tokenizer/. The BERT subset was
re-launched unchanged (same command, seeds, thresholds) into results/tfcl/confirm_bert.

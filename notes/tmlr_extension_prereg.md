# Extension study — declared 2026-10-10, before any extension confirmation run

Written in session 2. Before this file was committed, NO result of the primary
confirmation study (results/tfcl/confirm*, seeds 2027–2036) had been inspected by the
author of this file (only completion counts were checked), and no extension run with a
seed other than 2026 had been executed. Development of the extension used seed 2026
only (results/tfcl/vision_dev, results/tfcl/recon_dev, results/tfcl/tweet_dev).

The frozen files of notes/tmlr_confirmation_prereg.md are unchanged (sha256 verified at
the start of session 2). The extension adds new files only:

    src/tfcl/vision.py           ViT-B/16 (google/vit-base-patch16-224-in21k) encoder + CIFAR-100 streams
    src/tfcl/data_x.py           tweet_conc natural concept-drift stream
    experiments/tfcl_run_x.py    wrapper around the frozen runner (+ policy firstseg)
    experiments/tfcl_grid_x.py   grid runner for tfcl_run_x
    experiments/tfcl_extension_report.py   analysis below (committed with this file)
    experiments/tfcl_lace_openloop.py      open-loop LACE-style rule (XL)

(sha256 of these files at freeze time are listed at the end of this file.)

## Motivation (why the extension exists)

1. Generality: the primary study is text-only with small encoders. Most expansion work
   (SEMA, EASE, Online-LoRA) uses ViT-B/16 on image benchmarks.
2. An apparent contradiction: SEMA's ablation reports that self-expansion improves on
   first-session adaptation in class-incremental learning, while our development data show
   expansion below a single continually trained package. Development (seed 2026, text)
   suggests both are true: oracle > firstseg but oracle < single in CIL. The extension tests
   whether the value of expansion is baseline-dependent.
3. Realism of the drift positive control: all primary drift streams use synthetic label
   flips. tweet_conc is a naturally occurring change of annotation concept (TweetEval
   offensive -> hate -> irony -> offensive, shared label ids). Development showed only a
   small oracle advantage (+1.8 points, seed 2026), so it is reported descriptively.
4. Faithfulness of triggers: LACE (Tathe, 2026) reports 100% boundary precision for a
   loss-ratio rule with a published threshold (tau = 2.5). We evaluate that rule open-loop.

## Design

Seeds: 2027–2036 for text extensions; 2027–2034 for vision (8 seeds: the smallest number
for which a two-sided Wilcoxon signed-rank test can reach p < 0.01; compute is one RTX 4090
shared with the primary pipeline). Vision forward passes use bf16 autocast (decided before
any vision result was seen; all vision runs, including the seed-2026 null calibration,
use it). Seed is the unit of analysis. proto_mean final mean dev
accuracy is the endpoint, as in the primary study.

* XV (vision replication). Streams cifar_cil (CIL), cifar_dil (DIL), cifar_conf (DRIFT).
  Policies: frozen, single, oracle, random, firstseg, trigger:{label_novel, loss_z, repr_z,
  fresh_util, label_surprise} at q = 0.99 (thresholds: i.i.d.-null shadow traces, seed 2026,
  results/tfcl/vision_dev/shadow/report_seed2026.json). interf_logit, interf_proto and
  conflict_z are omitted for compute (the last two are failed development candidates).
  Rehearsal-free variant (--replay 0): single and oracle.
* XR (baseline dependence of expansion). Text streams of the primary study plus
  tweet_conc: policy firstseg (seeds 2027–2036); single and oracle with --replay 0
  (seeds 2027–2036); tweet_conc also single, oracle, random, frozen and the five triggers
  above (thresholds results/tfcl/tweet_dev/shadow/report_seed2026.json).
* XL (open-loop LACE-style rule) and open-loop AUROCs: single-package shadow traces for
  every text stream (incl. tweet_conc) on seeds 2027–2031 and vision streams on seeds
  2027–2029.

## Hypotheses (directional; paired seed-level t-CI + Wilcoxon, Holm within family)

* XV1. In cifar_cil, >= 80% of loss_z and fresh_util spawns fall within 2 batches after a
  batch with a never-seen label (pooled over seeds).
* XV2. In cifar_cil: oracle < single; loss_z, fresh_util, label_novel pools < single.
* XV3. In cifar_conf: oracle > single and label_surprise > single.
* XV4. label_surprise is non-inferior to single (lower 95% bound > -1.0 point) in cifar_cil
  and cifar_dil. label_novel spawns nothing in cifar_dil / cifar_conf.
* XR1. In every CIL stream (banking_rec, banking_cil, clinc_cil, news_cil, cifar_cil):
  oracle > firstseg (expansion beats first-session adaptation).
* XR2. In every CIL stream, without training replay: oracle < single (the harm of
  expansion in CIL does not depend on rehearsal).

Descriptive only (no hypothesis): tweet_conc contrasts; difference-in-differences of the
expansion effect with vs. without replay; XL firing counts (fraction of firings within 2
batches of label novelty in CIL, firings within 2 batches of a segment change in DIL/DRIFT);
open-loop AUROCs on confirmation seeds; all other contrasts.

## Development observations recorded before freezing (seed 2026; not confirmatory)

* Vision shadow AUROCs (k = 2 batches after a segment change): cifar_cil loss_z 1.00,
  fresh_util 1.00, label_surprise 0.02; cifar_dil loss_z 0.96, label_surprise 0.92;
  cifar_conf loss_z 1.00, label_surprise 0.97. LACE-style ratio rule (tau 2.5) fired at
  9/9 class-group arrivals of cifar_cil.
* Therefore XV4 for cifar_dil is AT RISK: LabelSurprise also responds to within-label
  subpopulation shift (a new CIFAR subclass under an existing superclass label), and
  oracle expansion is slightly below single on cifar_dil in development (.864 vs .872).
  The hypothesis is kept as declared; a failure will be reported as a limitation
  (label-conditional loss rises under P(x|y) shift as well as under P(y|x) shift).
* Vision CIL development: single .848, frozen .773, oracle .739, firstseg .580;
  no replay: single .764, oracle .752. Text CIL development: oracle > firstseg and
  oracle < single, with and without replay, on all three development CIL streams.

Deviations will be recorded in notes/tmlr_extension_results.md.

## Frozen extension code (sha256)

    033cac7ef39b69990e96b8c63c19ed801efd0d1426976568cab1b5441cf7428f  src/tfcl/vision.py
    97c45ecb78e1838c8af1865b716e16b663428921f2852bc073ccf65b5e393e0f  src/tfcl/data_x.py
    e417a386b37b2ebea7a7f045bdec3178713ba1fb906b0eebb19abe4add98c593  experiments/tfcl_run_x.py
    6de767813109689d51b669e92001aa63200d17569ed195d4eb249972e8e4cb36  experiments/tfcl_grid_x.py
    3a74126ced7675ed8b990af51a44efe01e646b635eff33609262be41e813b10a  experiments/tfcl_extension_report.py
    c3ba7a075681339654ac45bef7d34fd0f024689e9c42e82fccd358cf0a00f096  experiments/tfcl_lace_openloop.py
    625bdc96347e4e887ec4e76a329061483e0700b0f3791c454a3c52ce924aaa9c  experiments/tfcl_extension.sh
    556dcfe7d5cc44c02bc66546b53cd35214a50ea3785c6d02cdeb9d1b6d054e72  results/tfcl/vision_dev/shadow/report_seed2026.json
    8836a8ab10f4cd6d1cfc5108aa1ee1658a2052695e82dd1c4ef4eb1d49201ab1  results/tfcl/tweet_dev/shadow/report_seed2026.json

## Addendum A1 — declared 2026-10-10 ~03:10 UTC, before any ext_vision run had started

Development observation (seed 2026, cifar_conf): with the pre-registered proto_mean rule,
oracle expansion is BELOW single (.672 vs .800), but with ownership routing (proto_own: each
package scores only the classes in its own memory items, best package-class pair wins)
oracle is ABOVE single (.845 vs .800); cifar_dil similarly .906 (own) vs .872.
The text primary results had been partially seen at this point (amazon drift streams:
oracle - single small; proto_own did not help there), so this addendum concerns vision only.

* XV3 is unchanged and remains as declared (proto_mean; now expected to fail).
* XV3-own (added): in cifar_conf, oracle > single under proto_own (paired, same tests).
* XV-own-dil (descriptive): oracle - single under proto_own in cifar_dil.
These are computed by running experiments/tfcl_extension_report.py logic with
metric = proto_own (load(..., metric="proto_own")); no other change.

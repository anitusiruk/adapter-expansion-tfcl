# Extension 6 — audit controls: matched refresh, replay access, closed-loop false spawns

Frozen 2026-10-12 ~01:10 UTC (session 4), before any extension-6 confirmation run. Written in
response to an external methodological audit (summarised in the paper's revision notes). Frozen
files of earlier stages are not modified; all new code is in new files.

Seen before freezing (integrity checks of the new code, not tuning): four runs on BANKING-CIL,
seed 2027 (single/oracle with --mr --replay 0; single with --global-replay, which reproduced the
saved single run bit-for-bit; oracle with --global-replay at dev seed 2026). The two --mr runs
reproduced the saved proto_ef/proto_mean bit-for-bit and showed: single ms .334 / mf .615;
oracle ms .488 / mf .580 (one seed of block M). Global-replay oracle, dev seed 2026: .616.
Nothing was changed after these runs.

## Why
1. proto_ef (stored) and proto_mean (refreshed from memory) differ in sample, feature age,
   class-package coverage and aggregation at once, so the within-run difference-in-differences
   (extension 2, R2) does not isolate staleness (= feature age).
2. Replay draws only items owned by the active package, so a new package cannot rehearse examples
   observed under older packages; "same memory budget" is not "same replay access".
3. A common null quantile targets comparable *marginal* exceedance rates; closed-loop spawning
   (two consecutive exceedances, warm-up, evolving windows) may give different false-spawn rates.

## Block M — matched stale vs refreshed prototypes (src/tfcl/matched_refresh.py)
Exemplar-free regime (--replay 0), policies single and oracle, rules proto_ms (reservoir items,
coverage = packages existing at observation, features recorded at observation time), proto_mf
(identical items, coverage and aggregation; features recomputed under current parameters),
proto_mfa (identical items, all packages). Streams/seeds: banking_rec, banking_cil, clinc_cil,
news_cil (2027-2036), cifar_cil (2027-2034), imnr_cil (2027-2032). Training identical to the
saved extension-2/3 runs (integrity: proto_ef and proto_mean must match them exactly).
* M1: single package, proto_mf - proto_ms > 0 (feature age alone costs a continuing package).
* M2 (primary): (oracle - single | proto_ms) - (oracle - single | proto_mf) > 0 (refreshing only
  feature age reduces the value of expansion).
* Descriptive: oracle - single under each rule (incl. whether its sign under proto_mf differs
  from proto_ms); telescoping decomposition of the extension-2 DiD into sample (ef->ms),
  feature age (ms->mf), coverage+aggregation (mf->mfa), check (mfa->mean).

## Block G — expansion with global replay access
Exemplar-based regime (replay 16), policy oracle with --global-replay (new packages rehearse the
whole reservoir). Compared with the saved single and owned-replay oracle runs (same seeds).
Streams: all ten primary text streams (2027-2036), cifar_cil (2027-2034), imnr_cil (2027-2032).
Integrity: single with --global-replay equals saved single (text: verified; cifar s2027 rerun).
* G1: in class-incremental streams, oracle_global - single < 0 (expansion still costs accuracy
  with equal replay access).
* Descriptive: oracle_global - oracle_owned (share of the harm attributable to replay access);
  domain-shift and drift streams.

## Block F — closed-loop false spawns on independent shuffled null streams
Triggers run closed-loop with their confirmation thresholds (q = 0.99 null quantile, seed-2026
shadow reports) on <stream>_iid at seeds the thresholds never saw (each seed draws different
examples and a different shuffle). Every spawn there is false.
Exemplar-based text: ten primary streams x {label_novel, loss_z, repr_z, interf_logit,
fresh_util, label_surprise} x seeds 2027-2031. Exemplar-free text (--replay 0, extension-2
thresholds): four CIL streams x {loss_z, label_surprise} x 2027-2031. Vision: cifar_cil/dil/conf
x {label_novel, loss_z, repr_z, fresh_util, label_surprise} x 2027-2029.
* Descriptive only: spawns per run, fraction of runs with any spawn, spawns per 100 eligible
  batches, realised marginal exceedance, per trigger. The paper will describe calibration as
  quantile calibration whatever the outcome, and report these rates.

## Re-analyses of saved results (no new runs; in the same report script)
* A4: oracle - single under proto_mean within the exemplar-free runs, with t-intervals.
* A6: SEMA interaction with NCM on both sides: (sema - single | ncm_ef) - (sema - single | ncm_ex20).

## Decision rule
Streams with >= 8 seeds: two-sided Wilcoxon signed-rank, Holm-corrected across the five such CIL
streams within each hypothesis; supported if Holm p < 0.05 and the mean has the predicted sign.
imnr_cil (6 seeds): supported if the 95% t-interval excludes zero on the predicted side (as in
extension 3). Unit of analysis: seed. Endpoint: final mean accuracy over segments.

## Frozen code (sha256)
    b7a56143c28ef841891a82a3f70b6cf6eb7973829fbf1d0305de81b21d696c13  src/tfcl/matched_refresh.py
    756f0e7347382ebc94da8946df643848f388e7db68377ffb4b1c14ba6e043d80  experiments/tfcl_run_x6.py
    02f54e0a094cd591b4559bc30a6312550a98d3d44d0197db699a9f38d5083785  experiments/tfcl_ext6.sh
    fade89b28f849dcbf08269f9c57fbce6cae86959da8ea51da1e9945d4e6961a1  experiments/tfcl_ext6_report.py (analysis)
    6ae10e762c6a9df19dbd50666f964890c5113d2580f8de6cc1cb1036d0ca7637  experiments/run_jobs_fast.py (scheduling only)

## Addendum X1 — 2026-10-12 ~03:30 UTC (exploratory, declared after seeing block-F partial results)
Seen: FreshUtil cascades on shuffled CLINC (5/5 seeds, 41 spawns, every inter-spawn gap = 9 batches,
i.e. it re-fires at the first eligible window of every new package) and on BANKING-CIL/REC (2-3 of 5
seeds); 0 spawns on 20 Newsgroups/Amazon. Interpretation to test: each spawn creates package-level label
novelty (all labels already observed globally, new package's rows untrained or pushed down), which the
fresh-utility statistic rewards. Exploratory test (not a hypothesis of this stage): rerun the same
closed-loop null runs with --head-init inherit (new package copies the previous head; existing learner
option), clinc_cil_iid and banking_cil_iid, seeds 2027-2031, same thresholds. Prediction: cascades
disappear or shrink. Output results/tfcl/ext6/null_closed_inherit/. Reported as exploratory.

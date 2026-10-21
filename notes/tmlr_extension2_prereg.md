# Extension 2 — the exemplar-free regime (regime map). Frozen 2026-10-10, before any extension-2 run with a confirmation seed.

Written in session 2. At writing time the primary confirmation results for the development
streams had been inspected (interim log in notes/tmlr_confirmation_results.md); extension-1
first-segment and partial rehearsal-free results had been inspected. No extension-2 run with
a seed other than 2026 had been executed.

## Rationale (logic, not tuning)

Rehearsal-free pre-trained-model learners (APER/ADAM, EASE, SEMA) keep no exemplars, so a
class prototype is the mean of features computed when the class was observed and is never
recomputed. If the package that produced those features keeps training, the prototypes of
earlier classes become stale (feature drift); a frozen package's prototypes stay valid. Hence:

* In an exemplar-based learner, the classifier statistics can be refreshed under the current
  package, and the cost of continued training is small: expansion buys little and fragments
  training data (primary study: expansion hurts in CIL).
* In an exemplar-free learner, freezing a package when new classes arrive preserves the
  validity of the prototypes recorded so far: expansion at (package-level) label novelty
  should help, and a trigger that never spawns in CIL (LabelSurprise) should do badly.

The value of a trigger that detects label novelty is therefore regime-dependent. Proposition 1
says what loss-type triggers detect; this extension tests when that is the right moment to
expand.

## Implementation

src/tfcl/exemplar_free.py (mixin; records per-package running class sums of normalised
features at observation time; rule proto_ef), experiments/tfcl_run_x2.py (adds --ef to
tfcl_run_x; otherwise identical), experiments/run_jobs.py (job runner). Training uses
--replay 0 (no rehearsal in the loss). Endpoint: final mean dev accuracy under proto_ef.
Triggers are re-calibrated for this regime: thresholds are the q = 0.99 quantiles of each
statistic on i.i.d.-shuffled null shadow traces run with --replay 0, seed 2026
(results/tfcl/ef_dev/shadow/report_seed2026.json), because the statistics depend on the
training trajectory, which --replay 0 changes (--ef does not change training).

## Design

Streams: banking_rec, banking_cil, clinc_cil, news_cil (CIL); amazon_dil, senti_dil (DIL);
amazon_dilconf, senti_conf (DRIFT); cifar_cil, cifar_dil, cifar_conf (vision).
Seeds: 2027–2036 (text), 2027–2034 (vision).
Policies (all with --replay 0 --ef): single, oracle, random, firstseg,
trigger:{label_novel, loss_z, label_surprise} at q = 0.99.

## Hypotheses (paired seed-level t-CI + Wilcoxon; Holm within family)

* R1 (regime flip). In every CIL stream (4 text + cifar_cil), under proto_ef: oracle > single.
* R2 (interaction, isolating the refresh factor). Every extension-2 run also reports the
  exemplar-based rule proto_mean on the identical trajectory. In every CIL stream:
  (oracle - single | proto_ef) - (oracle - single | proto_mean) > 0 (paired by seed).
* R3 (triggers). In every CIL stream, under proto_ef: loss_z > label_surprise.
* R4 (timing). In every TEXT CIL stream (banking_rec, banking_cil, clinc_cil, news_cil), under
  proto_ef: oracle > random. (cifar_cil is descriptive for R4: in development random .804 > oracle .764.)

Descriptive: DIL and DRIFT streams; firstseg; label_novel; package counts.

## Development observations (seed 2026, before freezing)

proto_ef, replay 0 (single / oracle / random / firstseg / loss_z / label_surprise):
banking_rec .444/.738/.296/.675/.740/.433; banking_cil .262/.566/.349/.606/.564/.262;
clinc_cil .502/.775/.688/.698/.780/.502; cifar_cil .457/.764/.804/.698/.767/.457;
amazon_dil .855/.861/.848/.848/.855/.855; amazon_dilconf .525/.570/.580/.581/.577/.577;
cifar_conf .646/(oracle failed: OOM, re-run)/.718/.653/.688/.688.
Dev trigger runs used the replay-16 thresholds; confirmation uses the replay-0 recalibration.
Primary-study results had been seen (expansion harmful in exemplar-based CIL; senti_conf -18.5).

## Frozen code (sha256)

    d7b49f737a56ee405ea4aa54ee52efe875c73a3236be0213ef0a9da9595092f0  src/tfcl/exemplar_free.py
    252a6b316b37afc318ed834836f0eefc151388bac5e3a83636b81893898d13ba  experiments/tfcl_run_x2.py
    6b740b592c7a21194ba1000f515cb06dd556e4e419ec140c6185069b8ad28d9e  experiments/run_jobs.py
    c7089f2108dd825f7bef3f75c90ec95387c3f889bc9fdde8a2be1ae90905896e  experiments/tfcl_extension2.sh
    b0ff558d22d9f01b8b1de882d61c018371b3032ae31e994e60db21cb2d7106fb  experiments/tfcl_extension2_report.py
    356cee8ad769acfda9bd15b89073ef2b07080611b273cd98eb6b2693e389826c  results/tfcl/ef_dev/shadow/report_seed2026.json

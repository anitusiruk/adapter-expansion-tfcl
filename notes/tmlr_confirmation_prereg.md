# Pre-registered confirmation study — frozen 2026-10-09 before any confirmation run

Development used seed 2026 only (shadow traces and closed-loop pilots on the seven
development streams). Development history, including the failed trigger
candidates (`interf_proto`: did not detect drift; `conflict_z`: v1 fired on
recently-introduced labels, v2 fired on newly matured labels in CIL) and the
z-window alarm fix, is preserved in `results/tfcl/*superseded*`,
`results/tfcl/pilot_closed_v1_window_contaminated`, `pilot_closed_v2`,
`pilot_closed_v3` and git history. No confirmation seed (2027–2036) and no
non-null run of a held-out stream was executed before this file was committed.

## Frozen code and calibration (sha256)

    830c4a67d7e64903b7dff237d16990a8f449ba2a9b325bf9ec9ea807a022a8e7  src/tfcl/data.py
    4e792d72534a738547b2cf020d2e8fdef214a0f440126ea5f38338b160a5fc27  src/tfcl/learner.py
    79d4907e59554d2589946a8d7be88852189f5d9ed81b7f14e6cabe8764099dff  src/tfcl/model.py
    b55a7fdb7e9198a41a13aea129f804b68a9b7b388be6b09336eeeae41e12245e  src/tfcl/triggers.py
    cec0a21fd5dcf6b333f859700e3a7da08c9b2539fe91c7742415e5eee98097ed  experiments/tfcl_run.py
    e1e998dd56d7533f3240a926a54503015bd4dcfdd8d3cd850e0dc28d502450e1  experiments/tfcl_grid.py
    9f6ed0bb74d6943d21b8c3c25cd9729c5b9d6e45ea1831acc78eb963b8a37a1a  results/tfcl/shadow/report_seed2026.json
    a0863570889d21475a17b8147a6059043f2be78a9811ee6561dd836a7289802b  results/tfcl/shadow_bert/report_seed2026.json

Learner: lr 1e-3, 2 AdamW steps per 16-example batch, 16 replay items from the
trained package's own memory, 512-item global reservoir, LoRA r=8/alpha=16 on
Q/V, fresh LoRA and fresh head at spawn, max length 64. Triggers: warmup 8,
confirmation 2, window 20. Thresholds: q-quantile of the statistic on the
i.i.d.-shuffled null of the same stream, seed 2026, same backbone.

## Design

* Development streams (D): banking_rec, banking_cil, clinc_cil (class-incremental,
  CIL); amazon_rec, amazon_dil (same-label domain shift, DIL); amazon_conflict,
  amazon_dilconf (concept drift, DRIFT).
* Held-out streams (H), never run except as i.i.d. nulls: news_cil (CIL),
  senti_dil (DIL), senti_conf (DRIFT).
* Seeds 2027–2036 (10). The seed is the unit of analysis (it changes stream
  sampling/order, LoRA initialisation, replay and reservoir randomness).
* Policies: frozen, single, oracle, random (oracle count, random times), and
  triggers label_novel, loss_z, repr_z, interf_logit, fresh_util, interf_proto,
  conflict_z, label_surprise at q = 0.99 (primary) and q = 0.95 (sensitivity).
* Held-out backbone: bert-base-uncased, seeds 2027–2031, D and H streams,
  policies single, oracle, random and all triggers at q = 0.99.
* Attribution: single-package shadow traces with and without LoRA training
  (`--no-lora`) for CIL streams, seeds 2027–2031.

## Primary endpoint and contrasts

Endpoint: final mean development accuracy over segments, `proto_mean` inference.
Contrasts are paired by seed; report mean difference, 95% t-interval and
two-sided Wilcoxon signed-rank p; Holm correction within each hypothesis family.

* H1 (label-novelty attribution). In CIL streams, the spawn steps of loss_z and
  fresh_util fall within 2 batches after a batch containing a never-seen label
  for at least 80% of spawns (pooled over seeds per stream), and the label-group
  share of the boundary loss exceeds 0.3.
* H2 (expansion harms CIL). In each CIL stream: oracle < single; and loss_z,
  fresh_util and label_novel pools < single.
* H3 (expansion helps drift). In each DRIFT stream: oracle > single, and
  label_surprise > single. label_novel spawns nothing in DIL/DRIFT.
* H4 (specificity, non-inferiority). In every CIL and DIL stream, the
  label_surprise pool is non-inferior to single with margin 1.0 accuracy point
  (lower 95% bound of label_surprise − single > −0.010).

Everything else (other inference rules, q = 0.95, BERT, random control,
package counts, AUROCs, per-segment accuracies) is secondary and descriptive.
Any deviation from this plan will be recorded as such in the results notes.

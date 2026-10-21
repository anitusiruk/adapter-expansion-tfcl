# Exploratory robustness studies — declared 2026-10-09, before any confirmation result was inspected

These are exploratory (not part of the pre-registered hypotheses). They test whether the
*measured value of expansion* (oracle vs single) depends on design choices of the pool.

E1 (expansion design): streams banking_cil, clinc_cil, amazon_dil, amazon_conflict; seeds 2027-2031;
policies single and oracle with lora_init in {fresh, inherit} x head_init in {fresh, inherit};
all inference rules reported (proto_mean, proto_max, proto_own, linear_last).
E2 (learner hyperparameters): streams banking_cil, amazon_conflict; seeds 2027-2031; single and oracle
with (lr, steps) in {(5e-4, 2), (2e-3, 2), (1e-3, 1)}.
Outputs: results/tfcl/explore_design, results/tfcl/explore_hparams. Runs start only after the
confirmation pipeline finishes.

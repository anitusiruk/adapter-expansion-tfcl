# Capacity or Classifier? When Adapter Expansion Pays in Task-Free Continual Learning

Code, pre-registration documents and saved run outputs for the paper.

## Layout

- `src/tfcl/`: common LoRA learner, triggers, streams (text and vision), exemplar-free, matched-refresh and drift-compensation rules.
- `experiments/tfcl_*`: run scripts, job grids, reports, tables and figures. `experiments/sema_x/` holds the SEMA check.
- `notes/tmlr_*_prereg.md`: pre-registration documents with the SHA-256 hashes of the frozen code; `notes/tmlr_confirmation_results.md` logs results and deviations.
- `results/tfcl/`, `results/sema_x/`: every saved run (JSON) and report used in the paper.
- `tests/`: bitwise restoration test for trigger probes.

## Reproducing

Install `requirements-frozen.txt` (plus `requirements-plotting.txt` for figures) and run from the repository root with `PYTHONPATH=.`.

- Tables and statistics from saved runs (CPU): `experiments/tfcl_confirm_report.py`, `tfcl_extension_report.py`, `tfcl_extension2_report.py`, `tfcl_ext34_report.py`, `tfcl_ext6_report.py`, `tfcl_paper_tables.py`.
- Figures (CPU): `tfcl_fig_regime.py`, `tfcl_fig_mechanism.py`, `tfcl_staleness_decomp.py`.
- Re-running the experiments (GPU): the `experiments/tfcl_*.sh` pipelines, which call `tfcl_run*.py` through `run_jobs.py`.

Check the frozen code against the pre-registered hashes with `sha256sum` on the files listed in each `notes/tmlr_*_prereg.md`.

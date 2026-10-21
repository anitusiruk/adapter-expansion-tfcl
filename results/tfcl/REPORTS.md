# Index of analysis reports (results/tfcl)

| File | Stage / content | Produced by |
|---|---|---|
| confirm_report_q0.99.json | Primary study, DistilBERT, q=0.99 (H1-H4) | experiments/tfcl_confirm_report.py |
| confirm_report_bert.json (+ bert_report.txt) | Primary grid subset with held-out BERT-base (renamed from the script's default name confirm_report.json in session 4; contents unchanged) | experiments/tfcl_confirm_report.py --dir results/tfcl/confirm_bert |
| extension_report.json (+ ext1_report.txt) | Extension 1: vision, first-segment reference, no replay | experiments/tfcl_extension_report.py |
| extension_report_partial.json | Superseded partial version of the above (kept for the record) | same |
| extension2_report.json (+ ext2_report*.txt) | Extension 2: exemplar-free regime (R1-R4) | experiments/tfcl_extension2_report.py |
| extension34_report.json (+ ext34/ext4_report.txt) | Extensions 3-4: ImageNet-R (I1-I4) and drift compensation (S1-S2) | experiments/tfcl_ext34_report.py |
| ../sema_x/report.json | Extension 5: SEMA official code (E1-E3) | experiments/sema_x/report.py |
| ext6_report.json | Extension 6 (audit controls): matched refresh (M), global replay (G), closed-loop null spawns (F), re-analyses A4/A6 | experiments/tfcl_ext6_report.py |
| return_rows.json | Exploratory: head rows of returning labels (not pre-registered) | experiments/tfcl_return_rows.py |
| lace_openloop.json | LACE-style rule applied open-loop | experiments/tfcl_lace_openloop.py |

#!/bin/bash
# Extension pipeline exactly as declared in notes/tmlr_extension_prereg.md.
set -u
export PYTHONPATH=/workspace/mitosis-interference TRANSFORMERS_VERBOSITY=error HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/mitosis-interference
TEXT="banking_rec banking_cil clinc_cil amazon_rec amazon_dil amazon_conflict amazon_dilconf news_cil senti_dil senti_conf tweet_conc"
TS="2027 2028 2029 2030 2031 2032 2033 2034 2035 2036"
VS="2027 2028 2029 2030 2031 2032 2033 2034"
G="python -m experiments.tfcl_grid_x"
W=${EXT_WORKERS:-3}
# XR: first-segment baseline; rehearsal-free single/oracle (text)
$G --outdir results/tfcl/ext_recon --streams $TEXT --seeds $TS --workers $W --policies firstseg
$G --outdir results/tfcl/ext_recon_noreplay --streams $TEXT --seeds $TS --workers $W --policies single oracle --extra --replay 0
# tweet_conc: full policy set (thresholds from its own seed-2026 null)
$G --outdir results/tfcl/ext_recon --streams tweet_conc --seeds $TS --workers $W --q 0.99 \
   --taus results/tfcl/tweet_dev/shadow/report_seed2026.json \
   --policies frozen single oracle random trigger:label_novel trigger:loss_z trigger:repr_z trigger:fresh_util trigger:label_surprise
# XL / open-loop shadows (text, seeds 2027-2031)
$G --outdir results/tfcl/ext_shadow --streams $TEXT --seeds 2027 2028 2029 2030 2031 --workers $W --policies shadow
# XV: vision
VW=${EXT_VWORKERS:-2}
$G --outdir results/tfcl/ext_vision --streams cifar_cil cifar_dil cifar_conf --seeds $VS --workers $VW --q 0.99 \
   --taus results/tfcl/vision_dev/shadow/report_seed2026.json \
   --policies single oracle firstseg frozen random trigger:label_novel trigger:loss_z trigger:repr_z trigger:fresh_util trigger:label_surprise
$G --outdir results/tfcl/ext_vision_noreplay --streams cifar_cil cifar_dil cifar_conf --seeds $VS --workers $VW --policies single oracle --extra --replay 0
$G --outdir results/tfcl/ext_shadow --streams cifar_cil cifar_dil cifar_conf --seeds 2027 2028 2029 --workers $VW --policies shadow
echo EXTENSION_PIPELINE_DONE

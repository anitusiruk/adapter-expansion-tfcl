#!/bin/bash
# Re-run an idempotent job script until it reports no FAIL lines (OOM retries; runs are deterministic).
for i in $(seq 1 20); do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$1" > "$2.try$i" 2>&1
  grep -qE "FAIL --|fail\(|FAILED: \[" "$2.try$i" || { echo "clean after $i tries" >> "$2"; exit 0; }
  echo "try $i: $(grep -cE "FAIL --|fail\(" $2.try$i) failures" >> "$2"; sleep 60
done

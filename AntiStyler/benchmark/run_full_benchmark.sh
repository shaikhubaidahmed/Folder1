#!/bin/bash
# Runs the corrected Table 1 benchmark: Google Patch + DPatch in parallel
# (each ~2.97GB peak GPU memory, safe within this 8GB GPU), then M-PGD
# afterward. Each attack writes to its own --out-dir to avoid the two
# parallel processes colliding on results/scores.json.
cd "$(dirname "$0")"

python3 evaluate.py --attacks google --out-dir results_google > google_run.log 2>&1 &
PID_GOOGLE=$!

python3 evaluate.py --attacks dpatch --out-dir results_dpatch > dpatch_run.log 2>&1 &
PID_DPATCH=$!

wait $PID_GOOGLE $PID_DPATCH

python3 evaluate.py --attacks mpgd --out-dir results_mpgd > mpgd_run.log 2>&1

echo "ALL ATTACKS COMPLETE" >> run_full_benchmark.log

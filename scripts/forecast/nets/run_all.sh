#!/usr/bin/env bash
# The 23 training runs of plan step 3.3 (Task 9), one per line; re-running skips finished runs and resumes the others.
# Usage: bash scripts/forecast/nets/run_all.sh [extra train_nets.py args, e.g. --device cpu]
# Times: experiments/forecast/nets/runs.log
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
LOG=experiments/forecast/nets/runs.log
mkdir -p experiments/forecast/nets
run() {
  local t0=$SECONDS
  echo "$(date '+%F %T') inizio: $*" >> "$LOG"
  "$PY" scripts/forecast/nets/train_nets.py "$@" ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee -a "$LOG"
  echo "$(date '+%F %T') fine ($((SECONDS - t0)) s): $*" >> "$LOG"
}
EXTRA=("$@")
for s in 0 1 2; do
  run --arm R-F --regime clean --seed $s
  run --arm G --regime clean --seed $s
  run --arm R-CV --regime clean --seed $s
  run --arm R-F-P --regime clean --seed $s
  run --arm R-F --regime noisy --seed $s
  run --arm R-F --regime clean --rnn lstm --seed $s
  run --arm R-F --regime clean --use-mode --seed $s
done
run --arm R-F --regime clean --n-train 2500 --seed 0
run --arm R-F --regime clean --n-train 10000 --seed 0

#!/bin/bash
# run_single_perturber.sh
# Usage: bash run_single_perturber.sh <run_number> [--eta E] [--R0 R] [--vfrac V] [--N N]
# Derived parameters (eps, dtime, tstop, dtout) are read from params_recommended.json.

if [ -z "$1" ]; then
  echo "Usage: bash run_single_perturber.sh <run_number> [options]"
  exit 1
fi

RUN_NUM_RAW=$1
shift

ETA=0.05
R0=1.305
VFRAC=1.0
N_BG=10000
DTOUT=0.25

while [[ $# -gt 0 ]]; do
  case $1 in
  --eta)
    ETA="$2"
    shift 2
    ;;
  --R0)
    R0="$2"
    shift 2
    ;;
  --vfrac)
    VFRAC="$2"
    shift 2
    ;;
  --N)
    N_BG="$2"
    shift 2
    ;;
  --dtout)
    DTOUT="$2"
    shift 2
    ;;
  *)
    echo "unknown argument '$1'"
    exit 1
    ;;
  esac
done

BASE_DIR=$(pwd)
RUN_NUM=$(printf "%03d" "$RUN_NUM_RAW")
RUN_DIR="$BASE_DIR/run_perturber_${RUN_NUM}"
LOG_FILE="$RUN_DIR/run_perturber_${RUN_NUM}.log"
PARAMS_FILE="$RUN_DIR/params_recommended.json"

# pre-run checks
for f in treecode sampling_plummer_perturber.py summary_perturber.py; do
  [ ! -f "$BASE_DIR/$f" ] && [ ! -x "$BASE_DIR/$f" ] && echo "Error: $f not found in $BASE_DIR" && exit 1
done

mkdir -p "$RUN_DIR"
echo "run $RUN_NUM_RAW  started $(date)  eta=$ETA  R0=$R0  N=$N_BG  seed=$SEED" | tee "$LOG_FILE"

# ---- Step 1: generate ICs ----
echo "step 1: sampling_plummer_perturber.py" | tee -a "$LOG_FILE"
cd "$RUN_DIR" || exit 1

python3 "$BASE_DIR/sampling_plummer_perturber.py" \
  --eta "$ETA" --R0 "$R0" --vfrac "$VFRAC" --N "$N_BG" --dtout "$DTOUT" \
  --out plummer_perturber.txt --params params_recommended.json \
  >>"$LOG_FILE" 2>&1

[ $? -ne 0 ] && echo "[ERROR] IC generation failed" | tee -a "$LOG_FILE" && exit 1
[ ! -f "$PARAMS_FILE" ] && echo "[ERROR] params_recommended.json not written" | tee -a "$LOG_FILE" && exit 1

# read derived parameters
read_param() { python3 -c "import json; d=json.load(open('$PARAMS_FILE')); print(d['$1'])"; }

EPS=$(read_param eps)
DTIME=$(read_param dtime_str) # still a string like "1/2048"
TSTOP=$(read_param tstop)
DTOUT=$(read_param dtout) # now a float, e.g. 0.5
THETA=$(read_param theta)
N_SNAPS=$(read_param n_snapshots)
N_TOTAL=$((N_BG + 1))

echo "eps=$EPS  dtime=$DTIME  tstop=$TSTOP  dtout=$DTOUT  theta=$THETA  snaps=$N_SNAPS" | tee -a "$LOG_FILE"

# ---- Progress bar ----
LINES_PER_SNAP=$((3 + 4 * N_TOTAL))
TOTAL_LINES=$((N_SNAPS * LINES_PER_SNAP))

progress_bar() {
  local current=$1
  local total=$2
  local width=40
  local pct=$((current * 100 / (total > 0 ? total : 1)))
  local filled=$((current * width / (total > 0 ? total : 1)))
  local empty=$((width - filled))
  local bar=""
  for ((i = 0; i < filled; i++)); do bar="${bar}#"; done
  for ((i = 0; i < empty; i++)); do bar="${bar}-"; done
  printf "\r  run %s  [%s] %3d%%  snapshot %d / %d" \
    "$RUN_NUM" "$bar" "$pct" "$current" "$total"
}

# ---- Step 2: treecode ----
echo "step 2: treecode" | tee -a "$LOG_FILE"

"$BASE_DIR/treecode" \
  in=plummer_perturber.txt \
  out=plummer_perturber.out \
  dtime=$DTIME eps=$EPS theta=$THETA usequad=false \
  tstop=$TSTOP dtout=$DTOUT options=out-phi \
  >>"$LOG_FILE" 2>&1 &
TC_PID=$!

# Loop to update progress bar while treecode is running
while kill -0 $TC_PID 2>/dev/null; do
  CURRENT_LINES=$(wc -l <plummer_perturber.out 2>/dev/null || echo 0)
  CURRENT_SNAPS=$((CURRENT_LINES / LINES_PER_SNAP))
  if [ "$CURRENT_SNAPS" -gt "$N_SNAPS" ]; then
    CURRENT_SNAPS=$N_SNAPS
  fi
  progress_bar "$CURRENT_SNAPS" "$N_SNAPS"
  sleep 0.5
done

# Wait for the process to actually finish and grab its exit code
wait $TC_PID
TC_STATUS=$?

# Final progress bar update to ensure it reaches 100%
CURRENT_LINES=$(wc -l <plummer_perturber.out 2>/dev/null || echo 0)
CURRENT_SNAPS=$((CURRENT_LINES / LINES_PER_SNAP))
if [ "$CURRENT_SNAPS" -gt "$N_SNAPS" ]; then
  CURRENT_SNAPS=$N_SNAPS
fi
progress_bar "$CURRENT_SNAPS" "$N_SNAPS"
echo "" # Move to a new line after the progress bar finishes

[ $TC_STATUS -ne 0 ] && echo "[ERROR] treecode exit $TC_STATUS" | tee -a "$LOG_FILE" && exit 1
[ ! -f plummer_perturber.out ] && echo "[ERROR] plummer_perturber.out not found" | tee -a "$LOG_FILE" && exit 1

ACTUAL_LINES=$(wc -l <plummer_perturber.out)
echo "treecode done: $ACTUAL_LINES lines" | tee -a "$LOG_FILE"

# ---- Step 3: analysis ----
echo "step 3: summary_perturber.py" | tee -a "$LOG_FILE"

python3 "$BASE_DIR/summary_perturber.py" \
  "$RUN_DIR" --eta "$ETA" --R0 "$R0" --N "$N_BG" --eps "$EPS" \
  >>"$LOG_FILE" 2>&1

[ $? -ne 0 ] && echo "[WARN] summary_perturber.py non-zero exit" | tee -a "$LOG_FILE"

echo "run $RUN_NUM_RAW  completed $(date)  dir=$RUN_DIR" | tee -a "$LOG_FILE"
cd "$BASE_DIR" || exit 1

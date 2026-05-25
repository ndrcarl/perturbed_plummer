#!/bin/bash
# ============================================================
#  run_single_perturber.sh — runs ONE realisation of the
#                            Plummer sphere + perturber study.
#
#  Each run writes to its own run_perturber_NNN/ directory so
#  multiple runs can be launched in parallel without conflicts.
#
#  Pipeline per run:
#    Step 1  sampling_plummer_perturber.py  -> plummer_perturber.txt
#    Step 2  treecode                       -> plummer_perturber.out
#    Step 3  summary_perturber.py           -> PDFs + perturber_stats.npz
#
#  The run number is used as the random seed for the background
#  IC sampling, so every run is fully reproducible and independent.
#
#  Usage:
#    bash run_single_perturber.sh <run_number> [options]
#
#  Options (all have defaults from the analytical study):
#    --eta    FLOAT   perturber mass ratio M/M_tot  (default: 0.05)
#    --R0     FLOAT   initial orbital radius        (default: 1.305 = r_hm)
#    --vfrac  FLOAT   v_tang / v_circ(R0)           (default: 1.0 = circular)
#    --N      INT     background particle count     (default: 10000)
#    --eps    FLOAT   softening length              (default: 0.012)
#    --dtime  STR     treecode timestep fraction    (default: 1/2048)
#    --tstop  FLOAT   total integration time        (default: 100.0)
#    --dtout  STR     output interval fraction      (default: 1/10)
#    --theta  FLOAT   tree opening angle            (default: 0.50)
#    --seed   INT     random seed (default: run_number)
#
#  Example — run 5 realisations in parallel (eta=0.05, circular orbit):
#    for i in $(seq 1 5); do
#        bash run_single_perturber.sh $i --eta 0.05 --R0 1.305 &
#    done
#    wait && bash finalize_perturber.sh
# ============================================================

# ============================================================
#  ARGUMENT PARSING
# ============================================================

if [ -z "$1" ]; then
    echo "Usage: bash run_single_perturber.sh <run_number> [options]"
    echo "       bash setup_perturber.sh  for full parameter guide"
    exit 1
fi

RUN_NUM_RAW=$1
shift   # consume run number; remaining args are options

# ---- defaults (from analytical study, Tasks 4A-4E) ----
ETA=0.05          # perturber mass ratio
R0=1.305          # initial orbital radius (= r_hm by default)
VFRAC=1.0         # tangential speed fraction (1.0 = circular)
N_BG=10000        # background particle count
EPS=0.012         # softening length
DTIME="1/2048"    # treecode timestep fraction
TSTOP=100.0       # total integration time
DTOUT="1/10"      # output interval fraction
THETA=0.50        # tree opening angle
SEED=$RUN_NUM_RAW # use run number as seed (reproducible, unique per run)

# ---- parse optional arguments ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --eta)    ETA="$2";    shift 2 ;;
        --R0)     R0="$2";     shift 2 ;;
        --vfrac)  VFRAC="$2";  shift 2 ;;
        --N)      N_BG="$2";   shift 2 ;;
        --eps)    EPS="$2";    shift 2 ;;
        --dtime)  DTIME="$2";  shift 2 ;;
        --tstop)  TSTOP="$2";  shift 2 ;;
        --dtout)  DTOUT="$2";  shift 2 ;;
        --theta)  THETA="$2";  shift 2 ;;
        --seed)   SEED="$2";   shift 2 ;;
        *)
            echo "Error: unknown argument '$1'"
            echo "Run 'bash setup_perturber.sh' for usage."
            exit 1
            ;;
    esac
done

# ============================================================
#  DIRECTORIES AND PATHS
# ============================================================

BASE_DIR=$(pwd)
RUN_NUM=$(printf "%03d" "$RUN_NUM_RAW")
RUN_DIR="$BASE_DIR/run_perturber_${RUN_NUM}"
LOG_FILE="$RUN_DIR/run_perturber_${RUN_NUM}.log"

# ============================================================
#  DERIVED QUANTITIES FOR PROGRESS TRACKING
# ============================================================

# Total particles in the simulation (background + 1 perturber)
N_TOTAL=$((N_BG + 1))

# Lines per snapshot in treecode output (with out-phi):
#   3 header lines  +  N_TOTAL mass lines
#                   +  N_TOTAL position lines
#                   +  N_TOTAL velocity lines
#                   +  N_TOTAL potential lines
LINES_PER_SNAP=$((3 + 4 * N_TOTAL))

# Total number of snapshots expected
N_SNAPS=$(python3 -c "
import math
dtout_val = eval('$DTOUT')
print(round($TSTOP / dtout_val))
")

TOTAL_LINES=$((N_SNAPS * LINES_PER_SNAP))

# ============================================================
#  PROGRESS BAR FUNCTION
# ============================================================

progress_bar() {
    local current=$1
    local total=$2
    local width=40
    local pct=$(( current * 100 / total ))
    local filled=$(( current * width / total ))
    local empty=$(( width - filled ))
    local bar=""
    for ((i=0; i<filled; i++)); do bar="${bar}#"; done
    for ((i=0; i<empty;  i++)); do bar="${bar}-"; done
    printf "\r  run %s  [%s] %3d%%  snapshot %d / %d" \
           "$RUN_NUM" "$bar" "$pct" "$current" "$total"
}

# ============================================================
#  PRE-RUN CHECKS
# ============================================================

if [ ! -x "$BASE_DIR/treecode" ]; then
    echo "Error: treecode not found or not executable in $BASE_DIR"
    echo "       run: make && chmod +x treecode"
    exit 1
fi

if [ ! -f "$BASE_DIR/sampling_plummer_perturber.py" ]; then
    echo "Error: sampling_plummer_perturber.py not found in $BASE_DIR"
    exit 1
fi

if [ ! -f "$BASE_DIR/summary_perturber.py" ]; then
    echo "Error: summary_perturber.py not found in $BASE_DIR"
    exit 1
fi

# ============================================================
#  CREATE RUN DIRECTORY AND START LOG
# ============================================================

mkdir -p "$RUN_DIR"

# Write run header to log (tee to terminal and log file)
{
echo "========================================================"
echo "  run $RUN_NUM_RAW  started at $(date)"
echo "========================================================"
echo ""
echo "  [PARAMETERS]"
echo "    eta              : $ETA     (perturber mass ratio M/M_tot)"
echo "    R0               : $R0     (initial orbital radius)"
echo "    vfrac            : $VFRAC   (v_tang / v_circ — 1.0=circular)"
echo "    N_background     : $N_BG"
echo "    N_total          : $N_TOTAL  (background + perturber)"
echo "    eps              : $EPS     (softening length)"
echo "    dtime            : $DTIME   (treecode timestep)"
echo "    tstop            : $TSTOP   (total integration time)"
echo "    dtout            : $DTOUT   (output interval)"
echo "    theta            : $THETA   (tree opening angle)"
echo "    seed             : $SEED    (random seed)"
echo ""
echo "  [EXPECTED OUTPUT]"
echo "    snapshots        : $N_SNAPS"
echo "    lines/snapshot   : $LINES_PER_SNAP"
echo "    total lines      : $TOTAL_LINES"
echo "    run directory    : $RUN_DIR"
echo "========================================================"
} | tee "$LOG_FILE"

# ============================================================
#  STEP 1 — GENERATE INITIAL CONDITIONS
# ============================================================

echo "" | tee -a "$LOG_FILE"
echo "  Step 1: sampling_plummer_perturber.py" | tee -a "$LOG_FILE"

cd "$RUN_DIR" || exit 1

python3 "$BASE_DIR/sampling_plummer_perturber.py" \
    --eta   "$ETA"   \
    --R0    "$R0"    \
    --vfrac "$VFRAC" \
    --N     "$N_BG"  \
    --seed  "$SEED"  \
    --out   "plummer_perturber.txt" \
    >> "$LOG_FILE" 2>&1

IC_STATUS=$?
if [ $IC_STATUS -ne 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "  [ERROR] IC generation failed (exit code $IC_STATUS)." | tee -a "$LOG_FILE"
    echo "          Check $LOG_FILE for details." | tee -a "$LOG_FILE"
    exit 1
fi
echo "  IC generation complete." | tee -a "$LOG_FILE"

# ============================================================
#  STEP 2 — RUN TREECODE
# ============================================================

echo "" | tee -a "$LOG_FILE"
echo "  Step 2: treecode" | tee -a "$LOG_FILE"
echo "    in=plummer_perturber.txt  out=plummer_perturber.out" | tee -a "$LOG_FILE"
echo "    eps=$EPS  dtime=$DTIME  theta=$THETA  tstop=$TSTOP  dtout=$DTOUT" \
    | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

"$BASE_DIR/treecode"             \
    in=plummer_perturber.txt     \
    out=plummer_perturber.out    \
    dtime=$DTIME                 \
    eps=$EPS                     \
    theta=$THETA                 \
    usequad=false                \
    tstop=$TSTOP                 \
    dtout=$DTOUT                 \
    options=out-phi              \
    >> "$LOG_FILE" 2>&1 &

TREECODE_PID=$!

# ---- progress bar (polls output file line count) ----
echo ""
while kill -0 $TREECODE_PID 2>/dev/null; do
    if [ -f "plummer_perturber.out" ]; then
        current_lines=$(wc -l < "plummer_perturber.out")
        current_snaps=$(( current_lines / LINES_PER_SNAP ))
        [ $current_snaps -gt $N_SNAPS ] && current_snaps=$N_SNAPS
        progress_bar "$current_snaps" "$N_SNAPS"
    else
        printf "\r  run %s  waiting for first snapshot..." "$RUN_NUM"
    fi
    sleep 2
done

progress_bar "$N_SNAPS" "$N_SNAPS"
echo ""

# ---- check treecode exit status ----
wait $TREECODE_PID
TC_STATUS=$?
if [ $TC_STATUS -ne 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "  [ERROR] treecode exited with status $TC_STATUS." | tee -a "$LOG_FILE"
    echo "          Check $LOG_FILE for details." | tee -a "$LOG_FILE"
    exit 1
fi

echo "  treecode complete." | tee -a "$LOG_FILE"

# ---- verify output file was written ----
if [ ! -f "plummer_perturber.out" ]; then
    echo "  [ERROR] plummer_perturber.out not found after treecode run." \
        | tee -a "$LOG_FILE"
    exit 1
fi

ACTUAL_LINES=$(wc -l < "plummer_perturber.out")
ACTUAL_SNAPS=$(( ACTUAL_LINES / LINES_PER_SNAP ))
echo "  Output: $ACTUAL_LINES lines -> $ACTUAL_SNAPS snapshots written." \
    | tee -a "$LOG_FILE"

if [ "$ACTUAL_SNAPS" -lt $(( N_SNAPS / 2 )) ]; then
    echo "  [WARN] Fewer than half the expected snapshots were written." \
        | tee -a "$LOG_FILE"
    echo "         Expected $N_SNAPS, got $ACTUAL_SNAPS. Run may have crashed early." \
        | tee -a "$LOG_FILE"
fi

# ============================================================
#  STEP 3 — ANALYSIS
# ============================================================

echo "" | tee -a "$LOG_FILE"
echo "  Step 3: summary_perturber.py" | tee -a "$LOG_FILE"

python3 "$BASE_DIR/summary_perturber.py" \
    "$RUN_DIR"                           \
    --eta   "$ETA"                       \
    --R0    "$R0"                        \
    --N     "$N_BG"                      \
    --eps   "$EPS"                       \
    >> "$LOG_FILE" 2>&1

SUM_STATUS=$?
if [ $SUM_STATUS -ne 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "  [WARN] summary_perturber.py exited with status $SUM_STATUS." \
        | tee -a "$LOG_FILE"
    echo "         PDFs or npz may be incomplete." | tee -a "$LOG_FILE"
fi

# ============================================================
#  FINISH
# ============================================================

echo "" | tee -a "$LOG_FILE"
echo "========================================================"   | tee -a "$LOG_FILE"
echo "  run $RUN_NUM_RAW  completed at $(date)"                   | tee -a "$LOG_FILE"
echo "  output directory : $RUN_DIR"                              | tee -a "$LOG_FILE"
echo "  log file         : $LOG_FILE"                             | tee -a "$LOG_FILE"
echo "========================================================"   | tee -a "$LOG_FILE"

cd "$BASE_DIR" || exit 1

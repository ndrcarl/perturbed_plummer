#!/bin/bash
# ============================================================
#  finalize_perturber.sh — post-processing for the full
#                          Plummer + perturber ensemble.
#
#  Intended to be run AFTER all run_single_perturber.sh jobs
#  have completed (either sequentially or in parallel).
#
#  What this script does:
#    1. Verifies all expected run directories are complete
#       (treecode output present, npz present).
#    2. Concatenates all per-run logs into master_perturber.log.
#    3. Calls summary_perturber.py --combined to produce the
#       ensemble PDFs and print scalar statistics.
#    4. Prints a final quality-control report to stdout and
#       appends it to master_perturber.log.
#
#  Usage:
#    bash finalize_perturber.sh [options]
#
#  Options:
#    --eta   FLOAT   perturber mass ratio (passed to summary_perturber.py)
#                    (default: 0.05)
#    --R0    FLOAT   initial orbital radius (default: 1.305)
#    --runs  INT     expected number of runs to check (default: auto-detect)
#
#  Example:
#    # after launching 20 realisations:
#    bash finalize_perturber.sh --eta 0.05 --R0 1.305 --runs 20
# ============================================================

# ============================================================
#  ARGUMENT PARSING
# ============================================================

ETA=0.05
R0=1.305
EXPECTED_RUNS=0 # 0 = auto-detect from existing run_perturber_* dirs

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
  --runs)
    EXPECTED_RUNS="$2"
    shift 2
    ;;
  *)
    echo "Error: unknown argument '$1'"
    echo "Usage: bash finalize_perturber.sh [--eta E] [--R0 R] [--runs N]"
    exit 1
    ;;
  esac
done

# ============================================================
#  SETUP
# ============================================================

BASE_DIR=$(pwd)
MASTER_LOG="$BASE_DIR/master_perturber.log"
TIMESTAMP=$(date)

echo "========================================================"
echo "  PLUMMER + PERTURBER — FINALIZE"
echo "  $TIMESTAMP"
echo "========================================================"
echo "  eta  = $ETA"
echo "  R0   = $R0"
echo ""

# ============================================================
#  STEP 1 — DISCOVER AND VERIFY RUN DIRECTORIES
# ============================================================

echo "Step 1: Verifying run directories..."
echo ""

# collect all run_perturber_NNN directories, sorted
RUN_DIRS=()
while IFS= read -r -d '' d; do
  RUN_DIRS+=("$d")
done < <(find "$BASE_DIR" -maxdepth 1 -type d -name 'run_perturber_[0-9]*' -print0 | sort -z)

N_FOUND=${#RUN_DIRS[@]}

if [ "$N_FOUND" -eq 0 ]; then
  echo "  [ERROR] No run_perturber_* directories found in $BASE_DIR"
  echo "          Have any runs completed?"
  exit 1
fi

echo "  Found $N_FOUND run directories."

if [ "$EXPECTED_RUNS" -gt 0 ] && [ "$N_FOUND" -ne "$EXPECTED_RUNS" ]; then
  echo "  [WARN] Expected $EXPECTED_RUNS runs but found $N_FOUND."
  echo "         Continuing with the $N_FOUND runs that exist."
fi

# check each run for completeness
N_COMPLETE=0
N_MISSING_OUT=0
N_MISSING_NPZ=0
INCOMPLETE_RUNS=()

for rdir in "${RUN_DIRS[@]}"; do
  rname=$(basename "$rdir")
  has_out=false
  has_npz=false

  # check treecode output
  if [ -f "$rdir/plummer_perturber.out" ]; then
    out_lines=$(wc -l <"$rdir/plummer_perturber.out")
    if [ "$out_lines" -gt 0 ]; then
      has_out=true
    fi
  fi

  # check npz
  if [ -f "$rdir/perturber_stats.npz" ]; then
    has_npz=true
  fi

  if $has_out && $has_npz; then
    N_COMPLETE=$((N_COMPLETE + 1))
    echo "  [OK]   $rname  (out: ${out_lines} lines, npz: present)"
  else
    INCOMPLETE_RUNS+=("$rname")
    if ! $has_out; then
      N_MISSING_OUT=$((N_MISSING_OUT + 1))
      echo "  [FAIL] $rname  — plummer_perturber.out missing or empty"
    fi
    if ! $has_npz; then
      N_MISSING_NPZ=$((N_MISSING_NPZ + 1))
      echo "  [FAIL] $rname  — perturber_stats.npz missing"
      echo "         Re-run analysis: python3 summary_perturber.py $rdir --eta $ETA --R0 $R0"
    fi
  fi
done

echo ""
echo "  Complete runs : $N_COMPLETE / $N_FOUND"
[ $N_MISSING_OUT -gt 0 ] && echo "  Missing output: $N_MISSING_OUT runs"
[ $N_MISSING_NPZ -gt 0 ] && echo "  Missing npz   : $N_MISSING_NPZ runs"

if [ $N_COMPLETE -eq 0 ]; then
  echo ""
  echo "  [ERROR] No complete runs found. Cannot produce ensemble results."
  exit 1
fi

if [ $N_COMPLETE -lt 5 ]; then
  echo ""
  echo "  [WARN] Only $N_COMPLETE runs are complete."
  echo "         Ensemble statistics with < 5 runs are unreliable."
  echo "         Consider running more realisations before finalising."
fi

# ============================================================
#  STEP 2 — CONCATENATE PER-RUN LOGS
# ============================================================

echo ""
echo "Step 2: Concatenating per-run logs -> master_perturber.log"

>"$MASTER_LOG"

{
  echo "========================================================"
  echo "  MASTER LOG — Plummer + Perturber Ensemble"
  echo "  Finalised at: $TIMESTAMP"
  echo "  eta=$ETA  R0=$R0"
  echo "  Runs found  : $N_FOUND"
  echo "  Runs complete: $N_COMPLETE"
  echo "========================================================"
} >>"$MASTER_LOG"

for rdir in "${RUN_DIRS[@]}"; do
  rname=$(basename "$rdir")
  logfile="$rdir/${rname}.log"
  if [ -f "$logfile" ]; then
    echo "" >>"$MASTER_LOG"
    echo "-------- $rname --------" >>"$MASTER_LOG"
    cat "$logfile" >>"$MASTER_LOG"
  else
    echo "" >>"$MASTER_LOG"
    echo "-------- $rname  [no log file found] --------" >>"$MASTER_LOG"
  fi
done

echo "  Written: $MASTER_LOG"

# ============================================================
#  STEP 3 — COMBINED ANALYSIS
# ============================================================

echo ""
echo "Step 3: Running combined ensemble analysis..."
echo ""

{
  echo ""
  echo "========================================================"
  echo "  COMBINED ANALYSIS at $TIMESTAMP"
  echo "========================================================"
} >>"$MASTER_LOG"

python3 "$BASE_DIR/summary_perturber.py" \
  --combined \
  --eta "$ETA" \
  --R0 "$R0" \
  2>&1 | tee -a "$MASTER_LOG"

COMBINED_STATUS=$?

if [ $COMBINED_STATUS -ne 0 ]; then
  echo ""
  echo "  [WARN] summary_perturber.py --combined exited with status $COMBINED_STATUS."
  echo "         Combined PDFs may be incomplete."
fi

# ============================================================
#  STEP 4 — QUALITY CONTROL REPORT
# ============================================================

echo ""
echo "========================================================"
echo "  QUALITY CONTROL REPORT"
echo "========================================================"

# extract key scalar summaries from master log for quick display
echo ""
echo "  Per-run energy conservation check:"
N_ENERGY_OK=0
N_ENERGY_WARN=0
for rdir in "${RUN_DIRS[@]}"; do
  rname=$(basename "$rdir")
  npz="$rdir/perturber_stats.npz"
  if [ -f "$npz" ]; then
    # extract max_dE using python one-liner
    max_dE=$(python3 -c "
import numpy as np, sys
try:
    d = np.load('$npz', allow_pickle=False)
    print(f\"{float(d['max_dE']):.4e}\")
except Exception as e:
    print('err')
" 2>/dev/null)
    # compare with threshold 0.01 (1%)
    ok=$(python3 -c "
try:
    v = float('$max_dE')
    print('OK' if v < 0.01 else 'WARN')
except:
    print('WARN')
" 2>/dev/null)
    if [ "$ok" = "OK" ]; then
      N_ENERGY_OK=$((N_ENERGY_OK + 1))
      echo "    [OK]   $rname  max|dE/E0| = $max_dE"
    else
      N_ENERGY_WARN=$((N_ENERGY_WARN + 1))
      echo "    [WARN] $rname  max|dE/E0| = $max_dE  (> 1% threshold)"
    fi
  fi
done
echo ""
echo "  Energy conservation: $N_ENERGY_OK / $N_COMPLETE runs within 1% threshold."

echo ""
echo "  Per-run virial ratio check  (target: 2K_bg/|W_bg| = 1.000 ± 0.05):"
N_VIR_OK=0
N_VIR_WARN=0
for rdir in "${RUN_DIRS[@]}"; do
  rname=$(basename "$rdir")
  npz="$rdir/perturber_stats.npz"
  if [ -f "$npz" ]; then
    mean_vir=$(python3 -c "
import numpy as np
try:
    d = np.load('$npz', allow_pickle=False)
    print(f\"{float(d['mean_virial']):.4f}\")
except:
    print('err')
" 2>/dev/null)
    ok=$(python3 -c "
try:
    v = abs(float('$mean_vir') - 1.0)
    print('OK' if v < 0.05 else 'WARN')
except:
    print('WARN')
" 2>/dev/null)
    if [ "$ok" = "OK" ]; then
      N_VIR_OK=$((N_VIR_OK + 1))
      echo "    [OK]   $rname  virial = $mean_vir"
    else
      N_VIR_WARN=$((N_VIR_WARN + 1))
      echo "    [WARN] $rname  virial = $mean_vir  (>5% from 1)"
    fi
  fi
done
echo ""
echo "  Virial ratio: $N_VIR_OK / $N_COMPLETE runs within tolerance."

echo ""
echo "  Background heating check  (target: DK_bg / K_bg(0) < 20%):"
N_HEAT_OK=0
N_HEAT_WARN=0
for rdir in "${RUN_DIRS[@]}"; do
  rname=$(basename "$rdir")
  npz="$rdir/perturber_stats.npz"
  if [ -f "$npz" ]; then
    dk_frac=$(python3 -c "
import numpy as np
try:
    d = np.load('$npz', allow_pickle=False)
    print(f\"{float(d['DK_bg_frac']):.4f}\")
except:
    print('err')
" 2>/dev/null)
    ok=$(python3 -c "
try:
    v = abs(float('$dk_frac'))
    print('OK' if v < 0.20 else 'WARN')
except:
    print('WARN')
" 2>/dev/null)
    if [ "$ok" = "OK" ]; then
      N_HEAT_OK=$((N_HEAT_OK + 1))
      echo "    [OK]   $rname  DK_bg/K0 = $dk_frac"
    else
      N_HEAT_WARN=$((N_HEAT_WARN + 1))
      echo "    [WARN] $rname  DK_bg/K0 = $dk_frac  (> 20% — background significantly heated)"
    fi
  fi
done
echo ""
echo "  Background heating: $N_HEAT_OK / $N_COMPLETE runs within 20% threshold."

# ============================================================
#  STEP 5 — LIST COMBINED OUTPUT FILES
# ============================================================

echo ""
echo "========================================================"
echo "  COMBINED OUTPUT FILES"
echo "========================================================"
echo ""
for pdf in combined_orbital_decay.pdf \
  combined_scalar_distributions.pdf \
  combined_ensemble_convergence.pdf; do
  if [ -f "$BASE_DIR/$pdf" ]; then
    echo "  [OK]   $pdf"
  else
    echo "  [miss] $pdf  (not generated)"
  fi
done
echo ""
echo "  master_perturber.log"
echo ""

# ============================================================
#  CONVERGENCE CHECKLIST REMINDER
# ============================================================

echo "========================================================"
echo "  CONVERGENCE CHECKLIST  (from Task 4E)"
echo "========================================================"
echo ""
echo "  Have you completed all convergence levels?"
echo ""
echo "  Level 0 — background stability (no perturber):"
echo "    Run isolated Plummer with eps in {0.006, 0.012, 0.024}."
echo "    All Lagrangian radii flat, |dE/E0| < 1%."
echo ""
echo "  Level 1 — timestep convergence:"
echo "    Run 3 single-seed perturber runs with"
echo "    dtime in {1/1024, 1/2048, 1/4096}."
echo "    Check: |<dR/dt>(dt) - <dR/dt>(dt/2)| / <dR/dt>(dt/2) < 5%."
echo ""
echo "  Level 2 — softening convergence:"
echo "    Run 3 single-seed perturber runs with"
echo "    eps in {0.006, 0.012, 0.024}."
echo "    Check: |<dR/dt>(eps) - <dR/dt>(eps/2)| / <dR/dt>(eps/2) < 10%."
echo ""
echo "  Level 3 — particle number convergence:"
echo "    Run 5 realisations each at N in {2500, 5000, 10000}."
echo "    Check: <dR/dt> scales as ln(eta*N); sigma_dR ~ N^(-1/2)."
echo ""
echo "  Level 4 — ensemble size convergence (THIS RUN):"
echo "    Running mean of R_M(t_final) must stabilise."
echo "    Standard error < 5% of mean."
echo "    See: combined_ensemble_convergence.pdf"
echo ""
echo "  Level 5 — physics validation:"
echo "    Compare <R_M(t)> with Chandrasekhar ODE."
echo "    Extract ln_Lambda_eff from combined_orbital_decay.pdf."
echo "    Check: ln_Lambda_eff approx ln(1/eta) = $(python3 -c "import math; print(f'{math.log(1.0/float(\"$ETA\")):.3f}')" 2>/dev/null)"
echo ""
echo "  See combined_scalar_distributions.pdf for ln_Lambda_eff distribution."
echo "========================================================"

# ============================================================
#  FINAL STATUS LINE
# ============================================================

{
  echo ""
  echo "========================================================"
  echo "  Finalise complete at $(date)"
  echo "  Complete runs: $N_COMPLETE / $N_FOUND"
  echo "  Energy OK    : $N_ENERGY_OK / $N_COMPLETE"
  echo "  Virial OK    : $N_VIR_OK / $N_COMPLETE"
  echo "  Heating OK   : $N_HEAT_OK / $N_COMPLETE"
  echo "========================================================"
} | tee -a "$MASTER_LOG"

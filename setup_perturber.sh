#!/bin/bash
# ============================================================
#  setup_perturber.sh — environment check for the Plummer +
#                       perturber simulation pipeline.
#
#  Verifies:
#    - treecode binary is present and executable
#    - all required Python scripts are present
#    - Python 3 and all required packages are importable
#    - original Plummer scripts are present (reference baseline)
#
#  Also prints the analytically derived parameter recommendations
#  from the study (Tasks 4A-4E) as a pre-run reference guide.
#
#  Usage:
#    bash setup_perturber.sh
# ============================================================

BASE_DIR=$(pwd)
ALL_OK=true

echo "========================================================"
echo "  PLUMMER + PERTURBER — ENVIRONMENT CHECK"
echo "========================================================"

# ---- treecode ----
echo ""
echo "Checking treecode..."
if [ -x "$BASE_DIR/treecode" ]; then
    echo "  [OK] treecode found and executable"
else
    echo "  [FAIL] treecode not found or not executable"
    echo "         run: make && chmod +x treecode"
    ALL_OK=false
fi

# ---- required perturber scripts ----
echo ""
echo "Checking perturber pipeline scripts..."
SCRIPTS_PERTURBER=(
    sampling_plummer_perturber.py
    summary_perturber.py
    run_single_perturber.sh
    finalize_perturber.sh
)

for s in "${SCRIPTS_PERTURBER[@]}"; do
    if [ -f "$BASE_DIR/$s" ]; then
        echo "  [OK] $s"
    else
        echo "  [FAIL] $s not found"
        ALL_OK=false
    fi
done

# ---- original baseline scripts (reference, not required to run) ----
echo ""
echo "Checking original Plummer baseline scripts (reference)..."
SCRIPTS_ORIG=(
    sampling_plummer.py
    summary_runs.py
    run_single.sh
    finalize.sh
)

for s in "${SCRIPTS_ORIG[@]}"; do
    if [ -f "$BASE_DIR/$s" ]; then
        echo "  [OK] $s  (baseline reference)"
    else
        echo "  [warn] $s not found  (baseline reference only — not required)"
    fi
done

# ---- python 3 ----
echo ""
echo "Checking Python 3..."
if command -v python3 &>/dev/null; then
    PY_VER=$(python3 --version 2>&1)
    echo "  [OK] $PY_VER"
else
    echo "  [FAIL] python3 not found"
    ALL_OK=false
fi

# ---- required python packages ----
echo ""
echo "Checking Python packages..."

check_pkg() {
    local pkg=$1
    local import_name=${2:-$1}
    python3 -c "import $import_name" 2>/dev/null
    if [ $? -eq 0 ]; then
        local ver
        ver=$(python3 -c "import $import_name; print(getattr($import_name,'__version__','unknown'))" 2>/dev/null)
        echo "  [OK] $pkg  ($ver)"
    else
        echo "  [FAIL] $pkg  (pip install $pkg)"
        ALL_OK=false
    fi
}

check_pkg numpy
check_pkg matplotlib
check_pkg scipy
check_pkg argparse   # stdlib, should always be present

# ---- shell run script permissions ----
echo ""
echo "Checking shell script permissions..."
for sh in run_single_perturber.sh finalize_perturber.sh; do
    if [ -f "$BASE_DIR/$sh" ]; then
        if [ -x "$BASE_DIR/$sh" ]; then
            echo "  [OK] $sh is executable"
        else
            echo "  [warn] $sh exists but is not executable — fixing..."
            chmod +x "$BASE_DIR/$sh"
            echo "         chmod +x $sh applied"
        fi
    fi
done

# ---- log file ----
echo ""
echo "Initialising master_perturber.log..."
> "$BASE_DIR/master_perturber.log"
echo "  [OK] master_perturber.log cleared"

# ============================================================
#  PARAMETER GUIDE  (from analytical derivations, Tasks 4A–4E)
# ============================================================

echo ""
echo "========================================================"
echo "  PARAMETER RECOMMENDATIONS  (code units: G=M_tot=b=1)"
echo "========================================================"
echo ""
echo "  [PHYSICAL SCALES OF THE BACKGROUND PLUMMER SPHERE]"
echo "    r_hm  = b/sqrt(2^(2/3)-1)  ~ 1.305     half-mass radius"
echo "    t_dyn = pi/2               ~ 1.571     dynamical time"
echo "    T_orb (at r_hm)            ~ 13.24     orbital period"
echo "    t_relax (N=10^4, Spitzer)  ~ 220-438   relaxation time"
echo "    d_mean (near r_hm)         ~ 0.123     mean particle separation"
echo ""
echo "  [PERTURBER MASS RATIO eta = M/M_tot]"
echo "    Recommended range  :  0.01 <= eta <= 0.10"
echo "    Fiducial (Set A)   :  eta = 0.05   (moderate, measurable DF)"
echo "    Light (Set B)      :  eta = 0.01   (cleanest theory comparison)"
echo "    Heavy (exploratory):  eta = 0.10   (fast inspiral, strong back-reaction)"
echo "    Hard lower limit   :  eta > 0.022  (r_inf > eps_background required)"
echo ""
echo "  [SOFTENING LENGTH eps]"
echo "    Formula: eps = min_R [ alpha * G*M / (v_BH^2 + sigma^2(R)) ]"
echo "    alpha ~ 0.2 (recommended)"
echo "    The minimum is at R ~ r_hm for circular orbits in Plummer."
echo "    For eccentric orbits evaluate at pericenter."
echo "    Background collisionality lower bound: eps >= d_mean/10 ~ 0.012"
echo ""
echo "    eta=0.01 :  r_inf_corr ~ 0.021  =>  eps_rec ~ 0.004-0.008"
echo "    eta=0.05 :  r_inf_corr ~ 0.103  =>  eps_rec ~ 0.012-0.020 (*)"
echo "    eta=0.10 :  r_inf_corr ~ 0.207  =>  eps_rec ~ 0.020-0.040"
echo "    (*) fiducial: eps=0.012 is at the lower edge — run convergence test"
echo ""
echo "  [TIMESTEP dtime]"
echo "    Formula: dt = eta_acc * min(t_2body, t_potential)"
echo "    t_2body     = eps / v_rel         [encounter timescale]"
echo "    t_potential = T_orb / (2*pi)      [orbital timescale]"
echo "    eta_acc ~ 0.05"
echo "    Binding constraint: t_2body << t_potential (factor ~165 for eps=0.012)"
echo ""
echo "    eps=0.012, v_rel~0.94:  dt_rec ~ 6.4e-4  =>  dtime = 1/2048"
echo "    eps=0.008, v_rel~0.94:  dt_rec ~ 4.3e-4  =>  dtime = 1/4096"
echo "    eta=0.10 (accel. crit): dt_rec ~ 3.5e-4  =>  dtime = 1/4096"
echo "    Run Level-1 convergence test: dtime in {1/1024, 1/2048, 1/4096}"
echo ""
echo "  [TOTAL INTEGRATION TIME tstop]"
echo "    Lower bound (multiple orbits):  tstop >= 5 * T_orb ~ 66"
echo "    Upper bound (bg relaxation):    tstop <= t_relax/5 ~ 88"
echo "    Upper bound (bg heating 10%):   tstop depends on eta and t_DF"
echo "    Recommended:  tstop = 100  for eta=0.01-0.05"
echo "                  tstop = 50   for eta=0.10"
echo ""
echo "  [DYNAMICAL FRICTION TIMESCALE  (SIS lower bound)]"
echo "    t_DF ~ R0^2 * v_circ(R0) / (0.8 * G * M * ln(1/eta))"
echo "    eta=0.01, R0=1.5:  t_DF ~  38-115  (SIS to corrected)"
echo "    eta=0.05, R0=1.305:t_DF ~   9-30   (SIS to corrected)"
echo "    eta=0.10, R0=1.305:t_DF ~   6-18   (SIS to corrected)"
echo "    Note: Plummer core stalling makes the true t_DF longer than SIS."
echo ""
echo "  [NUMBER OF REALISATIONS]"
echo "    Per-realisation scatter: sigma_R / <R> ~ 30-50% for eta*N=500"
echo "    Required for 5% standard error on mean:"
echo "    eta=0.05, N=10^4:  N_real >= 30-50"
echo "    eta=0.01, N=10^4:  N_real >= 50-100"
echo "    Start with 10, assess scatter, add realisations as needed."
echo ""
echo "  [CONVERGENCE TEST ORDER  (Tasks 4E)]"
echo "    Level 0:  background stability  (no perturber)"
echo "    Level 1:  timestep  {1/1024, 1/2048, 1/4096}, fixed seed"
echo "    Level 2:  softening {0.006, 0.012, 0.024},    fixed seed"
echo "    Level 3:  particle N {2500, 5000, 10000},      5 seeds each"
echo "    Level 4:  production ensemble (20-50 seeds)"
echo "    Level 5:  physics validation vs Chandrasekhar"
echo ""
echo "  [CONTROL TESTS  (run before production)]"
echo "    Null test:  eta=m/M_tot=1e-4 (test particle) — no decay expected"
echo "    Static:     v0=0 (perturber at rest at R0) — background re-equilibrates"
echo "========================================================"

# ---- result ----
echo ""
if [ "$ALL_OK" = true ]; then
    echo "  All checks passed."
    echo ""
    echo "  To generate initial conditions:"
    echo "    python3 sampling_plummer_perturber.py --eta 0.05 --R0 1.305"
    echo ""
    echo "  To run a single realisation:"
    echo "    bash run_single_perturber.sh <run_number> [--eta 0.05] [--R0 1.305]"
    echo ""
    echo "  To launch N realisations in parallel (example: 20 runs):"
    echo "    for i in \$(seq 1 20); do"
    echo "      bash run_single_perturber.sh \$i --eta 0.05 --R0 1.305 &"
    echo "    done"
    echo "    wait"
    echo "    bash finalize_perturber.sh"
    echo ""
    echo "  Ready to run."
else
    echo "  Some checks FAILED. Fix the issues above before running."
fi
echo "========================================================"

#!/usr/bin/env python3
# ============================================================
#  summary_perturber.py
#
#  Two operating modes:
#
#  MODE A — per-run analysis (called by run_single_perturber.sh):
#    python3 summary_perturber.py <run_dir> --eta E --R0 R --N N --eps EPS
#
#    Reads plummer_perturber.out from <run_dir>, computes all
#    diagnostics on the fly (streaming, low RAM), writes PDFs
#    and a lightweight perturber_stats.npz to <run_dir>.
#
#  MODE B — combined analysis (called by finalize_perturber.sh):
#    python3 summary_perturber.py --combined [--eta E] [--R0 R]
#
#    Loads all perturber_stats.npz files from run_perturber_*/,
#    computes ensemble statistics, and writes combined PDFs to
#    the current directory.
#
#  PERTURBER CONVENTION:
#    Particle index 0 in every snapshot is the perturber.
#    Indices 1..N are the background.
#    This matches the output of sampling_plummer_perturber.py.
#
#  STREAMING READER:
#    The treecode output file is read one snapshot at a time.
#    Each snapshot is processed and discarded before the next
#    is read.  RAM usage is O(N) regardless of N_snaps.
#
#  UNITS: G=1, M_tot=1, b=1 throughout.
# ============================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import argparse
import math
from scipy import stats
from scipy.ndimage import uniform_filter1d

# ============================================================
#  ARGUMENT PARSING
# ============================================================

parser = argparse.ArgumentParser(
    description='Analyse Plummer + perturber N-body simulation output.')

parser.add_argument('run_dir', nargs='?', default=None,
                    help='Path to a single run directory (Mode A). '
                         'Omit when using --combined (Mode B).')
parser.add_argument('--eta',      type=float, default=0.05,
                    help='Perturber mass ratio M/M_tot (default: 0.05)')
parser.add_argument('--R0',       type=float, default=None,
                    help='Initial orbital radius (default: r_hm ~ 1.305)')
parser.add_argument('--N',        type=int,   default=10000,
                    help='Number of background particles (default: 10000)')
parser.add_argument('--eps',      type=float, default=0.012,
                    help='Softening length used in the run (default: 0.012)')
parser.add_argument('--combined', action='store_true',
                    help='Run in combined (Mode B) ensemble analysis.')

args = parser.parse_args()

# ============================================================
#  GLOBAL PHYSICAL CONSTANTS  (code units: G=1, M_tot=1, b=1)
# ============================================================

G      = 1.0
M_tot  = 1.0
b      = 1.0
r_hm   = b / math.sqrt(2.0**(2.0/3.0) - 1.0)   # ~ 1.305
rho_c  = 3.0 * M_tot / (4.0 * math.pi * b**3)   # = 3/(4*pi)
t_dyn  = math.sqrt(3.0 * math.pi / (16.0 * G * rho_c))  # = pi/2

ETA    = args.eta
M_pert = ETA * M_tot
N_BG   = args.N
m_bg   = M_tot / N_BG                           # background particle mass
R0_ARG = args.R0 if args.R0 is not None else r_hm
EPS    = args.eps

# ============================================================
#  ANALYTICAL PLUMMER FUNCTIONS
# ============================================================

def plummer_density(r):
    """Plummer density profile rho(r)."""
    return (3.0 * M_tot / (4.0 * math.pi * b**3)) * (1.0 + r**2 / b**2)**(-2.5)

def plummer_mass_enc(r):
    """Enclosed mass M(<r) for the Plummer sphere."""
    return M_tot * r**3 / (r**2 + b**2)**1.5

def plummer_sigma2(r):
    """Isotropic 1D velocity dispersion squared sigma^2(r)."""
    return G * M_tot / (6.0 * math.sqrt(r**2 + b**2))

def plummer_vcirc(r):
    """Circular speed v_circ(r) = sqrt(G*M(<r)/r)."""
    if r <= 0.0:
        return 0.0
    return math.sqrt(G * plummer_mass_enc(r) / r)

def chandrasekhar_bracket(X):
    """Chandrasekhar bracket B(X) = erf(X) - (2X/sqrt(pi))*exp(-X^2).
    X = v_M / (sqrt(2) * sigma).
    B(X) is the fraction of background particles slower than v_M,
    weighted by the Maxwellian distribution."""
    from math import erf, exp, sqrt, pi
    return erf(X) - (2.0 * X / sqrt(pi)) * exp(-X**2)

def chandrasekhar_decel(R, v_M, M_p, ln_lam):
    """Chandrasekhar dynamical friction deceleration magnitude.
    Returns a_fric = |dv_M/dt| in code units.
    Uses local Plummer density and velocity dispersion at radius R."""
    rho   = plummer_density(R)
    sig2  = plummer_sigma2(R)
    sigma = math.sqrt(sig2)
    if v_M <= 0.0 or sigma <= 0.0:
        return 0.0
    X     = v_M / (math.sqrt(2.0) * sigma)
    B     = chandrasekhar_bracket(X)
    if B <= 0.0:
        return 0.0
    return 4.0 * math.pi * G**2 * M_p * rho * ln_lam * B / v_M**2

# ============================================================
#  TREECODE OUTPUT READER  (streaming, O(N) RAM)
# ============================================================

def iter_snapshots(filepath, N_total):
    """Generator: yield one snapshot at a time from a Barnes treecode
    output file.  Each snapshot is a dict:
        t       : float, simulation time
        masses  : (N_total,) array
        pos     : (N_total, 3) array
        vel     : (N_total, 3) array
        phi     : (N_total,) array  (gravitational potential)
    N_total = N_background + 1 (perturber is index 0).
    """
    with open(filepath, 'r') as fh:
        while True:
            # --- header: N, ndim, t ---
            line_N = fh.readline()
            if not line_N:
                return                           # EOF
            line_N = line_N.strip()
            if not line_N:
                continue                         # skip blank separator lines
            try:
                n_snap = int(line_N)
            except ValueError:
                continue
            ndim = int(fh.readline())
            t    = float(fh.readline())

            if n_snap != N_total:
                # mismatch: skip this snapshot defensively
                for _ in range(4 * n_snap):
                    fh.readline()
                continue

            # --- masses ---
            masses = np.empty(N_total)
            for i in range(N_total):
                masses[i] = float(fh.readline())

            # --- positions ---
            pos = np.empty((N_total, 3))
            for i in range(N_total):
                pos[i] = [float(x) for x in fh.readline().split()]

            # --- velocities ---
            vel = np.empty((N_total, 3))
            for i in range(N_total):
                vel[i] = [float(x) for x in fh.readline().split()]

            # --- potentials ---
            phi = np.empty(N_total)
            for i in range(N_total):
                phi[i] = float(fh.readline())

            yield {'t': t, 'masses': masses, 'pos': pos,
                   'vel': vel, 'phi': phi}

# ============================================================
#  LAGRANGIAN RADII HELPER
# ============================================================

LAGR_FRACS = [0.10, 0.25, 0.50, 0.75, 0.90]

def lagrangian_radii(r_sorted, N):
    """Return Lagrangian radii at LAGR_FRACS for a sorted radius array."""
    return [r_sorted[max(0, int(f * N) - 1)] for f in LAGR_FRACS]

# ============================================================
#  WAKE DENSITY MAP HELPER
# ============================================================

def build_wake_map(pos_bg, pos_pert, vel_pert, n_bins=60, half_size=2.0):
    """Return a 2D histogram of background particles in the comoving
    frame of the perturber, with x-axis aligned with v_perturber.
    Returns (H, xedges, yedges) where H[i,j] is the particle count."""
    v_mag = np.linalg.norm(vel_pert)
    if v_mag < 1e-10:
        e_x = np.array([1.0, 0.0, 0.0])
    else:
        e_x = vel_pert / v_mag

    # build an orthonormal frame: e_x (motion dir), e_y (in orbital plane)
    # e_z is cross product — not used in the 2D map
    trial = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(e_x, trial)) > 0.9:
        trial = np.array([0.0, 0.0, 1.0])
    e_y = np.cross(e_x, trial)
    e_y /= np.linalg.norm(e_y)

    # relative positions in the perturber frame
    dr = pos_bg - pos_pert           # (N_bg, 3)
    xi = dr @ e_x                   # component along velocity
    yi = dr @ e_y                   # component perpendicular (in-plane)

    edges = np.linspace(-half_size, half_size, n_bins + 1)
    H, xe, ye = np.histogram2d(xi, yi, bins=[edges, edges])
    return H, xe, ye

# ============================================================
#  MODE A — PER-RUN ANALYSIS
# ============================================================

def run_single_analysis(run_dir, eta, R0_init, N_bg, eps):
    """Full analysis of one run.  Writes PDFs and perturber_stats.npz."""

    outfile = os.path.join(run_dir, 'plummer_perturber.out')
    if not os.path.isfile(outfile):
        print(f"[ERROR] Output file not found: {outfile}", flush=True)
        sys.exit(1)

    N_total  = N_bg + 1        # perturber is index 0
    M_p      = eta * M_tot
    m_i      = M_tot / N_bg
    ln_lam   = math.log(1.0 / eta) if eta < 1.0 else 1.0  # Coulomb log ~ ln(1/eta)

    print(f"\n{'='*60}", flush=True)
    print(f"  summary_perturber.py — per-run analysis", flush=True)
    print(f"  run_dir  : {run_dir}", flush=True)
    print(f"  eta={eta:.4f}  R0={R0_init:.4f}  N_bg={N_bg}  eps={eps:.4f}", flush=True)
    print(f"  M_pert={M_p:.6f}  m_bg={m_i:.6e}  ln_Lambda={ln_lam:.3f}", flush=True)
    print(f"{'='*60}", flush=True)

    # ---- storage arrays (one entry per snapshot) ----
    times         = []

    # perturber kinematic time series
    R_M           = []   # orbital radius
    v_M           = []   # speed
    Lz_M          = []   # z-component of specific angular momentum
    L_M           = []   # magnitude of specific angular momentum
    E_orb_M       = []   # specific orbital energy (kinetic + phi_background)

    # background global quantities
    K_bg_arr      = []   # background kinetic energy
    W_bg_arr      = []   # background potential energy (from phi)
    E_tot_arr     = []   # total system energy
    virial_arr    = []   # 2K_bg / |W_bg|
    sigma_v_arr   = []   # background 3D rms velocity
    r_cm_bg_arr   = []   # background CM position magnitude

    # Lagrangian radii of background only
    lagr_arr      = []   # list of [r10, r25, r50, r75, r90]

    # Chandrasekhar prediction (evaluated at each snapshot)
    a_chandra_arr = []   # theoretical friction deceleration

    # snapshots for wake map (store 3 snapshots: early, mid, late)
    wake_snaps    = {}   # {label: (H, xe, ye)}

    # perturber trajectory (for orbit plot)
    x_M_traj      = []
    y_M_traj      = []

    # ---- first pass: count snapshots ----
    print("  Counting snapshots...", end=' ', flush=True)
    n_snaps_total = 0
    with open(outfile, 'r') as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    n_check = int(line)
                    if n_check == N_total:
                        n_snaps_total += 1
                except ValueError:
                    pass
    print(f"{n_snaps_total} snapshots found.", flush=True)

    if n_snaps_total == 0:
        print("[ERROR] No valid snapshots found.", flush=True)
        sys.exit(1)

    snap_early = max(0,  n_snaps_total //  10)
    snap_mid   = max(0,  n_snaps_total //  2)
    snap_late  = max(0,  n_snaps_total  -  1)

    # ---- second pass: streaming analysis ----
    print("  Streaming analysis...", flush=True)

    E0_total  = None   # first-snapshot total energy (for conservation check)
    snap_idx  = 0

    for snap in iter_snapshots(outfile, N_total):
        t      = snap['t']
        pos    = snap['pos']
        vel    = snap['vel']
        phi    = snap['phi']

        # --- separate perturber (index 0) from background (1..N_total) ---
        pos_p  = pos[0]
        vel_p  = vel[0]
        phi_p  = phi[0]        # potential at perturber location (total)

        pos_bg = pos[1:]       # (N_bg, 3)
        vel_bg = vel[1:]       # (N_bg, 3)
        phi_bg_arr = phi[1:]   # (N_bg,)  potential at each bg particle

        # --- background CM (excluding perturber) ---
        r_cm_bg = np.mean(pos_bg, axis=0)
        v_cm_bg = np.mean(vel_bg, axis=0)

        # --- perturber in CM frame of background ---
        dr_p   = pos_p - r_cm_bg
        dv_p   = vel_p - v_cm_bg

        R_p    = float(np.linalg.norm(dr_p))   # orbital radius
        v_p_s  = float(np.linalg.norm(dv_p))   # speed

        # specific angular momentum
        L_vec  = np.cross(dr_p, dv_p)
        Lz_p   = float(L_vec[2])
        L_p    = float(np.linalg.norm(L_vec))

        # orbital energy of perturber:
        #   E_orb = (1/2) v^2 + phi_background
        # phi[0] is the TOTAL potential at the perturber (including self).
        # The perturber's self-potential is phi_self = G*M_p/eps (softened,
        # at r=0 from itself), which is a constant and not physically
        # meaningful.  We approximate phi_background ~ phi_total - phi_self_correction.
        # For diagnostic purposes we use phi[0] directly; the self-energy
        # term cancels when comparing E_orb at different times.
        E_orb_p = 0.5 * v_p_s**2 + phi_p

        # --- background energetics ---
        v_bg_sq    = np.sum(vel_bg**2, axis=1)       # (N_bg,)
        K_bg       = 0.5 * m_i * np.sum(v_bg_sq)
        W_bg       = 0.5 * m_i * np.sum(phi_bg_arr)  # W = (1/2)*sum(m_i * phi_i)
        # Note: phi_bg_arr already includes the perturber's contribution to
        # each background particle's potential, which is physical.

        # --- total system energy ---
        K_pert    = 0.5 * M_p * v_p_s**2
        E_total   = K_bg + K_pert + 0.5 * (M_p * phi_p + m_i * np.sum(phi_bg_arr))

        if E0_total is None:
            E0_total = E_total

        # --- virial ratio of background ---
        virial = -2.0 * K_bg / W_bg if abs(W_bg) > 1e-12 else 0.0

        # --- background velocity dispersion ---
        sigma_v = float(math.sqrt(np.mean(v_bg_sq)))

        # --- background Lagrangian radii ---
        r_bg_mag  = np.linalg.norm(pos_bg - r_cm_bg, axis=1)
        r_bg_sort = np.sort(r_bg_mag)
        lagr      = lagrangian_radii(r_bg_sort, N_bg)

        # --- Chandrasekhar deceleration at current R_p, v_p ---
        a_ch = chandrasekhar_decel(R_p, v_p_s, M_p, ln_lam)

        # --- store time series ---
        times.append(t)
        R_M.append(R_p)
        v_M.append(v_p_s)
        Lz_M.append(Lz_p)
        L_M.append(L_p)
        E_orb_M.append(E_orb_p)
        K_bg_arr.append(K_bg)
        W_bg_arr.append(W_bg)
        E_tot_arr.append(E_total)
        virial_arr.append(virial)
        sigma_v_arr.append(sigma_v)
        r_cm_bg_arr.append(float(np.linalg.norm(r_cm_bg)))
        lagr_arr.append(lagr)
        a_chandra_arr.append(a_ch)
        x_M_traj.append(float(pos_p[0]))
        y_M_traj.append(float(pos_p[1]))

        # --- save wake maps at three epochs ---
        if snap_idx == snap_early:
            H, xe, ye = build_wake_map(pos_bg, pos_p, vel_p)
            wake_snaps['early'] = (H, xe, ye, t)
        elif snap_idx == snap_mid:
            H, xe, ye = build_wake_map(pos_bg, pos_p, vel_p)
            wake_snaps['mid'] = (H, xe, ye, t)
        elif snap_idx == snap_late:
            H, xe, ye = build_wake_map(pos_bg, pos_p, vel_p)
            wake_snaps['late'] = (H, xe, ye, t)

        snap_idx += 1

        if snap_idx % 200 == 0:
            print(f"    processed {snap_idx}/{n_snaps_total} snapshots...",
                  flush=True)

    print(f"  Streaming complete. {snap_idx} snapshots processed.", flush=True)

    # ---- convert to numpy arrays ----
    times         = np.array(times)
    R_M           = np.array(R_M)
    v_M           = np.array(v_M)
    Lz_M          = np.array(Lz_M)
    L_M           = np.array(L_M)
    E_orb_M       = np.array(E_orb_M)
    K_bg_arr      = np.array(K_bg_arr)
    W_bg_arr      = np.array(W_bg_arr)
    E_tot_arr     = np.array(E_tot_arr)
    virial_arr    = np.array(virial_arr)
    sigma_v_arr   = np.array(sigma_v_arr)
    r_cm_bg_arr   = np.array(r_cm_bg_arr)
    lagr_arr      = np.array(lagr_arr)       # (n_snaps, 5)
    a_chandra_arr = np.array(a_chandra_arr)
    x_M_traj      = np.array(x_M_traj)
    y_M_traj      = np.array(y_M_traj)

    # ---- derived diagnostics ----

    # energy conservation
    dE_over_E0 = np.abs((E_tot_arr - E0_total) / E0_total) if E0_total != 0 else E_tot_arr * 0

    # orbital decay rate (smoothed finite difference of R_M)
    # smooth over ~10 snapshots to suppress one-orbit oscillations
    smooth_win  = max(3, len(times) // 50)
    R_M_smooth  = uniform_filter1d(R_M, size=smooth_win)
    v_M_smooth  = uniform_filter1d(v_M, size=smooth_win)
    L_M_smooth  = uniform_filter1d(L_M, size=smooth_win)
    dt_snap     = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    dR_dt       = np.gradient(R_M_smooth, dt_snap)  # dR/dt

    # mean decay rate (slope of R_M over the full run)
    if len(times) > 1:
        slope_R, intercept_R, r_R, _, _ = stats.linregress(times, R_M)
        slope_L, _, _, _, _ = stats.linregress(times, L_M)
    else:
        slope_R = slope_L = 0.0

    # Coulomb logarithm ratio: measured decel / Chandrasekhar
    # estimated deceleration from smoothed v_M
    dv_dt = np.gradient(v_M_smooth, dt_snap)
    # ratio = |dv/dt| / a_chandra  (where a_chandra uses ln_lam=1)
    a_chandra_lam1 = np.array([chandrasekhar_decel(R_M[i], v_M[i], M_p, 1.0)
                                for i in range(len(times))])
    with np.errstate(divide='ignore', invalid='ignore'):
        lnlam_eff = np.where(a_chandra_lam1 > 1e-12,
                             np.abs(dv_dt) / a_chandra_lam1,
                             np.nan)

    # Chandrasekhar prediction trajectory:
    # integrate dR/dt = -a_fric(R,v_circ(R)) / v_circ(R) * R
    # using forward Euler from R0 at each time step dt_snap
    R_chandra = np.empty(len(times))
    R_chandra[0] = R_M[0]
    for i in range(1, len(times)):
        R_prev = R_chandra[i-1]
        if R_prev < 0.01:
            R_chandra[i] = R_prev
            continue
        vc    = plummer_vcirc(R_prev)
        a_fc  = chandrasekhar_decel(R_prev, max(vc, 0.01), M_p, ln_lam)
        dR    = -a_fc * R_prev / max(vc, 0.01) * dt_snap
        R_chandra[i] = max(0.01, R_prev + dR)

    # background heating: DeltaK_bg = K_bg(t) - K_bg(0)
    DK_bg = K_bg_arr - K_bg_arr[0]

    # print scalar diagnostics
    print(f"\n  [SCALAR DIAGNOSTICS]", flush=True)
    print(f"    t_final              : {times[-1]:.2f}", flush=True)
    print(f"    R_M(0)               : {R_M[0]:.4f}", flush=True)
    print(f"    R_M(final)           : {R_M[-1]:.4f}", flush=True)
    print(f"    Delta R / R0         : {(R_M[0]-R_M[-1])/R_M[0]:.4f}", flush=True)
    print(f"    dR/dt (linear fit)   : {slope_R:.4e}", flush=True)
    print(f"    dL/dt (linear fit)   : {slope_L:.4e}", flush=True)
    print(f"    max |dE/E0|          : {np.max(dE_over_E0):.4e}", flush=True)
    print(f"    mean virial (bg)     : {np.mean(virial_arr):.4f}  (target: 1.000)", flush=True)
    print(f"    sigma_v drift        : {(sigma_v_arr[-1]-sigma_v_arr[0])/sigma_v_arr[0]:.4f}", flush=True)
    print(f"    DeltaK_bg / K_bg(0)  : {DK_bg[-1]/K_bg_arr[0]:.4f}  (bg heating)", flush=True)
    lnlam_valid = lnlam_eff[np.isfinite(lnlam_eff)]
    if len(lnlam_valid) > 0:
        print(f"    ln_Lambda_eff (med)  : {np.nanmedian(lnlam_eff):.3f}", flush=True)
        print(f"    ln_Lambda theory     : {ln_lam:.3f}", flush=True)

    # ============================================================
    #  PLOTS
    # ============================================================

    print("\n  Generating plots...", flush=True)
    plt.rcParams.update({'font.size': 9, 'figure.dpi': 120})
    figs_written = []

    # ---------- Figure 1: Orbital decay ----------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f'Orbital decay  |  η={eta:.3f}  R₀={R0_init:.3f}  '
                 f'ε={eps:.4f}  N={N_bg}', fontsize=10)

    ax = axes[0, 0]
    ax.plot(times / t_dyn, R_M, color='steelblue', lw=0.8,
            alpha=0.6, label='simulated R_M(t)')
    ax.plot(times / t_dyn, R_M_smooth, color='navy', lw=1.5,
            label=f'smoothed (win={smooth_win})')
    ax.plot(times / t_dyn, R_chandra, color='firebrick', lw=1.5,
            ls='--', label=f'Chandrasekhar (ln Λ={ln_lam:.2f})')
    ax.axhline(r_hm, color='gray', ls=':', lw=0.8, label=f'r_hm={r_hm:.3f}')
    ax.set_xlabel('t / t_dyn')
    ax.set_ylabel('R_M  [code units]')
    ax.set_title('Orbital radius decay')
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    ax.plot(times / t_dyn, L_M, color='steelblue', lw=0.8,
            alpha=0.6, label='|L_M|')
    ax.plot(times / t_dyn, L_M_smooth, color='navy', lw=1.5,
            label='smoothed')
    ax.plot(times / t_dyn, Lz_M, color='darkorange', lw=0.8,
            alpha=0.6, label='L_z')
    ax.set_xlabel('t / t_dyn')
    ax.set_ylabel('Specific angular momentum')
    ax.set_title('Angular momentum loss')
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    ax.plot(times / t_dyn, v_M, color='steelblue', lw=0.8,
            alpha=0.5, label='|v_M|')
    ax.plot(times / t_dyn, v_M_smooth, color='navy', lw=1.5,
            label='smoothed')
    vc_theory = np.array([plummer_vcirc(r) for r in R_M_smooth])
    ax.plot(times / t_dyn, vc_theory, color='firebrick', lw=1.2,
            ls='--', label='v_circ(R_M) theory')
    ax.set_xlabel('t / t_dyn')
    ax.set_ylabel('Speed  [code units]')
    ax.set_title('Perturber speed vs circular velocity')
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.plot(times / t_dyn, np.abs(dv_dt), color='steelblue', lw=0.8,
            alpha=0.6, label='|dv/dt| measured')
    ax.plot(times / t_dyn, a_chandra_arr, color='firebrick', lw=1.5,
            ls='--', label=f'a_Chandra (ln Λ={ln_lam:.2f})')
    ax.set_xlabel('t / t_dyn')
    ax.set_ylabel('Deceleration  [code units]')
    ax.set_title('Friction deceleration vs Chandrasekhar')
    ax.set_yscale('log')
    ax.legend(fontsize=7)

    plt.tight_layout()
    fname1 = os.path.join(run_dir, 'plot_orbital_decay.pdf')
    fig.savefig(fname1, bbox_inches='tight')
    plt.close(fig)
    figs_written.append(fname1)
    print(f"    written: {os.path.basename(fname1)}", flush=True)

    # ---------- Figure 2: Energy and background ----------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f'Energetics and background  |  η={eta:.3f}  N={N_bg}', fontsize=10)

    ax = axes[0, 0]
    ax.plot(times / t_dyn, dE_over_E0, color='navy', lw=1.0)
    ax.set_xlabel('t / t_dyn')
    ax.set_ylabel('|ΔE_tot / E_0|')
    ax.set_title('Total energy conservation')
    ax.set_yscale('log')
    ax.axhline(1e-3, color='firebrick', ls='--', lw=0.8, label='0.1% threshold')
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    ax.plot(times / t_dyn, E_orb_M - E_orb_M[0], color='firebrick',
            lw=1.2, label='ΔE_orb (perturber)')
    ax.plot(times / t_dyn, DK_bg, color='steelblue', lw=1.2,
            label='ΔK_bg (background heating)')
    ax.plot(times / t_dyn,
            (E_orb_M - E_orb_M[0]) + DK_bg,
            color='gray', lw=1.0, ls='--', label='Sum (should ~ 0)')
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xlabel('t / t_dyn')
    ax.set_ylabel('Energy change  [code units]')
    ax.set_title('Energy partitioning')
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    ax.plot(times / t_dyn, virial_arr, color='steelblue', lw=1.0)
    ax.axhline(1.0, color='firebrick', ls='--', lw=0.8, label='virial = 1')
    ax.set_xlabel('t / t_dyn')
    ax.set_ylabel('2K_bg / |W_bg|')
    ax.set_title('Background virial ratio')
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    lagr_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for k, (frac, col) in enumerate(zip(LAGR_FRACS, lagr_colors)):
        ax.plot(times / t_dyn, lagr_arr[:, k], color=col, lw=1.0,
                label=f'{int(frac*100)}%')
    ax.axhline(r_hm, color='gray', ls=':', lw=0.8, label='r_hm')
    ax.set_xlabel('t / t_dyn')
    ax.set_ylabel('Lagrangian radius  [code units]')
    ax.set_title('Background Lagrangian radii')
    ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    fname2 = os.path.join(run_dir, 'plot_energetics.pdf')
    fig.savefig(fname2, bbox_inches='tight')
    plt.close(fig)
    figs_written.append(fname2)
    print(f"    written: {os.path.basename(fname2)}", flush=True)

    # ---------- Figure 3: Orbit trajectory + wake maps ----------
    n_wake = len(wake_snaps)
    fig, axes = plt.subplots(1, n_wake + 1, figsize=(4 * (n_wake + 1), 4))
    if n_wake + 1 == 1:
        axes = [axes]

    ax = axes[0]
    sc = ax.scatter(x_M_traj, y_M_traj, c=times, cmap='viridis',
                    s=1.5, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='t [code units]')
    ax.set_aspect('equal')
    ax.set_xlabel('x  [code units]')
    ax.set_ylabel('y  [code units]')
    ax.set_title(f'Perturber trajectory  η={eta:.3f}')
    circle = plt.Circle((0, 0), r_hm, color='gray', fill=False,
                         ls='--', lw=0.8, label='r_hm')
    ax.add_patch(circle)
    ax.legend(fontsize=7)

    for k, (label, (H, xe, ye, t_wake)) in enumerate(sorted(wake_snaps.items())):
        ax = axes[k + 1]
        Hlog = np.log10(H + 1)
        im = ax.pcolormesh(xe, ye, Hlog.T, cmap='inferno')
        plt.colorbar(im, ax=ax, label='log10(N+1)')
        ax.set_xlabel('Along v_M  [code units]')
        ax.set_ylabel('Perpendicular')
        ax.set_title(f'Wake map ({label}, t={t_wake:.1f})')
        ax.axvline(0, color='white', ls='--', lw=0.5)
        ax.set_aspect('equal')

    plt.tight_layout()
    fname3 = os.path.join(run_dir, 'plot_orbit_wake.pdf')
    fig.savefig(fname3, bbox_inches='tight')
    plt.close(fig)
    figs_written.append(fname3)
    print(f"    written: {os.path.basename(fname3)}", flush=True)

    # ---------- Figure 4: Coulomb logarithm ----------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f'Effective Coulomb logarithm  |  η={eta:.3f}', fontsize=10)

    ax = axes[0]
    ax.plot(times / t_dyn, lnlam_eff, color='steelblue', lw=0.8, alpha=0.6)
    ax.axhline(ln_lam, color='firebrick', ls='--', lw=1.0,
               label=f'ln(1/η) = {ln_lam:.2f}')
    ax.axhline(math.log(N_bg), color='darkorange', ls=':', lw=1.0,
               label=f'ln(N) = {math.log(N_bg):.2f}')
    ax.set_xlabel('t / t_dyn')
    ax.set_ylabel('ln Λ_eff')
    ax.set_title('Effective Coulomb logarithm vs time')
    ax.set_ylim(0, max(5, math.log(N_bg) * 1.5))
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(R_M[np.isfinite(lnlam_eff)],
            lnlam_eff[np.isfinite(lnlam_eff)],
            '.', color='steelblue', ms=2, alpha=0.5)
    ax.axhline(ln_lam, color='firebrick', ls='--', lw=1.0,
               label=f'ln(1/η) = {ln_lam:.2f}')
    ax.set_xlabel('R_M  [code units]')
    ax.set_ylabel('ln Λ_eff')
    ax.set_title('Effective Coulomb logarithm vs orbital radius')
    ax.legend(fontsize=7)

    plt.tight_layout()
    fname4 = os.path.join(run_dir, 'plot_coulomb_log.pdf')
    fig.savefig(fname4, bbox_inches='tight')
    plt.close(fig)
    figs_written.append(fname4)
    print(f"    written: {os.path.basename(fname4)}", flush=True)

    # ============================================================
    #  SAVE LIGHTWEIGHT NPZ
    # ============================================================

    npz_path = os.path.join(run_dir, 'perturber_stats.npz')
    np.savez(npz_path,
             # scalars
             eta=np.float64(eta),
             R0_init=np.float64(R0_init),
             N_bg=np.int64(N_bg),
             eps=np.float64(eps),
             ln_lam=np.float64(ln_lam),
             E0_total=np.float64(E0_total),
             # time series
             times=times,
             R_M=R_M,
             v_M=v_M,
             L_M=L_M,
             Lz_M=Lz_M,
             E_orb_M=E_orb_M,
             K_bg=K_bg_arr,
             W_bg=W_bg_arr,
             E_tot=E_tot_arr,
             dE_over_E0=dE_over_E0,
             virial=virial_arr,
             sigma_v=sigma_v_arr,
             r_cm_bg=r_cm_bg_arr,
             lagr=lagr_arr,
             a_chandra=a_chandra_arr,
             R_chandra=R_chandra,
             lnlam_eff=lnlam_eff,
             dR_dt=dR_dt,
             # scalar summaries
             slope_R=np.float64(slope_R),
             slope_L=np.float64(slope_L),
             max_dE=np.float64(np.max(dE_over_E0)),
             mean_virial=np.float64(np.mean(virial_arr)),
             lnlam_eff_median=np.float64(np.nanmedian(lnlam_eff)),
             DK_bg_final=np.float64(DK_bg[-1]),
             DK_bg_frac=np.float64(DK_bg[-1] / K_bg_arr[0]),
             )

    print(f"    saved: {os.path.basename(npz_path)}", flush=True)
    print(f"\n  PDFs written: {len(figs_written)}", flush=True)
    for f in figs_written:
        print(f"    {f}", flush=True)
    print(f"{'='*60}", flush=True)

# ============================================================
#  MODE B — COMBINED ENSEMBLE ANALYSIS
# ============================================================

def run_combined_analysis(eta, R0_init):
    """Load all perturber_stats.npz files and produce ensemble plots."""

    base_dir  = os.getcwd()
    run_dirs  = sorted([
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if d.startswith('run_perturber_') and
           os.path.isdir(os.path.join(base_dir, d))
    ])

    npz_files = [os.path.join(d, 'perturber_stats.npz')
                 for d in run_dirs
                 if os.path.isfile(os.path.join(d, 'perturber_stats.npz'))]

    print(f"\n{'='*60}", flush=True)
    print(f"  summary_perturber.py — combined ensemble analysis", flush=True)
    print(f"  Found {len(npz_files)} completed runs.", flush=True)
    print(f"{'='*60}", flush=True)

    if len(npz_files) == 0:
        print("[ERROR] No perturber_stats.npz files found.", flush=True)
        sys.exit(1)

    # ---- load all runs ----
    all_times        = []
    all_R_M          = []
    all_L_M          = []
    all_E_orb        = []
    all_dE           = []
    all_virial       = []
    all_K_bg         = []
    all_lnlam        = []
    all_R_chandra    = []
    all_slope_R      = []
    all_slope_L      = []
    all_max_dE       = []
    all_mean_virial  = []
    all_DK_bg_frac   = []
    all_lnlam_med    = []

    ln_lam_ref = None

    for npz_path in npz_files:
        try:
            d = np.load(npz_path, allow_pickle=False)
        except Exception as e:
            print(f"  [warn] Could not load {npz_path}: {e}", flush=True)
            continue

        all_times.append(d['times'])
        all_R_M.append(d['R_M'])
        all_L_M.append(d['L_M'])
        all_E_orb.append(d['E_orb_M'])
        all_dE.append(d['dE_over_E0'])
        all_virial.append(d['virial'])
        all_K_bg.append(d['K_bg'])
        all_lnlam.append(d['lnlam_eff'])
        all_R_chandra.append(d['R_chandra'])
        all_slope_R.append(float(d['slope_R']))
        all_slope_L.append(float(d['slope_L']))
        all_max_dE.append(float(d['max_dE']))
        all_mean_virial.append(float(d['mean_virial']))
        all_DK_bg_frac.append(float(d['DK_bg_frac']))
        all_lnlam_med.append(float(d['lnlam_eff_median']))
        if ln_lam_ref is None:
            ln_lam_ref = float(d['ln_lam'])

    n_runs = len(all_R_M)
    print(f"  Successfully loaded: {n_runs} runs", flush=True)

    if n_runs == 0:
        print("[ERROR] All npz files failed to load.", flush=True)
        sys.exit(1)

    # ---- interpolate all runs onto a common time grid ----
    # use the time array of the first run as the reference
    t_ref  = all_times[0]
    R_interp = []
    L_interp = []

    for i in range(n_runs):
        t_i   = all_times[i]
        R_i   = all_R_M[i]
        L_i   = all_L_M[i]
        t_min = max(t_ref[0],  t_i[0])
        t_max = min(t_ref[-1], t_i[-1])
        mask  = (t_ref >= t_min) & (t_ref <= t_max)
        R_interp_i = np.interp(t_ref[mask], t_i, R_i)
        L_interp_i = np.interp(t_ref[mask], t_i, L_i)
        R_interp.append((t_ref[mask], R_interp_i))
        L_interp.append((t_ref[mask], L_interp_i))

    # ensemble mean and std on the full reference grid
    # (use only the common time range across ALL runs)
    t_start = max(ri[0][0]  for ri in R_interp)
    t_end   = min(ri[0][-1] for ri in R_interp)
    t_mask  = (t_ref >= t_start) & (t_ref <= t_end)
    t_common = t_ref[t_mask]

    R_stack = np.array([np.interp(t_common, ri[0], ri[1]) for ri in R_interp])
    L_stack = np.array([np.interp(t_common, li[0], li[1]) for li in L_interp])

    R_mean  = np.mean(R_stack, axis=0)
    R_std   = np.std(R_stack,  axis=0)
    L_mean  = np.mean(L_stack, axis=0)
    L_std   = np.std(L_stack,  axis=0)

    # Chandrasekhar on common grid (from first run, re-evaluated)
    ln_lam_use = ln_lam_ref if ln_lam_ref else math.log(1.0 / eta)
    R_chandra_mean = np.interp(t_common, all_times[0], all_R_chandra[0])

    # scalar summary statistics
    slope_R_arr     = np.array(all_slope_R)
    max_dE_arr      = np.array(all_max_dE)
    virial_arr_comb = np.array(all_mean_virial)
    DK_frac_arr     = np.array(all_DK_bg_frac)
    lnlam_med_arr   = np.array(all_lnlam_med)

    print(f"\n  [ENSEMBLE SUMMARY  (n_runs={n_runs})]", flush=True)
    print(f"    <dR/dt>        : {np.mean(slope_R_arr):.4e} ± {np.std(slope_R_arr):.4e}", flush=True)
    print(f"    <max |dE/E0|>  : {np.mean(max_dE_arr):.4e} ± {np.std(max_dE_arr):.4e}", flush=True)
    print(f"    <virial>       : {np.mean(virial_arr_comb):.4f} ± {np.std(virial_arr_comb):.4f}", flush=True)
    print(f"    <DK_bg/K0>     : {np.mean(DK_frac_arr):.4f} ± {np.std(DK_frac_arr):.4f}", flush=True)
    print(f"    <ln Lam_eff>   : {np.nanmean(lnlam_med_arr):.3f} ± {np.nanstd(lnlam_med_arr):.3f}", flush=True)
    print(f"    ln Lam theory  : {ln_lam_use:.3f}  (ln(1/eta))", flush=True)

    # ============================================================
    #  COMBINED PLOTS
    # ============================================================

    print("\n  Generating combined plots...", flush=True)
    plt.rcParams.update({'font.size': 9, 'figure.dpi': 120})
    figs_written = []

    # ---------- Figure C1: Ensemble orbital decay ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Ensemble orbital decay  |  η={eta:.3f}  '
                 f'N_runs={n_runs}  ln Λ={ln_lam_use:.2f}', fontsize=10)

    ax = axes[0]
    for i, (t_i, R_i) in enumerate(R_interp):
        ax.plot(t_i / t_dyn, R_i, color='steelblue', lw=0.5, alpha=0.3)
    ax.plot(t_common / t_dyn, R_mean, color='navy', lw=2.0,
            label=f'ensemble mean (n={n_runs})')
    ax.fill_between(t_common / t_dyn,
                    R_mean - R_std, R_mean + R_std,
                    color='steelblue', alpha=0.2, label='±1σ')
    ax.plot(t_common / t_dyn, R_chandra_mean, color='firebrick',
            lw=1.5, ls='--', label=f'Chandrasekhar')
    ax.axhline(r_hm, color='gray', ls=':', lw=0.8, label=f'r_hm')
    ax.set_xlabel('t / t_dyn')
    ax.set_ylabel('R_M  [code units]')
    ax.set_title('Orbital radius decay — ensemble')
    ax.legend(fontsize=7)

    ax = axes[1]
    for i, (t_i, L_i) in enumerate(L_interp):
        ax.plot(t_i / t_dyn, L_i, color='darkorange', lw=0.5, alpha=0.3)
    ax.plot(t_common / t_dyn, L_mean, color='saddlebrown', lw=2.0,
            label=f'ensemble mean')
    ax.fill_between(t_common / t_dyn,
                    L_mean - L_std, L_mean + L_std,
                    color='darkorange', alpha=0.2, label='±1σ')
    ax.set_xlabel('t / t_dyn')
    ax.set_ylabel('|L_M|  [code units]')
    ax.set_title('Angular momentum — ensemble')
    ax.legend(fontsize=7)

    plt.tight_layout()
    fname = 'combined_orbital_decay.pdf'
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    figs_written.append(fname)
    print(f"    written: {fname}", flush=True)

    # ---------- Figure C2: Scalar distributions ----------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f'Scalar distributions across realisations  |  '
                 f'η={eta:.3f}  n_runs={n_runs}', fontsize=10)

    ax = axes[0, 0]
    ax.hist(slope_R_arr, bins=max(5, n_runs//3), color='steelblue',
            edgecolor='white', lw=0.5)
    ax.axvline(np.mean(slope_R_arr), color='firebrick', ls='--',
               label=f'mean={np.mean(slope_R_arr):.3e}')
    ax.set_xlabel('dR_M/dt  [code units / code time]')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of orbital decay rate')
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    ax.hist(max_dE_arr, bins=max(5, n_runs//3), color='darkorange',
            edgecolor='white', lw=0.5)
    ax.axvline(1e-3, color='firebrick', ls='--', lw=1.0,
               label='0.1% threshold')
    ax.set_xlabel('max |ΔE/E₀|')
    ax.set_ylabel('Count')
    ax.set_title('Energy conservation per run')
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    ax.hist(virial_arr_comb, bins=max(5, n_runs//3), color='green',
            edgecolor='white', lw=0.5)
    ax.axvline(1.0, color='firebrick', ls='--', lw=1.0, label='virial = 1')
    ax.set_xlabel('Mean 2K_bg / |W_bg|')
    ax.set_ylabel('Count')
    ax.set_title('Background virial ratio distribution')
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    valid_mask = np.isfinite(lnlam_med_arr)
    if np.any(valid_mask):
        ax.hist(lnlam_med_arr[valid_mask], bins=max(5, n_runs//3),
                color='purple', edgecolor='white', lw=0.5)
        ax.axvline(ln_lam_use, color='firebrick', ls='--',
                   label=f'ln(1/η) = {ln_lam_use:.2f}')
        ax.axvline(math.log(N_BG), color='darkorange', ls=':',
                   label=f'ln(N) = {math.log(N_BG):.2f}')
    ax.set_xlabel('Median ln Λ_eff per run')
    ax.set_ylabel('Count')
    ax.set_title('Effective Coulomb logarithm distribution')
    ax.legend(fontsize=7)

    plt.tight_layout()
    fname = 'combined_scalar_distributions.pdf'
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    figs_written.append(fname)
    print(f"    written: {fname}", flush=True)

    # ---------- Figure C3: R_M(t_final) convergence ----------
    fig, ax = plt.subplots(figsize=(7, 4))
    # running mean as realisations are added
    R_final_each = np.array([np.interp(t_common[-1], R_interp[i][0], R_interp[i][1])
                              for i in range(n_runs)])
    running_mean = np.cumsum(R_final_each) / np.arange(1, n_runs + 1)
    running_se   = np.array([np.std(R_final_each[:k+1]) / math.sqrt(k+1)
                              for k in range(n_runs)])
    ax.plot(np.arange(1, n_runs + 1), running_mean, color='navy', lw=1.5,
            label='Running mean of R_M(t_final)')
    ax.fill_between(np.arange(1, n_runs + 1),
                    running_mean - running_se,
                    running_mean + running_se,
                    color='steelblue', alpha=0.3, label='±1 S.E.')
    ax.axhline(np.interp(t_common[-1], t_common, R_chandra_mean),
               color='firebrick', ls='--', lw=1.0,
               label='Chandrasekhar prediction')
    ax.set_xlabel('Number of realisations included')
    ax.set_ylabel('R_M(t_final)  [code units]')
    ax.set_title(f'Ensemble convergence check  |  η={eta:.3f}')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fname = 'combined_ensemble_convergence.pdf'
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    figs_written.append(fname)
    print(f"    written: {fname}", flush=True)

    print(f"\n  Combined PDFs written: {len(figs_written)}", flush=True)
    for f in figs_written:
        print(f"    {f}", flush=True)
    print(f"{'='*60}", flush=True)

# ============================================================
#  MAIN ENTRY POINT
# ============================================================

if args.combined:
    # Mode B
    run_combined_analysis(eta=ETA, R0_init=R0_ARG)
elif args.run_dir is not None:
    # Mode A
    if not os.path.isdir(args.run_dir):
        print(f"[ERROR] run_dir does not exist: {args.run_dir}", flush=True)
        sys.exit(1)
    run_single_analysis(run_dir=args.run_dir,
                        eta=ETA,
                        R0_init=R0_ARG,
                        N_bg=N_BG,
                        eps=EPS)
else:
    parser.print_help()
    sys.exit(1)

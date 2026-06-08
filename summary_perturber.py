#!/usr/bin/env python3
# summary_perturber.py
# Per-run analysis of Plummer sphere + perturber N-body simulation.
# Particle index 0 = perturber; indices 1..N_BG = background Plummer sphere.
#
# python3 summary_perturber.py <run_dir> --eta E --R0 R --N N [--eps EPS]
#
# Changes vs original:
#   [ADD M4] B(X(t)) diagnostic panel in plot_orbital_decay.pdf.
#   [ADD M5] Lz/|L| diagnostic panel in plot_orbital_decay.pdf.
#   [FIX 3]  chandrasekhar_decel uses the document's simplified Plummer form
#            (Sections 2-3):
#              a_DF(R) = 3*(M_p+m)*lnL*B(X) / (R^2*(1+R^2))
#              X(R)    = R*sqrt(3/(1+R^2))   [exact for Plummer+Jeans sigma]
#              B(X)    = erf(X) - (2X/sqrt(pi))*exp(-X^2)
#            use_plummer_df=True keeps the exact Eddington bracket as comparison,
#            evaluated using Monte Carlo integration.
#   [FIX 4]  ODE uses document Section 6 formula directly:
#              dR/dt = -6*(M_p+m)*lnL*B(X(R))/R^2 * (1+R^2)^(3/4)/(4+R^2)
#            integrated with Euler dt=0.01 (document Section 10).
#   [FIX 5]  chandrasekhar_decel now uses the exact Chandrasekhar formula at
#            the actual perturber speed v_M, not the circular-orbit approximation:
#              a_DF = 3*(M_p+m)*lnL*B(v_M) / (v_M^2 * (1+R^2)^(5/2))
#            The old form 3*(M_p+m)*lnL*B / (R^2*(1+R^2)) is equivalent only
#            when v_M = v_circ = R/(1+R^2)^(3/4).  Using v_circ^2 instead of
#            v_M^2 in the denominator caused lnlam_eff to carry a spurious
#            factor of v_circ^2/v_M^2, which oscillates at the orbital frequency
#            and is the dominant source of noise in plot_coulomb_log.pdf.
#            The Maxwell branch X argument is also fixed to v_M/(sqrt(2)*sigma(R))
#            instead of the circular-orbit form R*sqrt(3/(1+R^2)).
#            The Euler ODE in _chandra_ode_euler has its own hardcoded circular-
#            orbit formula and does NOT call chandrasekhar_decel; it is unchanged.

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys, math, json, argparse
from scipy import stats
from scipy.ndimage import gaussian_filter1d

# ---- Parsing ----

parser = argparse.ArgumentParser()
parser.add_argument("run_dir", help="directory containing plummer_perturber.out")
parser.add_argument("--eta", type=float, default=0.05)
parser.add_argument("--R0", type=float, default=None)
parser.add_argument("--N", type=int, default=10000)
parser.add_argument("--eps", type=float, default=None)
args = parser.parse_args()

run_dir = args.run_dir
if not os.path.isdir(run_dir):
    print(f"[ERROR] run_dir not found: {run_dir}")
    sys.exit(1)

# ---- constants ----

G = 1.0
M_tot = 1.0
b = 1.0
ETA = args.eta
M_p = ETA * M_tot
N_BG = args.N
m_bg = M_tot / N_BG
N_total = N_BG + 1

r_hm = b / math.sqrt(2.0 ** (2.0 / 3.0) - 1.0)
rho_c = 3.0 * M_tot / (4.0 * math.pi * b**3)
t_dyn = math.sqrt(3.0 * math.pi / (16.0 * G * rho_c))
R0_ARG = args.R0 if args.R0 is not None else r_hm
sv_theory = math.sqrt(3.0 * math.pi * G * M_tot / (32.0 * b))

LAG_FRACS = [0.10, 0.25, 0.50, 0.75, 0.90]

# resolve eps
EPS = args.eps
if EPS is None:
    pfile = os.path.join(run_dir, "params_recommended.json")
    if os.path.isfile(pfile):
        with open(pfile) as f:
            EPS = json.load(f)["eps"]
        print(f"eps read from params_recommended.json: {EPS}")
    else:
        print("[ERROR] --eps not given and params_recommended.json not found")
        sys.exit(1)

outfile = os.path.join(run_dir, "plummer_perturber.out")
if not os.path.isfile(outfile):
    print(f"[ERROR] not found: {outfile}")
    sys.exit(1)

print(f"run_dir={run_dir}  eta={ETA}  R0={R0_ARG:.4f}  N={N_BG}  eps={EPS:.4f}")

# ---- Plummer functions ----


def plummer_rho(r):
    return (3.0 * M_tot / (4.0 * math.pi * b**3)) * (1.0 + (r / b) ** 2) ** (-2.5)


def plummer_mass_enc(r):
    return M_tot * r**3 / (r**2 + b**2) ** 1.5


def plummer_vcirc(r):
    return math.sqrt(G * plummer_mass_enc(r) / r) if r > 0 else 0.0


def plummer_dLdR(R):
    """d(R * v_circ) / dR for a Plummer sphere."""
    if R <= 0:
        return 1e-10
    R2 = R * R
    b2 = b * b
    vc = plummer_vcirc(R)
    if vc < 1e-12:
        return 1e-10
    dvc = G * M_tot * R * (2.0 * b2 - R2) / (2.0 * vc * (R2 + b2) ** 2.5)
    return vc + R * dvc


def plummer_cdf(r):
    return r**3 / (r**2 + b**2) ** 1.5


def sigma_theory_jeans(r_arr):
    return np.sqrt(G * M_tot / (6.0 * np.sqrt(r_arr**2 + b**2)))


# ---- Plummer DF bracket — exact, from Eddington inversion ----          [FIX 3]
#
# f(eps) propto psi^(7/2).  The velocity-space integral up to v_M is:
#   I_v(r, v_M) = rho(r) * B_Plummer(r, v_M)
# where:
#   B_Plummer = integral_0^{q_M} q^2*(1-q^2)^(7/2) dq / I_q_full
#   q_M = v_M / v_esc(r),   I_q_full = integral_0^1 q^2*(1-q^2)^(7/2) dq = 7*pi/512
#
# Precomputed deterministically using the cumulative trapezoid rule on a fine
# q grid.  This replaces the original Monte Carlo approach which was slow
# (~20M random draws in the Euler loop) and introduced stochastic noise.
# Exact analytic value: I_q_full = 7*pi/512 ~ 0.042938.

_q_cdf_grid = np.linspace(0.0, 1.0, 10000)
_f_cdf_grid = _q_cdf_grid**2 * (1.0 - _q_cdf_grid**2) ** 3.5
_dq_cdf     = _q_cdf_grid[1] - _q_cdf_grid[0]
_I_cum_cdf  = np.zeros(10000)
_I_cum_cdf[1:] = np.cumsum(0.5 * (_f_cdf_grid[:-1] + _f_cdf_grid[1:]) * _dq_cdf)
_I_q_full   = _I_cum_cdf[-1]   # numerically: 7*pi/512 ~ 0.042938
_B_cdf      = _I_cum_cdf / _I_q_full   # CDF table: B_Plummer(q_M)

# ---- Maxwellian bracket — CDF table, mirrors B_Plummer approach ----
#
# B_Maxwell(X) = (4/sqrt(pi)) * integral_0^X u^2 * exp(-u^2) du
#              = erf(X) - (2X/sqrt(pi)) * exp(-X^2)   [analytic equivalent]
# where u = v / (sqrt(2)*sigma),  X = v_M / (sqrt(2)*sigma).
#
# For a Plummer circular orbit X = R*sqrt(3/(1+R^2)) in [0, sqrt(3)] ~ [0, 1.73].
# Grid extends to 5.0 to cover non-circular v_M values safely.
# 50000 points gives the same point-density-per-unit as the Plummer grid.
# Verified to agree with math.erf form to < 2e-9 across the full X range.

_u_cdf_grid  = np.linspace(0.0, 5.0, 50000)
_f_maxw_grid = _u_cdf_grid**2 * np.exp(-_u_cdf_grid**2)
_du_cdf      = _u_cdf_grid[1] - _u_cdf_grid[0]
_I_maxw_cum  = np.zeros(50000)
_I_maxw_cum[1:] = np.cumsum(0.5 * (_f_maxw_grid[:-1] + _f_maxw_grid[1:]) * _du_cdf)
_I_maxw_full = _I_maxw_cum[-1]   # numerically: sqrt(pi)/4 ~ 0.44311
_B_maxw_cdf  = _I_maxw_cum / _I_maxw_full   # CDF table: B_Maxwell(X)


def plummer_vesc(r):
    """Escape speed from Plummer potential at radius r."""
    return math.sqrt(2.0 * G * M_tot / math.sqrt(r**2 + b**2)) if r >= 0 else 0.0


def B_Plummer(r, v_M=None):
    """
    Exact Plummer DF bracket: fraction of background particles slower than v_M.
    If v_M is None uses v_circ(r) (circular orbit).  Returns value in [0, 1].
    Uses precomputed CDF table — deterministic and fast.
    """
    if v_M is None:
        v_M = plummer_vcirc(r)
    v_e = plummer_vesc(r)
    if v_e < 1e-12 or v_M <= 0.0:
        return 0.0
    q_M = min(v_M / v_e, 1.0 - 1e-10)
    return float(np.interp(q_M, _q_cdf_grid, _B_cdf))


def lag_radius_theory(frac):
    lo, hi = 0.0, 1000.0 * b
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if mid**3 / (mid**2 + b**2) ** 1.5 < frac:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


LAG_THEORY = [lag_radius_theory(f) for f in LAG_FRACS]

# ---- Chandrasekhar friction functions ----
#
# Derivation document Section 3 gives the simplified Plummer form:
#
#   a_DF(R) = 3*(M_p+m)*lnL*B(X) / (R^2*(1+R^2))
#
# obtained by substituting rho(R) = 3/(4pi)*(1+R^2)^(-5/2) and
# v_circ^2(R) = R^2/(1+R^2)^(3/2) into 4pi*(M_p+m)*rho*lnL*B/v_circ^2.
# The Maxwellian bracket argument simplifies to X(R) = R*sqrt(3/(1+R^2)).
# Both forms are numerically identical; the document's form is used here
# because it directly reflects the derivation with no intermediate quantities.


def chandrasekhar_bracket(X):
    """
    Maxwellian Chandrasekhar bracket B(X): fraction of background particles
    slower than v_M, for a Maxwellian DF.  Evaluated via precomputed CDF
    table — mirrors B_Plummer approach.
    X = v_M / (sqrt(2)*sigma) = R*sqrt(3/(1+R^2)) for Plummer circular orbit.
    Analytically equivalent to erf(X) - (2X/sqrt(pi))*exp(-X^2);
    agreement verified to < 2e-9 across the full X range.
    """
    if X <= 0.0:
        return 0.0
    return float(np.interp(X, _u_cdf_grid, _B_maxw_cdf))


def chandrasekhar_decel(R, v_M, M_p_val, ln_lam_val, use_plummer_df=False):
    """
    DF deceleration magnitude at radius R, using the actual perturber speed v_M.

    [FIX 5] Exact Chandrasekhar formula, valid for any v_M (not just circular):
        a_DF = 4π G² (M_p+m) ρ(R) lnΛ B(v_M) / v_M²

    For the Plummer sphere (G = M_tot = b = 1):
        ρ(R) = 3/(4π) · (1 + R²)^{-5/2}

    so the formula simplifies to:
        a_DF = 3·(M_p+m)·lnΛ·B(v_M) / (v_M² · (1 + R²)^{5/2})

    The previous form  3·(M_p+m)·lnΛ·B / (R²·(1+R²))  is recovered only when
    v_M equals the circular speed v_circ = R·(1+R²)^{-3/4}.  Using v_circ² in
    the denominator instead of the actual v_M² introduced a spurious factor of
    v_circ²/v_M² into lnlam_eff, oscillating at the orbital frequency.

    Note: _chandra_ode_euler has its own hardcoded circular-orbit formula and
    does NOT call this function, so the secular inspiral ODE is unaffected.

    use_plummer_df=True  — exact Plummer DF bracket B_Plummer(R, v_M)
    use_plummer_df=False — Maxwellian bracket with X = v_M / (sqrt(2)·σ(R))
                          [FIX 5: X now uses actual v_M, not R·sqrt(3/(1+R²))]
    """
    if R <= 0.0 or v_M <= 0.0 or ln_lam_val <= 0.0:
        return 0.0
    if use_plummer_df:
        B = B_Plummer(R, v_M)
        if B <= 0.0:
            return 0.0
    else:
        # [FIX 5] Use actual v_M to compute X, not the circular-orbit shortcut
        # X = v_circ / (sqrt(2)*sigma) = R*sqrt(3/(1+R²)) only when v_M = v_circ.
        sig2 = G * M_tot / (6.0 * math.sqrt(R**2 + b**2))
        sigma = math.sqrt(max(sig2, 0.0))
        X = v_M / (math.sqrt(2.0) * sigma) if sigma > 1e-30 else 0.0
        B = chandrasekhar_bracket(X)
        if B <= 0.0:
            return 0.0
    # [FIX 5] ρ_Plummer(R) = 3/(4π)·(1+R²)^{-5/2}  →  4π·ρ/v_M² = 3/(v_M²·(1+R²)^{5/2})
    return 3.0 * M_p_val * ln_lam_val * B / (v_M**2 * (1.0 + R**2)**2.5)


def ln_lam_at_R(R, v_M=None):
    """
    R-dependent Coulomb logarithm.
    If v_M is given: ln(R * (v_M^2 + sigma^2(R)) / (G*M_p))  -- exact form.
    If v_M is None:  ln(M(<R) / M_p)  -- circular orbit approximation.
    Both floored at ln(1.0) = 0, which triggers the ln_lam <= 0 guard in
    chandrasekhar_decel and cleanly switches off friction when M(<R) <= M_p.
    """
    if v_M is not None:
        sig2 = G * M_tot / (6.0 * math.sqrt(R**2 + b**2))
        v_rel2 = v_M**2 + sig2
        return math.log(max(R * v_rel2 / (G * M_p), 1.0))
    return math.log(max(plummer_mass_enc(R) / M_p, 1.0))


# ---- helper functions ----


def compute_reduced_chi2(obs, exp, sigma, dof_adj=0):
    obs = np.asarray(obs, dtype=float)
    exp = np.asarray(exp, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    mask = (exp > 0) & (sigma > 0) & np.isfinite(obs) & np.isfinite(exp)
    n_eff = int(mask.sum())
    if n_eff <= dof_adj:
        return np.nan, np.nan
    chi2 = np.sum(((obs[mask] - exp[mask]) / sigma[mask]) ** 2)
    dof = n_eff - dof_adj
    return chi2 / dof, 1.0 - stats.chi2.cdf(chi2, dof)


def lagrangian_radii(r_sorted, N):
    return [r_sorted[max(0, int(f * N) - 1)] for f in LAG_FRACS]


def velocity_stats(pos, vel):
    pos_c = pos - np.mean(pos, axis=0)
    vel_c = vel - np.mean(vel, axis=0)
    r_mag = np.linalg.norm(pos_c, axis=1, keepdims=True)
    r_hat = pos_c / np.where(r_mag == 0, 1e-10, r_mag)
    vr = np.sum(vel_c * r_hat, axis=1)
    v2 = np.sum(vel_c**2, axis=1)
    return (
        math.sqrt(float(np.mean(v2))),
        float(np.std(vr)),
        float(np.std(np.sqrt(np.maximum(v2 - vr**2, 0.0)))),
        float(np.mean(vr)),
    )


def bin_density_profile(pos, n_bins=35):
    cm = np.mean(pos, axis=0)
    r = np.linalg.norm(pos - cm, axis=1)
    if len(r) < 5:
        return None, None, None
    bins = np.logspace(np.log10(max(r.min(), 1e-6)), np.log10(r.max()), n_bins)
    cnt, edges = np.histogram(r, bins=bins)
    r_mid = np.sqrt(edges[:-1] * edges[1:])
    vol = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    ok = cnt > 0
    return r_mid[ok], (cnt * m_bg / vol)[ok], cnt[ok]


def jeans_profile(pos, vel, n_bins=18):
    cm = np.mean(pos, axis=0)
    pos_c = pos - cm
    vel_c = vel - np.mean(vel, axis=0)
    r = np.linalg.norm(pos_c, axis=1)
    r_hat = pos_c / np.where(r[:, None] == 0, 1e-10, r[:, None])
    vr = np.sum(vel_c * r_hat, axis=1)
    vt2 = np.maximum(np.sum(vel_c**2, axis=1) - vr**2, 0.0)
    bins = np.logspace(-1.0, 1.0, n_bins + 1)
    r_mid = np.sqrt(bins[:-1] * bins[1:])
    sig_r = np.full(n_bins, np.nan)
    sig_t = np.full(n_bins, np.nan)
    ratio = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=float)
    for k in range(n_bins):
        mask = (r >= bins[k]) & (r < bins[k + 1])
        counts[k] = mask.sum()
        if counts[k] > 5:
            sr = np.std(vr[mask])
            st = math.sqrt(float(np.mean(vt2[mask])) / 2.0)
            sig_r[k] = sr
            sig_t[k] = st
            if sr > 0:
                ratio[k] = st / sr
    return r_mid, sig_r, sig_t, ratio, counts


def anisotropy_profile(pos, vel, n_bins=15):
    cm = np.mean(pos, axis=0)
    pos_c = pos - cm
    vel_c = vel - np.mean(vel, axis=0)
    r = np.linalg.norm(pos_c, axis=1)
    r_hat = pos_c / np.where(r[:, None] == 0, 1e-10, r[:, None])
    vr = np.sum(vel_c * r_hat, axis=1)
    vt2 = np.maximum(np.sum(vel_c**2, axis=1) - vr**2, 0.0)
    bins = np.logspace(np.log10(max(r.min(), 1e-6)), np.log10(r.max()), n_bins + 1)
    r_mid = np.sqrt(bins[:-1] * bins[1:])
    ratio = np.full(n_bins, np.nan)
    for k in range(n_bins):
        mask = (r >= bins[k]) & (r < bins[k + 1])
        if mask.sum() > 5:
            sr = np.std(vr[mask])
            st = math.sqrt(float(np.mean(vt2[mask])) / 2.0)
            if sr > 0:
                ratio[k] = st / sr
    ok = np.isfinite(ratio)
    return r_mid[ok], ratio[ok]


# ---- snapshot reader ----

import re as _re
_FLOAT_RE = _re.compile(r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?')

def _parse_floats(line):
    """
    Split a line into floats, handling Barnes treecode column-overflow.

    The treecode uses fixed-width output with no guaranteed separator between
    fields.  When a value is large (|x| >= 10) the sign of the next field can
    fuse with the last digit of the previous one, producing tokens like
    '3.33-2.1628068E-01' instead of '3.33 -2.1628068E-01'.

    Strategy: try the fast path (str.split + float()) first; fall back to
    regex tokenisation only when that raises ValueError.
    """
    tokens = line.split()
    try:
        return [float(t) for t in tokens]
    except ValueError:
        # Regex extracts every well-formed float sub-string, including the
        # fused ones that str.split left as a single malformed token.
        return [float(m) for m in _FLOAT_RE.findall(line)]


def iter_snapshots(filepath, N_tot):
    with open(filepath, "r") as fh:
        while True:
            line = fh.readline()
            if not line:
                return
            line = line.strip()
            if not line:
                continue
            try:
                n_snap = int(line)
            except ValueError:
                continue
            int(fh.readline())  # ndim
            t = float(fh.readline())
            if n_snap != N_tot:
                for _ in range(4 * n_snap):
                    fh.readline()
                continue
            for _ in range(N_tot):  # masses (discard)
                fh.readline()
            # Read all three blocks as raw lines first, then parse.
            # If any particle line yields != 3 floats (Barnes column-overflow
            # severe enough to merge two rows), the whole snapshot is corrupt —
            # skip it with a warning rather than crashing.
            try:
                pos_rows = [_parse_floats(fh.readline()) for _ in range(N_tot)]
                vel_rows = [_parse_floats(fh.readline()) for _ in range(N_tot)]
                phi_vals = [float(fh.readline())         for _ in range(N_tot)]
                if any(len(r) != 3 for r in pos_rows) or \
                   any(len(r) != 3 for r in vel_rows):
                    print(f"[WARN] t={t:.4f}: corrupt snapshot (column overflow), skipping",
                          flush=True)
                    continue
            except Exception as exc:
                print(f"[WARN] t={t:.4f}: parse error ({exc}), skipping", flush=True)
                continue
            pos = np.array(pos_rows)
            vel = np.array(vel_rows)
            phi = np.array(phi_vals)
            yield t, pos, vel, phi


# ============================================================
#  PASS 1 — MAIN STREAMING PASS
# ============================================================

times = []
R_M_arr = []
v_M_arr = []
Lz_M_arr = []
L_M_arr = []
Lz_over_L_arr = []  # [ADD M5]
E_orb_M_arr = []
v_r_arr = []        # radial velocity in live CM frame
v_t_arr = []        # tangential velocity in live CM frame
x_M_traj = []
y_M_traj = []
K_bg_arr = []
W_bg_arr = []
E_tot_arr = []
virial_arr = []
sigma_v_arr = []
sigma_vr_arr = []
sigma_vt_arr = []
mean_vr_arr = []
r_cm_bg_arr = []
lagr_arr = []

r_max_bg = np.zeros(N_BG)
r_min_bg = np.full(N_BG, np.inf)
r_sum_bg = np.zeros(N_BG)
n_orb = 0

pos_bg_initial = vel_bg_initial = phi_bg_initial = None
pos_bg_last = vel_bg_last = phi_bg_last = None

E0_total = None
snap_idx = 0

print("pass 1: streaming snapshots...", flush=True)

for t, pos, vel, phi in iter_snapshots(outfile, N_total):
    pos_p = pos[0]
    vel_p = vel[0]
    phi_p = phi[0]
    pos_bg = pos[1:]
    vel_bg = vel[1:]
    phi_bg = phi[1:]

    cm_bg = np.mean(pos_bg, axis=0)
    vcm_bg = np.mean(vel_bg, axis=0)
    dr_p = pos_p - cm_bg
    dv_p = vel_p - vcm_bg
    R_p = float(np.linalg.norm(dr_p))
    v_p_s = float(np.linalg.norm(dv_p))
    # Velocity decomposition: radial and tangential (same as test.py)
    v_r_p = float(np.dot(dv_p, dr_p) / R_p) if R_p > 1e-12 else 0.0
    v_t_p = float(np.sqrt(max(0.0, v_p_s**2 - v_r_p**2)))
    L_vec = np.cross(dr_p, dv_p)
    Lz_p = float(L_vec[2])
    L_p = float(np.linalg.norm(L_vec))
    E_orb_p = 0.5 * v_p_s**2 + phi_p

    v2_bg = np.sum(vel_bg**2, axis=1)
    K_bg = 0.5 * m_bg * float(np.sum(v2_bg))
    W_bg = 0.5 * m_bg * float(np.sum(phi_bg))
    K_pert = 0.5 * M_p * v_p_s**2
    E_tot = K_bg + K_pert + 0.5 * (M_p * phi_p + m_bg * float(np.sum(phi_bg)))
    if E0_total is None:
        E0_total = E_tot
    virial = -2.0 * K_bg / W_bg if abs(W_bg) > 1e-12 else 0.0

    sv, svr, svt, mvr = velocity_stats(pos_bg, vel_bg)

    r_bg = np.linalg.norm(pos_bg - cm_bg, axis=1)
    r_bg_sort = np.sort(r_bg)
    lagr = lagrangian_radii(r_bg_sort, N_BG)

    r_max_bg = np.maximum(r_max_bg, r_bg)
    r_min_bg = np.minimum(r_min_bg, r_bg)
    r_sum_bg += r_bg
    n_orb += 1

    times.append(t)
    R_M_arr.append(R_p)
    v_M_arr.append(v_p_s)
    Lz_M_arr.append(Lz_p)
    L_M_arr.append(L_p)
    Lz_over_L_arr.append(Lz_p / L_p if L_p > 1e-12 else np.nan)  # [ADD M5]
    E_orb_M_arr.append(E_orb_p)
    v_r_arr.append(v_r_p)
    v_t_arr.append(v_t_p)
    x_M_traj.append(float(pos_p[0]))
    y_M_traj.append(float(pos_p[1]))
    K_bg_arr.append(K_bg)
    W_bg_arr.append(W_bg)
    E_tot_arr.append(E_tot)
    virial_arr.append(virial)
    sigma_v_arr.append(sv)
    sigma_vr_arr.append(svr)
    sigma_vt_arr.append(svt)
    mean_vr_arr.append(mvr)
    r_cm_bg_arr.append(float(np.linalg.norm(cm_bg)))
    lagr_arr.append(lagr)

    if pos_bg_initial is None:
        pos_bg_initial = pos_bg.copy()
        vel_bg_initial = vel_bg.copy()
        phi_bg_initial = phi_bg.copy()
    pos_bg_last = pos_bg.copy()
    vel_bg_last = vel_bg.copy()
    phi_bg_last = phi_bg.copy()

    snap_idx += 1
    if snap_idx % 200 == 0:
        print(f"  {snap_idx} snapshots...", flush=True)

if snap_idx == 0:
    print("[ERROR] no valid snapshots read")
    sys.exit(1)

n_snaps = snap_idx
print(f"pass 1 done: {n_snaps} snapshots  t=0..{times[-1]:.2f}")

times = np.array(times)
R_M = np.array(R_M_arr)
v_M = np.array(v_M_arr)
Lz_M = np.array(Lz_M_arr)
L_M = np.array(L_M_arr)
Lz_over_L = np.array(Lz_over_L_arr)  # [ADD M5]
E_orb_M = np.array(E_orb_M_arr)
v_r_M = np.array(v_r_arr)
v_t_M = np.array(v_t_arr)
x_M_traj = np.array(x_M_traj)
y_M_traj = np.array(y_M_traj)
K_bg_arr = np.array(K_bg_arr)
W_bg_arr = np.array(W_bg_arr)
E_tot_arr = np.array(E_tot_arr)
virial_arr = np.array(virial_arr)
sigma_v_arr = np.array(sigma_v_arr)
sigma_vr_arr = np.array(sigma_vr_arr)
sigma_vt_arr = np.array(sigma_vt_arr)
mean_vr_arr = np.array(mean_vr_arr)
r_cm_bg_arr = np.array(r_cm_bg_arr)
lagr_arr = np.array(lagr_arr)

# ============================================================
#  DERIVED DIAGNOSTICS — PERTURBER
# ============================================================

dE_over_E0 = np.abs((E_tot_arr - E0_total) / E0_total)
DK_bg = K_bg_arr - K_bg_arr[0]

# Sort all arrays by time — skipped corrupt snapshots can leave the times
# array non-monotonic, which makes any line plot of a time series zigzag.
_sort = np.argsort(times)
if not np.all(_sort == np.arange(len(_sort))):
    print(f"[WARN] times not monotonic — resorting {len(_sort)} snapshots", flush=True)
    times        = times[_sort]
    R_M          = R_M[_sort]
    v_M          = v_M[_sort]
    Lz_M         = Lz_M[_sort]
    L_M          = L_M[_sort]
    Lz_over_L    = Lz_over_L[_sort]
    E_orb_M      = E_orb_M[_sort]
    v_r_M        = v_r_M[_sort]
    v_t_M        = v_t_M[_sort]
    x_M_traj     = x_M_traj[_sort]
    y_M_traj     = y_M_traj[_sort]
    K_bg_arr     = K_bg_arr[_sort]
    W_bg_arr     = W_bg_arr[_sort]
    E_tot_arr    = E_tot_arr[_sort]
    virial_arr   = virial_arr[_sort]
    sigma_v_arr  = sigma_v_arr[_sort]
    sigma_vr_arr = sigma_vr_arr[_sort]
    sigma_vt_arr = sigma_vt_arr[_sort]
    mean_vr_arr  = mean_vr_arr[_sort]
    r_cm_bg_arr  = r_cm_bg_arr[_sort]
    lagr_arr     = lagr_arr[_sort]
    dE_over_E0   = dE_over_E0[_sort]
    DK_bg        = DK_bg[_sort]

dt_snap = float(np.median(np.diff(times))) if n_snaps > 1 else 1.0
dR_dt = np.gradient(R_M, dt_snap)

slope_R, *_ = stats.linregress(times, R_M)
slope_L, *_ = stats.linregress(times, L_M)

# R-dependent Coulomb log along the actual trajectory
ln_lam_R_arr = np.array([ln_lam_at_R(R_M[i], v_M[i]) for i in range(n_snaps)])


# [ADD M4 / FIX 3] Plummer DF bracket and Maxwellian bracket along trajectory.
# B_Plummer is the exact bracket from the Eddington DF.
# B_Maxwell (Maxwellian erf-form) is kept for direct comparison.
def _compute_B_X():
    B_plum_arr = np.empty(n_snaps)
    B_maxw_arr = np.empty(n_snaps)
    X_arr_ = np.empty(n_snaps)
    for i in range(n_snaps):
        R = R_M[i]
        vM = v_M[i]
        B_plum_arr[i] = B_Plummer(R, vM)
        sig2 = G * M_tot / (6.0 * math.sqrt(R**2 + b**2))
        sigma = math.sqrt(sig2)
        X = vM / (math.sqrt(2.0) * sigma) if sigma > 0 else 0.0
        X_arr_[i] = X
        B_maxw_arr[i] = chandrasekhar_bracket(X) if X > 0 else 0.0
    return X_arr_, B_plum_arr, B_maxw_arr


X_arr, B_X_arr, B_X_Maxwell_arr = _compute_B_X()  # [ADD M4, FIX 3]

# Measured deceleration: -dL/dt * |v| / |L|
# Smooth L first to remove N-body noise, then differentiate.
L_M_smooth = gaussian_filter1d(L_M, sigma=4.0)
dL_dt = np.gradient(L_M_smooth, dt_snap)
with np.errstate(divide="ignore", invalid="ignore"):
    # Corrects the v_tang/|v| bias for non-circular orbits
    a_meas = np.where(
        (R_M > 0.05) & (L_M_smooth > 0), -dL_dt * v_M / L_M_smooth, np.nan
    )

# Theory deceleration: exact Plummer DF bracket, R-dependent ln_Lambda [FIX 3]
# Mass = (M_p + m_bg) per Chandrasekhar derivation (Section 3 of document).
a_chandra_Rdep = np.array(
    [
        chandrasekhar_decel(
            R_M[i], v_M[i], M_p + m_bg, ln_lam_at_R(R_M[i], v_M[i]), use_plummer_df=True
        )
        for i in range(n_snaps)
    ]
)
# Maxwellian theory deceleration (comparison)
a_chandra_Rdep_Maxwell = np.array(
    [
        chandrasekhar_decel(
            R_M[i],
            v_M[i],
            M_p + m_bg,
            ln_lam_at_R(R_M[i], v_M[i]),
            use_plummer_df=False,
        )
        for i in range(n_snaps)
    ]
)

# Effective Coulomb log: a_meas / a_DF(ln=1, Plummer bracket)
a_chandra_lam1 = np.array(
    [
        chandrasekhar_decel(R_M[i], v_M[i], M_p + m_bg, 1.0, use_plummer_df=True)
        for i in range(n_snaps)
    ]
)
with np.errstate(divide="ignore", invalid="ignore"):
    lnlam_eff = np.where(
        (a_chandra_lam1 > 1e-12) & np.isfinite(a_meas), a_meas / a_chandra_lam1, np.nan
    )


# Chandrasekhar ODE: Euler method with dt=0.01                           [FIX 4]
#
# The ODE is  dR/dt = f(R) = -a_DF(R)*R / dLdR(R).
# f(R) has no closed-form antiderivative (B(X(R)) involves erf),
# so we integrate numerically.
#
# Why Euler and not solve_ivp?
# The derivation document (Section 10) shows that Euler is accurate here
# because f(R) is smooth and slowly varying.  The minimum orbital period
# is T_orb_min = 2*pi (at r->0), so dt=0.01 gives ~628 steps per orbit —
# well within the accurate regime.  At dt=0.01 the error in R at any
# snapshot time is < 1e-4, negligible against N-body noise.
# solve_ivp adds complexity for no measurable gain.
#
# The N-body trajectory R_M(t) is NEVER used after the initial condition R_0;
# the two curves (theory and simulation) evolve independently.

# R_stall: radius where M_enc(R) = M_p → lnΛ = 0 → Chandrasekhar friction off.
# Physical boundary of formula validity; solved by bisection.
_lo_s, _hi_s = 1e-4, R_M[0]
for _ in range(100):
    _mid_s = 0.5 * (_lo_s + _hi_s)
    if plummer_mass_enc(_mid_s) < M_p:
        _lo_s = _mid_s
    else:
        _hi_s = _mid_s
R_stall = 0.5 * (_lo_s + _hi_s)

_DT_FINE = 0.01  # fixed fine timestep for Euler integration


def _chandra_ode_euler(use_plummer_df, lnL_floor=0.0):
    """
    Euler integration of dR/dt = f(R) with dt=_DT_FINE.
    Outputs R at the snapshot times by linear interpolation.
    lnL_floor=0.0  (default): stops at R_stall where lnL=0 — use for t_DF.
    lnL_floor>0.0           : continues below R_stall — use for plot curves only.
    """
    t_fine = np.arange(times[0], times[-1] + _DT_FINE, _DT_FINE)
    R_fine = np.empty(len(t_fine))
    R_cur = R_M[0]
    R_fine[0] = R_cur

    for i in range(1, len(t_fine)):
        if R_cur < 0.01:
            R_fine[i] = R_cur
            continue
        if use_plummer_df:
            B = B_Plummer(R_cur)
        else:
            X = R_cur * math.sqrt(3.0 / (1.0 + R_cur**2))
            B = chandrasekhar_bracket(X)
        ln_lam = max(ln_lam_at_R(R_cur), lnL_floor)
        if ln_lam <= 0.0:
            R_fine[i] = R_cur
            continue
        dR = (
            -6.0
            * (M_p + m_bg)
            * ln_lam
            * B
            / R_cur**2
            * (1.0 + R_cur**2) ** 0.75
            / (4.0 + R_cur**2)
        )
        R_cur = max(0.01, R_cur + dR * _DT_FINE)
        R_fine[i] = R_cur

    return np.interp(times, t_fine, R_fine)


# floor=0.0: stops at R_stall — used for t_DF and stats
R_chandra_Rdep        = _chandra_ode_euler(use_plummer_df=True)
R_chandra_Rdep_Maxwell = _chandra_ode_euler(use_plummer_df=False)

# floor=ln(1.1): continues below R_stall — used for plot only (extrapolation)
_lnL_plot_floor = math.log(1.1)
R_chandra_Rdep_plot        = _chandra_ode_euler(use_plummer_df=True,  lnL_floor=_lnL_plot_floor)
R_chandra_Rdep_Maxwell_plot = _chandra_ode_euler(use_plummer_df=False, lnL_floor=_lnL_plot_floor)

# ============================================================
#  DERIVED DIAGNOSTICS — BACKGROUND
# ============================================================

r_hm_t = lagr_arr[:, 2]
r_hm_initial = float(r_hm_t[0])
r_hm_median = float(np.median(r_hm_t))
r_hm_offset = (r_hm_median - r_hm) / r_hm
virial_mean = float(np.mean(virial_arr))
virial_std = float(np.std(virial_arr))
sv_drift = (float(sigma_v_arr[-1]) - float(sigma_v_arr[0])) / float(sigma_v_arr[0])
energy_err = float(np.max(dE_over_E0))
is_good = energy_err < 0.01

print(f"R_M(0)={R_M[0]:.4f}  R_M(f)={R_M[-1]:.4f}  dR/dt={slope_R:.4e}")
print(
    f"max|dE/E0|={energy_err:.2e}  virial={virial_mean:.4f}  r_hm_offset={100 * r_hm_offset:+.2f}%"
)

# ============================================================
#  PASS 2 — MID SNAPSHOT
# ============================================================

t_mid_target = times[-1] / 2.0
pos_bg_mid = vel_bg_mid = phi_bg_mid = None
t_mid_actual = np.inf
best_mid_diff = np.inf

print("pass 2: mid snapshot...", flush=True)

for t, pos, vel, phi in iter_snapshots(outfile, N_total):
    diff = abs(t - t_mid_target)
    if diff < best_mid_diff:
        best_mid_diff = diff
        t_mid_actual = t
        pos_bg_mid = pos[1:].copy()
        vel_bg_mid = vel[1:].copy()
        phi_bg_mid = phi[1:].copy()

print(f"pass 2 done.  mid snapshot at t={t_mid_actual:.4f}")

# ============================================================
#  GOODNESS-OF-FIT TESTS  (final background snapshot)
# ============================================================

cm_last = np.mean(pos_bg_last, axis=0)
r_last = np.linalg.norm(pos_bg_last - cm_last, axis=1)

# 1. density chi2
bins_chi = np.logspace(np.log10(2.0 * EPS), 1.5, 25)
cnt_chi, edges_chi = np.histogram(r_last, bins=bins_chi)
r_mid_chi = np.sqrt(edges_chi[:-1] * edges_chi[1:])
vol_chi = (4.0 / 3.0) * np.pi * (edges_chi[1:] ** 3 - edges_chi[:-1] ** 3)
rho_sim_chi = cnt_chi * m_bg / vol_chi
rho_th_chi = np.array([plummer_rho(r) for r in r_mid_chi])
cnt_th_chi = rho_th_chi * vol_chi / m_bg
chi2_rho, p_rho = compute_reduced_chi2(
    rho_sim_chi, rho_th_chi, rho_th_chi / np.sqrt(np.maximum(cnt_th_chi, 1))
)

# 2. KS test
ks_stat, p_ks = stats.kstest(r_last, plummer_cdf)

# 3. P(q) chi2
if phi_bg_last is not None:
    v_mag_chi = np.linalg.norm(vel_bg_last, axis=1)
    v_esc_chi = np.sqrt(np.maximum(-2.0 * phi_bg_last, 0.0))
    valid_chi = (v_esc_chi > 0) & (
        v_mag_chi / np.where(v_esc_chi > 0, v_esc_chi, 1.0) < 0.98
    )
    q_sim_chi = (v_mag_chi / v_esc_chi)[valid_chi]
    qc, qe = np.histogram(q_sim_chi, bins=25)
    qm = 0.5 * (qe[:-1] + qe[1:])
    q_th_chi = qm**2 * (1.0 - qm**2) ** 3.5
    q_th_chi *= len(q_sim_chi) / np.sum(q_th_chi)
    chi2_q, p_q = compute_reduced_chi2(
        qc, q_th_chi, np.sqrt(np.maximum(q_th_chi, 1)), dof_adj=1
    )
    has_q = True
else:
    chi2_q = p_q = np.nan
    has_q = False

# 4. Jeans chi2
r_mid_jchi, sigr_chi, sigt_chi, _, cnt_jchi = jeans_profile(pos_bg_last, vel_bg_last)
sig_th_jchi = sigma_theory_jeans(r_mid_jchi)
sig_err_jchi = sig_th_jchi / np.sqrt(2.0 * np.maximum(cnt_jchi - 1, 1))
chi2_sigr, p_sigr = compute_reduced_chi2(sigr_chi, sig_th_jchi, sig_err_jchi)
chi2_sigt, p_sigt = compute_reduced_chi2(sigt_chi, sig_th_jchi, sig_err_jchi)

# ============================================================
#  BACKGROUND FIGURES
# ============================================================

# BG3: density profile initial / mid / final  +  fractional residuals (bottom row)
r_theory = np.logspace(-1.5, 1.5, 300)
rho_th_plot = np.array([plummer_rho(r) for r in r_theory])

fig, axes = plt.subplots(2, 3, figsize=(14, 9),
                         gridspec_kw={"height_ratios": [3, 1.8]})
for col, (pos_s, label) in enumerate([
    (pos_bg_initial,    "t=0"),
    (pos_bg_mid,        f"t={t_mid_actual:.2f}"),
    (pos_bg_last,       f"t={times[-1]:.2f}"),
]):
    ax_top = axes[0, col]
    ax_bot = axes[1, col]
    r_m, rho_m, cnts = bin_density_profile(pos_s)

    # ---- top: absolute density ----
    if r_m is not None:
        log_sig = 1.0 / np.sqrt(cnts)
        ax_top.errorbar(r_m, rho_m,
                        yerr=[rho_m*(1.0 - 10.0**(-log_sig)),
                              rho_m*(10.0**log_sig - 1.0)],
                        fmt="o", ms=3, color="k", label="N-body")
    ax_top.plot(r_theory, rho_th_plot, "r--", label="Plummer theory")
    ax_top.set_xscale("log")
    ax_top.set_yscale("log")
    ax_top.set_xlabel("r")
    ax_top.set_title(label)
    ax_top.legend(fontsize=7)
    if col == 0:
        ax_top.set_ylabel("density")

    # ---- bottom: fractional residual (rho - rho_theory) / rho_theory ----
    ax_bot.axhline(0.0, color="r", ls="--", lw=0.8, label="Plummer")
    if r_m is not None:
        rho_th_at_rm = np.array([plummer_rho(r) for r in r_m])
        valid = rho_th_at_rm > 0
        resid = np.where(valid, (rho_m - rho_th_at_rm) / rho_th_at_rm, np.nan)
        err   = np.where(valid, 1.0 / np.sqrt(np.maximum(cnts, 1)), np.nan)
        ax_bot.errorbar(r_m, resid, yerr=err, fmt="o", ms=3, color="k")
        # shade Poisson noise floor
        poisson_floor = np.nanmean(err)
        ax_bot.axhspan(-poisson_floor, poisson_floor,
                       color="0.85", alpha=0.5, label=f"±1/√N  (~{poisson_floor:.2f})")
    ax_bot.axhline(0.0, color="r", ls="--", lw=0.8)
    ax_bot.set_xscale("log")
    ax_bot.set_xlabel("r")
    ax_bot.set_ylim(-0.5, 0.5)
    ax_bot.legend(fontsize=7)
    if col == 0:
        ax_bot.set_ylabel(r"$(\rho - \rho_\mathrm{th})\,/\,\rho_\mathrm{th}$")

plt.tight_layout()
fname = os.path.join(run_dir, "plot_bg_density.pdf")
fig.savefig(fname, bbox_inches="tight")
plt.close(fig)
print(f"written: {os.path.basename(fname)}")

# BG4: cumulative mass and circular velocity
r_th_vc = np.logspace(-2, 1.5, 500)
M_th_vc = np.array([plummer_mass_enc(r) for r in r_th_vc])
vc_th = np.sqrt(G * M_th_vc / r_th_vc)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for col, (pos_s, label) in enumerate(
    [(pos_bg_initial, "t=0"), (pos_bg_last, f"t={times[-1]:.2f}")]
):
    cm_s = np.mean(pos_s, axis=0)
    r_s = np.sort(np.linalg.norm(pos_s - cm_s, axis=1))
    M_cum = np.arange(1, N_BG + 1) * m_bg
    mask_vc = r_s > 0.01
    r_vc = r_s[mask_vc]
    vcirc_s = np.sqrt(G * M_cum[mask_vc] / r_vc)
    axes[0, col].plot(r_th_vc, vc_th, "r--", label="Plummer theory")
    axes[0, col].plot(r_vc, vcirc_s, color="k", lw=0.8, label="N-body")
    axes[0, col].set_title(f"v_circ  {label}")
    axes[0, col].set_xlabel("r")
    axes[0, col].set_xscale("log")
    axes[0, col].set_xlim(r_th_vc[0], r_th_vc[-1])
    axes[0, col].legend(fontsize=7)
    axes[1, col].plot(r_th_vc, M_th_vc, "r--", label="theory")
    axes[1, col].plot(r_vc, M_cum[mask_vc], color="k", lw=0.8, label="N-body")
    axes[1, col].axhline(M_tot, color="k", ls=":", lw=0.8, label="M_tot")
    axes[1, col].set_title(f"M(<r)  {label}")
    axes[1, col].set_xlabel("r")
    axes[1, col].set_xscale("log")
    axes[1, col].set_xlim(r_th_vc[0], r_th_vc[-1])
    axes[1, col].legend(fontsize=7)
axes[0, 0].set_ylabel("v_circ")
axes[1, 0].set_ylabel("M(<r)")
plt.tight_layout()
fname = os.path.join(run_dir, "plot_bg_mass_vcirc.pdf")
fig.savefig(fname, bbox_inches="tight")
plt.close(fig)
print(f"written: {os.path.basename(fname)}")

# BG5: velocity distribution P(q)
if phi_bg_initial is not None and phi_bg_last is not None:
    q_theory = np.linspace(0, 1, 300)
    p_theory = q_theory**2 * (1.0 - q_theory**2) ** 3.5
    p_theory /= np.trapezoid(p_theory, q_theory)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, label, vel_s, phi_s in [
        (axes[0], "t=0", vel_bg_initial, phi_bg_initial),
        (axes[1], f"t={times[-1]:.2f}", vel_bg_last, phi_bg_last),
    ]:
        v_mag = np.linalg.norm(vel_s, axis=1)
        v_esc = np.sqrt(np.maximum(-2.0 * phi_s, 0.0))
        q_sim = np.where(v_esc > 0, v_mag / v_esc, 1.0)
        q_sim = q_sim[q_sim < 1.0]
        ax.hist(q_sim, bins=40, density=True, color="k", alpha=0.6, label="N-body")
        ax.plot(q_theory, p_theory, "r-", label="q^2(1-q^2)^(7/2)")
        ax.set_xlabel("q = v / v_esc")
        ax.set_ylabel("pdf")
        ax.set_title(label)
        ax.legend(fontsize=7)
    plt.tight_layout()
    fname = os.path.join(run_dir, "plot_bg_vel_dist.pdf")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"written: {os.path.basename(fname)}")

# BG6: Jeans equation
fig, axes = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})
for col, (pos_s, vel_s, label) in enumerate(
    [
        (pos_bg_initial, vel_bg_initial, "t=0"),
        (pos_bg_last, vel_bg_last, f"t={times[-1]:.2f}"),
    ]
):
    ax_top = axes[0, col]
    ax_bot = axes[1, col]
    r_mid_j, sig_r_j, sig_t_j, ratio_j, cnt_j = jeans_profile(pos_s, vel_s)
    valid = ~np.isnan(sig_r_j)
    if valid.any():
        r_th_j = np.logspace(
            np.log10(r_mid_j[valid].min()), np.log10(r_mid_j[valid].max()), 200
        )
        ax_top.plot(r_th_j, sigma_theory_jeans(r_th_j), "r--", label="Jeans theory")
    sig_err_j = sigma_theory_jeans(r_mid_j) / np.sqrt(2.0 * np.maximum(cnt_j - 1, 1))
    ax_top.errorbar(
        r_mid_j, sig_r_j, yerr=sig_err_j, fmt="o", ms=3, color="k", label="sigma_r"
    )
    ax_top.errorbar(
        r_mid_j, sig_t_j, yerr=sig_err_j, fmt="s", ms=3, color="0.5", label="sigma_t"
    )
    ax_top.set_xscale("log")
    ax_top.set_ylabel("sigma")
    ax_top.set_title(label)
    ax_top.legend(fontsize=7)
    ax_bot.axhline(1.0, color="r", ls="--", lw=0.8, label="sigma_t/sigma_r = 1")
    ax_bot.scatter(r_mid_j, ratio_j, s=15, color="k")
    ax_bot.set_xscale("log")
    ax_bot.set_ylim(0.5, 1.5)
    ax_bot.set_xlabel("r")
    ax_bot.set_ylabel("sigma_t / sigma_r")
    ax_bot.legend(fontsize=7)
plt.tight_layout()
fname = os.path.join(run_dir, "plot_bg_jeans.pdf")
fig.savefig(fname, bbox_inches="tight")
plt.close(fig)
print(f"written: {os.path.basename(fname)}")

# BG7: Lagrangian radii vs time
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for k, frac in enumerate(LAG_FRACS):
    label = f"r_{int(frac * 100)}%  (th={LAG_THEORY[k]:.3f})"
    axes[0].plot(times, lagr_arr[:, k], lw=1.0, label=label)
    axes[0].axhline(LAG_THEORY[k], ls="--", lw=0.6, alpha=0.5)
    pct_dev = (lagr_arr[:, k] / lagr_arr[0, k] - 1.0) * 100.0
    axes[1].plot(times, pct_dev, lw=1.0, label=f"r_{int(frac * 100)}%")
axes[0].axhline(r_hm, color="k", ls=":", lw=0.8)
axes[0].set_xlabel("t")
axes[0].set_ylabel("Lagrangian radius")
axes[0].set_title("absolute")
axes[0].legend(fontsize=7)
axes[1].axhline(0.0, color="k", ls="--", lw=0.8, label="0% (no change)")
axes[1].set_xlabel("t")
axes[1].set_ylabel("deviation from t=0  (%)")
axes[1].set_title("fractional change  [(r(t)/r(0) − 1) × 100%]")
axes[1].legend(fontsize=7)
plt.tight_layout()
fname = os.path.join(run_dir, "plot_bg_lagrangian.pdf")
fig.savefig(fname, bbox_inches="tight")
plt.close(fig)
print(f"written: {os.path.basename(fname)}")

# BG8: velocity diagnostics vs time
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
axes[0].plot(times, virial_arr, color="k", lw=1.0, label="2K/|W|")
axes[0].axhline(1.0, color="k", ls="--", lw=0.8, label="= 1")
axes[0].axhline(virial_mean, color="k", ls=":", lw=0.8, label=f"mean={virial_mean:.4f}")
axes[0].set_ylim(0.8, 1.2)
axes[0].set_xlabel("t")
axes[0].set_ylabel("2K / |W|")
axes[0].set_title("virial ratio")
axes[0].legend(fontsize=7)
axes[1].plot(times, sigma_v_arr, color="k", lw=1.0, label="sigma_v total")
axes[1].plot(times, sigma_vr_arr, color="0.4", lw=0.8, ls="--", label="sigma_vr")
axes[1].plot(times, sigma_vt_arr, color="0.6", lw=0.8, ls=":", label="sigma_vt")
axes[1].axhline(sv_theory, color="r", ls="--", lw=0.8, label=f"theory={sv_theory:.4f}")
axes[1].set_xlabel("t")
axes[1].set_ylabel("velocity dispersion")
axes[1].set_title("velocity dispersion")
axes[1].legend(fontsize=7)
axes[2].plot(times, mean_vr_arr, color="k", lw=1.0, label="<v_r>")
axes[2].axhline(0.0, color="k", ls="--", lw=0.8, label="= 0")
axes[2].set_xlabel("t")
axes[2].set_ylabel("<v_r>")
axes[2].set_title("mean radial velocity")
axes[2].legend(fontsize=7)
plt.tight_layout()
fname = os.path.join(run_dir, "plot_bg_velocities.pdf")
fig.savefig(fname, bbox_inches="tight")
plt.close(fig)
print(f"written: {os.path.basename(fname)}")

# BG9: energy distribution bound/unbound (final snapshot)
if phi_bg_last is not None:
    E_final = 0.5 * np.sum(vel_bg_last**2, axis=1) + phi_bg_last
    n_unbound = int(np.sum(E_final > 0))
    n_bound = N_BG - n_unbound
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(
        E_final[E_final <= 0], bins=50, color="k", alpha=0.6, label=f"bound ({n_bound})"
    )
    ax.hist(
        E_final[E_final > 0],
        bins=50,
        color="0.5",
        alpha=0.6,
        label=f"unbound ({n_unbound})",
    )
    ax.axvline(0, color="k", ls="--", lw=0.8, label="E=0")
    ax.set_xlabel("specific energy  (0.5*v^2 + phi)")
    ax.set_ylabel("N particles")
    ax.set_title(
        f"bound vs unbound  t={times[-1]:.2f}  ({100 * n_unbound / N_BG:.2f}% unbound)"
    )
    ax.legend(fontsize=7)
    plt.tight_layout()
    fname = os.path.join(run_dir, "plot_bg_energy.pdf")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"written: {os.path.basename(fname)}")

# ============================================================
#  PERTURBER FIGURES
# ============================================================

# Shared arrays used across multiple perturber figures.
# Compute once here to avoid repeated loops.
vc_theory_arr = np.array([plummer_vcirc(r) for r in R_M])
J_circ_arr    = R_M * vc_theory_arr
with np.errstate(divide="ignore", invalid="ignore"):
    circularity_raw = np.where(J_circ_arr > 1e-12, L_M / J_circ_arr, np.nan)
circularity = np.where(np.isfinite(circularity_raw), circularity_raw, np.nan)

# P_RADIUS: orbital radius decay and centre-of-mass reflex motion
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

ax = axes[0]
ax.plot(times, R_M, color="k", lw=0.8, label="R_M(t)  N-body")

# Split each theory curve at R_stall: solid where valid, dotted extrapolation below.
for R_plot, color, label in [
    (R_chandra_Rdep_plot,         "b", "Chandra — Plummer DF"),
    (R_chandra_Rdep_Maxwell_plot, "r", "Chandra — Maxwell"),
]:
    i_stall = next((i for i, r in enumerate(R_plot) if r <= R_stall + 1e-4), len(times) - 1)
    ax.plot(times[:i_stall+1], R_plot[:i_stall+1], color=color, lw=1.5, ls="-.",
            label=label)
    if i_stall < len(times) - 1:
        ax.plot(times[i_stall:], R_plot[i_stall:], color=color, lw=0.9, ls=":",
                alpha=0.55)

ax.axhline(R_stall, color="0.5", ls="--", lw=0.8,
           label=rf"$R_\mathrm{{stall}}={R_stall:.2f}$  (lnΛ=0)")
ax.axhline(r_hm, color="k", ls=":", lw=0.8, label="r_hm")
ax.set_xlabel("t")
ax.set_ylabel("R_M")
ax.set_title("orbital radius decay")
ax.legend(fontsize=7)

ax = axes[1]
ax.plot(times, r_cm_bg_arr, color="m", lw=0.8)
ax.set_xlabel("t")
ax.set_ylabel(r"$|R_\mathrm{cm}|$")
ax.set_title("reflex motion of host centre of mass")

plt.tight_layout()
fname = os.path.join(run_dir, "plot_perturber_radius.pdf")
fig.savefig(fname, bbox_inches="tight")
plt.close(fig)
print(f"written: {os.path.basename(fname)}")

# P_MOMENTA: angular momentum loss, circularity, orbital plane stability
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
ax.plot(times, L_M, color="k", lw=0.8, label="|L_M|")
ax.plot(times, Lz_M, color="0.5", lw=0.6, alpha=0.6, label="L_z")
ax.set_xlabel("t")
ax.set_ylabel("specific angular momentum")
ax.set_title("angular momentum loss")
ax.legend(fontsize=7)

ax = axes[1]
ax.plot(times, circularity, color="k", lw=0.8)
ax.axhline(1.0, color="b", ls="--", lw=0.8, label=r"$\eta=1$ (circular)")
ax.set_xlabel("t")
ax.set_ylabel(r"$\eta = |L|\,/\,(R\,v_c(R))$")
ax.set_title("circularity  (1 = perfectly circular orbit)")
ax.legend(fontsize=7)

ax = axes[2]
ax.plot(times, Lz_over_L, color="k", lw=0.8, label=r"$L_z\,/\,|L|$")
ax.axhline(1.0, color="b", ls=":", lw=0.8, label="1.0 (initial value)")
ax.axhline(0.95, color="0.5", ls="--", lw=0.7, label="0.95 (5% threshold)")
ax.set_xlabel("t")
ax.set_ylabel(r"$L_z\,/\,|L|$")
ax.set_title("orbital plane stability")
ax.set_ylim(
    min(float(np.nanmin(Lz_over_L)) * 0.95, 0.85),
    1.05,
)
ax.legend(fontsize=7)

plt.tight_layout()
fname = os.path.join(run_dir, "plot_perturber_momenta.pdf")
fig.savefig(fname, bbox_inches="tight")
plt.close(fig)
print(f"written: {os.path.basename(fname)}")

# P_VELOCITY: tangential, radial, and circular velocity along orbit
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(times, v_t_M, color="k",   lw=0.8, label=r"$v_t$ (tangential)")
ax.plot(times, v_r_M, color="0.5", lw=0.8, ls="--", label=r"$v_r$ (radial)")
ax.plot(times, vc_theory_arr, color="r", lw=1.0, ls=":", label=r"$v_c(R)$ theory")
ax.set_xlabel("t")
ax.set_ylabel("velocity")
ax.set_title(r"velocity decomposition  $v_t$ and $v_r$")
ax.legend(fontsize=7)
plt.tight_layout()
fname = os.path.join(run_dir, "plot_perturber_velocity.pdf")
fig.savefig(fname, bbox_inches="tight")
plt.close(fig)
print(f"written: {os.path.basename(fname)}")

# P_FRICTION: deceleration and DF bracket ratio
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
a_Rdep_plot = np.where(a_chandra_Rdep > 0, a_chandra_Rdep, np.nan)
a_Rdep_plot_Maxwell = np.where(
    a_chandra_Rdep_Maxwell > 0, a_chandra_Rdep_Maxwell, np.nan
)
ax.plot(
    times, a_meas, color="k", lw=0.8, alpha=0.7, label=r"$-\dot{L}|v|/|L|$"
)
ax.plot(
    times,
    a_Rdep_plot,
    color="b",
    lw=1.5,
    ls="-.",
    label="Chandra — Plummer DF",
)
ax.plot(
    times,
    a_Rdep_plot_Maxwell,
    color="r",
    lw=1.0,
    ls=":",
    label="Chandra — Maxwell",
)
valid_pos = np.concatenate(
    [
        a_meas[np.isfinite(a_meas) & (a_meas > 0)]
        if np.any(a_meas > 0)
        else np.array([]),
        a_Rdep_plot[np.isfinite(a_Rdep_plot)],
    ]
)
if len(valid_pos):
    ax.set_ylim(valid_pos.min() * 0.3, valid_pos.max() * 3.0)
ax.set_xlabel("t")
ax.set_ylabel("deceleration")
ax.set_title("friction deceleration  (-dL/dt |v|/|L|)")
ax.set_yscale("log")
ax.legend(fontsize=7)

ax = axes[1]
with np.errstate(divide="ignore", invalid="ignore"):
    B_ratio = np.where(B_X_Maxwell_arr > 1e-6, B_X_arr / B_X_Maxwell_arr, np.nan)
ax.plot(times, B_ratio, color="b", lw=1.2, label=r"$\Gamma_\mathrm{Plummer}\,/\,\Gamma_\mathrm{Maxwell}$")
ax.axhline(1.0, color="0.4", ls="--", lw=0.8, label="ratio = 1")
ax.set_xlabel("t")
ax.set_ylabel(r"$\Gamma_\mathrm{Plummer}\,/\,\Gamma_\mathrm{Maxwell}$")
ax.set_title("DF bracket ratio  (Plummer / Maxwell)")
ax.legend(fontsize=7)

# Panel 3: bracket CDFs as functions of v / v_c(R_0).
# Common x-axis = speed normalised to the initial circular velocity,
# so both curves pass through their bracket value at x=1 (the perturber speed).
# Plummer CDF ends at v_esc(R_0)/v_c(R_0) = 1/q_M_0  (finite support).
# Maxwell CDF continues past that — infinite support, shown up to x=3.
ax = axes[2]
v_esc_0 = plummer_vesc(R_M[0])
q_M_0   = min(v_M[0] / v_esc_0, 1.0 - 1e-10) if v_esc_0 > 1e-12 else 0.5
X_0     = X_arr[0]
# Plummer: q -> v/v_c = q / q_M_0  (curve ends at v/v_c = 1/q_M_0)
x_plum = _q_cdf_grid / q_M_0
ax.plot(x_plum, _B_cdf, color="b", lw=1.5,
        label=r"Plummer  $B(q)$,  $q=v/v_\mathrm{esc}$")
# Maxwell: X -> v/v_c = X / X_0  (truncated at x=3 for visibility)
x_maxw = _u_cdf_grid / X_0
mask   = x_maxw <= 3.0
ax.plot(x_maxw[mask], _B_maxw_cdf[mask], color="r", lw=1.5, ls="--",
        label=r"Maxwell  $B(X)$,  $X=v/\sqrt{2}\,\sigma$")
# mark v = v_c (the circular-orbit speed at t=0, x=1)
B_P_vc = float(np.interp(q_M_0, _q_cdf_grid, _B_cdf))
B_M_vc = float(np.interp(X_0,   _u_cdf_grid, _B_maxw_cdf))
ax.axvline(1.0, color="0.4", ls=":", lw=0.8,
           label=r"$v=v_c(R_0)$  (perturber at $t=0$)")
ax.scatter([1.0, 1.0], [B_P_vc, B_M_vc], color=["b", "r"], s=40, zorder=5)
# mark v_esc boundary: Plummer CDF ends here
v_esc_norm = 1.0 / q_M_0
ax.axvline(v_esc_norm, color="b", ls="--", lw=0.7, alpha=0.5,
           label=rf"$v_\mathrm{{esc}}(R_0)$  (Plummer support ends,  $\times${v_esc_norm:.2f}$v_c$)")
ax.set_xlabel(r"$v\,/\,v_c(R_0)$")
ax.set_ylabel("B  (fraction of slower particles)")
ax.set_title("bracket CDFs: Plummer vs Maxwell")
ax.set_xlim(0, 3.0)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=7)

plt.tight_layout()
fname = os.path.join(run_dir, "plot_perturber_friction.pdf")
fig.savefig(fname, bbox_inches="tight")
plt.close(fig)
print(f"written: {os.path.basename(fname)}")

# P_METHODS_CLEAN: standalone methods figure.
# Left:  normalised speed PDFs at R_0 — the integrand shapes.
# Right: bracket CDFs — the integrated result.
# Both plotted vs v/v_c(R_0) so the operating point sits at x=1 in both panels.
# No simulation data: purely theoretical, suitable for a methods section.

v_esc_0_m = plummer_vesc(R_M[0])
q_M_0_m   = min(v_M[0] / v_esc_0_m, 1.0 - 1e-10) if v_esc_0_m > 1e-12 else 0.5
X_0_m     = X_arr[0]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# ---- Left: speed PDFs ----
ax = axes[0]
# Plummer PDF in v/v_c: g(q)*q_M_0 where g(q) = q^2(1-q^2)^{7/2} / I_full
x_p = _q_cdf_grid / q_M_0_m
y_p = (_f_cdf_grid / _I_q_full) * q_M_0_m   # change of variable: dq = q_M_0 d(v/v_c)
# Maxwell PDF in v/v_c: f(u)*X_0 where f(u) = (4/sqrt(pi)) u^2 exp(-u^2)
x_m = _u_cdf_grid / X_0_m
y_m = (4.0 / math.sqrt(math.pi)) * _f_maxw_grid * X_0_m
mask_m = x_m <= 3.0
ax.plot(x_p, y_p, color="b", lw=1.5,
        label=r"Plummer  $g(q) \propto q^2(1-q^2)^{7/2}$")
ax.plot(x_m[mask_m], y_m[mask_m], color="r", lw=1.5, ls="--",
        label=r"Maxwell  $f(u) \propto u^2 e^{-u^2}$")
ax.axvline(1.0, color="0.4", ls=":", lw=0.8,
           label=r"$v = v_c(R_0)$")
ax.axvline(1.0 / q_M_0_m, color="b", ls="--", lw=0.7, alpha=0.5,
           label=r"$v_\mathrm{esc}(R_0)$  (Plummer cutoff)")
ax.set_xlabel(r"$v\,/\,v_c(R_0)$")
ax.set_ylabel("normalised PDF")
ax.set_title("speed distributions at $R_0$")
ax.set_xlim(0, 3.0)
ax.set_ylim(bottom=0)
ax.legend(fontsize=8)

# ---- Right: bracket CDFs ----
ax = axes[1]
x_p2 = _q_cdf_grid / q_M_0_m
x_m2 = _u_cdf_grid / X_0_m
mask_m2 = x_m2 <= 3.0
ax.plot(x_p2, _B_cdf, color="b", lw=1.5,
        label=r"$B_\mathrm{Plummer}(q)$")
ax.plot(x_m2[mask_m2], _B_maxw_cdf[mask_m2], color="r", lw=1.5, ls="--",
        label=r"$B_\mathrm{Maxwell}(X)$")
B_P_vc_m = float(np.interp(q_M_0_m, _q_cdf_grid, _B_cdf))
B_M_vc_m = float(np.interp(X_0_m,   _u_cdf_grid, _B_maxw_cdf))
ax.axvline(1.0, color="0.4", ls=":", lw=0.8,
           label=rf"$v=v_c(R_0)$:  $B_P={B_P_vc_m:.2f}$,  $B_M={B_M_vc_m:.2f}$")
ax.scatter([1.0, 1.0], [B_P_vc_m, B_M_vc_m], color=["b", "r"], s=50, zorder=5)
ax.axvline(1.0 / q_M_0_m, color="b", ls="--", lw=0.7, alpha=0.5,
           label=r"$v_\mathrm{esc}(R_0)$  (Plummer support ends,  $B_P \to 1$)")
ax.set_xlabel(r"$v\,/\,v_c(R_0)$")
ax.set_ylabel("$B$  (fraction of slower particles)")
ax.set_title("bracket CDFs: Plummer vs Maxwell")
ax.set_xlim(0, 3.0)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=8)

plt.tight_layout()
fname = os.path.join(run_dir, "plot_methods_clean.pdf")
fig.savefig(fname, bbox_inches="tight")
plt.close(fig)
print(f"written: {os.path.basename(fname)}")

# P2: energetics
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
ax = axes[0]
ax.plot(times, dE_over_E0, color="k", lw=1.0)
ax.axhline(1e-3, color="k", ls="--", lw=0.8, label="0.1%")
ax.set_xlabel("t")
ax.set_ylabel("|dE/E0|")
ax.set_title("energy conservation")
ax.set_yscale("log")
ax.legend(fontsize=7)

ax = axes[1]
# Decompose into 4 components that sum EXACTLY to ΔE_tot.
# E_tot = K_pert + K_bg + W_bg_arr + 0.5*M_p*phi_p
# → ΔE_tot = DK_pert + DPhi_half + DK_bg + DW_bg   (see code comments)
phi_p_arr  = E_orb_M - 0.5 * v_M**2                         # potential at perturber
DK_pert    = 0.5 * M_p * (v_M**2 - v_M[0]**2)               # perturber KE change
DPhi_half  = 0.5 * M_p * (phi_p_arr - phi_p_arr[0])         # ½ perturber PE change
# (factor ½ because the perturber-background interaction is split 50/50 in E_tot)
DK_bg_plot = K_bg_arr - K_bg_arr[0]                          # background KE change
DW_bg_plot = W_bg_arr - W_bg_arr[0]                          # background PE change
DE_tot     = E_tot_arr - E_tot_arr[0]                        # total (≈ 0)

ax.plot(times, DK_pert,    color="darkred",        lw=1.2, label=r"$\Delta K_p$  perturber KE")
ax.plot(times, DPhi_half,  color="tomato",         lw=1.2, label=r"$\Delta\Phi_p$  perturber PE")
ax.plot(times, DK_bg_plot, color="navy",           lw=1.2, label=r"$\Delta K_\mathrm{bg}$  background KE")
ax.plot(times, DW_bg_plot, color="cornflowerblue", lw=1.2, label=r"$\Delta W_\mathrm{bg}$  background PE")
ax.plot(times, DE_tot,     color="green",          lw=1.8, ls="-.",
        label=r"$\Delta E_\mathrm{tot}$  (sum = conservation check)")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("t")
ax.set_ylabel("total energy change")
ax.set_title("energy partitioning  (4 components + conservation)")
ax.legend(fontsize=7)

plt.tight_layout()
fname = os.path.join(run_dir, "plot_energetics.pdf")
fig.savefig(fname, bbox_inches="tight")
plt.close(fig)
print(f"written: {os.path.basename(fname)}")

# P3: perturber trajectory
fig, ax = plt.subplots(figsize=(5, 5))
sc = ax.scatter(x_M_traj, y_M_traj, c=times, s=1.5, cmap="viridis")
plt.colorbar(sc, ax=ax, label="t")
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("perturber trajectory")
ax.add_patch(
    plt.Circle((0, 0), r_hm, color="gray", fill=False, ls="--", lw=0.8, label="r_hm")
)
ax.legend(fontsize=7)
plt.tight_layout()
fname = os.path.join(run_dir, "plot_orbit_wake.pdf")
fig.savefig(fname, bbox_inches="tight")
plt.close(fig)
print(f"written: {os.path.basename(fname)}")

# P4: Coulomb logarithm — measured vs R-dependent theory only
ln_lam_R_theory = np.array([ln_lam_at_R(R_M[i], v_M[i]) for i in range(n_snaps)])

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

ax = axes[0]
ax.plot(
    times,
    lnlam_eff,
    color="k",
    lw=0.8,
    alpha=0.7,
    label=r"measured  $(-\dot{L}\,|v|/|L|)\,/\,a_\mathrm{DF}(v_M,\,\ln\Lambda{=}1)$  [FIX 5]",
)
ax.plot(
    times,
    np.where(ln_lam_R_theory > 0, ln_lam_R_theory, np.nan),
    color="b",
    lw=1.2,
    ls="-.",
    label="theory  ln(M(<R)/M_p)",
)
ax.set_xlabel("t")
ax.set_ylabel("ln Lambda")
ax.set_title("Coulomb logarithm vs time")
y_max = float(np.nanmax(ln_lam_R_theory)) * 1.4 if np.any(ln_lam_R_theory > 0) else 1.0
ax.set_ylim(0, y_max)
ax.legend(fontsize=7)

ax = axes[1]
finite = np.isfinite(lnlam_eff)
ax.plot(
    R_M[finite], lnlam_eff[finite], ".", color="k", ms=2, alpha=0.5, label="measured"
)
r_plot = np.linspace(0.01, R_M.max() * 1.05, 400)
ln_plot = np.array([ln_lam_at_R(r) for r in r_plot])
ax.plot(
    r_plot[ln_plot > 0],
    ln_plot[ln_plot > 0],
    color="b",
    lw=1.2,
    ls="-.",
    label="theory  ln(M(<R)/M_p)",
)
ax.set_xlabel("R_M")
ax.set_ylabel("ln Lambda")
ax.set_xlim(0, R_M.max() * 1.05)
ax.set_ylim(0, y_max)
ax.set_title("Coulomb logarithm vs orbital radius")
ax.legend(fontsize=7)

plt.tight_layout()
fname = os.path.join(run_dir, "plot_coulomb_log.pdf")
fig.savefig(fname, bbox_inches="tight")
plt.close(fig)
print(f"written: {os.path.basename(fname)}")

# ============================================================
#  SUMMARY PRINT
# ============================================================

n_unbound_val = (
    int(np.sum((0.5 * np.sum(vel_bg_last**2, axis=1) + phi_bg_last) > 0))
    if phi_bg_last is not None
    else -1
)

lnlam_eff_med = float(np.nanmedian(lnlam_eff))

print(f"\nSTATISTICAL GOODNESS-OF-FIT  (background, final snapshot t={times[-1]:.2f})")
print(
    f"  rho(r)    chi2_nu={chi2_rho:6.3f}  p={p_rho:.4f}"
    f"  [Pearson, sigma=E-based, r>2eps]"
)
print(
    f"  CDF(r)    KS D   ={ks_stat:6.4f}  p={p_ks:.4f}"
    f"  [KS test on raw radii vs Plummer CDF]"
)
if has_q:
    print(
        f"  P(q)      chi2_nu={chi2_q:6.3f}  p={p_q:.4f}"
        f"  [Pearson, sigma=sqrt(E), dof_adj=1]"
    )
print(
    f"  sigma_r   chi2_nu={chi2_sigr:6.3f}  p={p_sigr:.4f}"
    f"  [sigma=sigma_th/sqrt(2(n-1))]"
)
print(f"  sigma_t   chi2_nu={chi2_sigt:6.3f}  p={p_sigt:.4f}")

print(f"\n{'=' * 65}")
print(f"  Snapshots:              {n_snaps}")
print(f"  Time span:              {times[-1]:.4f}  ({times[-1] / t_dyn:.1f} t_dyn)")
print(f"  r_hm(t=0):              {r_hm_initial:.4f}  (theory = {r_hm:.4f})")
print(f"  r_hm offset:            {100 * r_hm_offset:+.2f}%")
print(f"  sigma_v(t=0):           {sigma_v_arr[0]:.4f}  (theory = {sv_theory:.4f})")
print(f"  sigma_v drift:          {100 * sv_drift:+.2f}%")
print(f"  Mean 2K/|W|:            {virial_mean:.4f} +/- {virial_std:.4f}")
print(f"  Max |dE/E0|:            {energy_err:.2e}  ({'PASS' if is_good else 'FAIL'})")
if n_unbound_val >= 0:
    print(
        f"  Unbound particles:      {n_unbound_val} / {N_BG}"
        f"  ({100 * n_unbound_val / N_BG:.2f}%)"
    )
print(f"  R_M(0) -> R_M(f):       {R_M[0]:.4f} -> {R_M[-1]:.4f}  (dR/dt={slope_R:.3e})")
print(f"  R_stall:                {R_stall:.4f}  [M(<R)=M_p, lnΛ=0]")
print(f"  ln_Lambda_eff (median): {lnlam_eff_med:.3f}")
print(f"  ln_Lambda(R_M(0)):      {ln_lam_at_R(R_M[0]):.3f}  [= ln(M(<R0)/M_p)]")
print(f"  ln_Lambda(R_M(f)):      {ln_lam_at_R(R_M[-1]):.3f}  [at final radius]")
print(f"  DK_bg / K_bg(0):        {DK_bg[-1] / K_bg_arr[0]:.4f}")
print(f"  B_Plummer at t=0:       {B_X_arr[0]:.4f}  [exact Plummer DF, FIX 3]")
print(f"  B_Maxwell at t=0:       {B_X_Maxwell_arr[0]:.4f}  [Maxwellian, CDF table]")
print(f"  B_Plummer at t=f:       {B_X_arr[-1]:.4f}")
print(
    f"  Lz/|L| min:             {float(np.nanmin(Lz_over_L)):.4f}  (1.000 = no precession)"
)
print("=" * 65)

# ============================================================
#  SAVE NPZ
# ============================================================

E_final_bg = (
    0.5 * np.sum(vel_bg_last**2, axis=1) + phi_bg_last
    if phi_bg_last is not None
    else np.full(N_BG, np.nan)
)

lag_keys = {f"lag_{int(f * 100):02d}": lagr_arr[:, k] for k, f in enumerate(LAG_FRACS)}

np.savez(
    os.path.join(run_dir, "perturber_stats.npz"),
    # parameters
    eta=np.float64(ETA),
    R0_init=np.float64(R0_ARG),
    N_bg=np.int64(N_BG),
    eps=np.float64(EPS),
    E0_total=np.float64(E0_total),
    # time axis
    times=times,
    # perturber time series
    R_M=R_M,
    v_M=v_M,
    Lz_M=Lz_M,
    L_M=L_M,
    E_orb_M=E_orb_M,
    dR_dt=dR_dt,
    R_chandra_Rdep=R_chandra_Rdep,  # Plummer DF + RK45  [FIX 3+4]
    R_chandra_Rdep_Maxwell=R_chandra_Rdep_Maxwell,  # Maxwell + RK45 (comparison)
    lnlam_eff=lnlam_eff,
    ln_lam_R_arr=ln_lam_R_arr,
    a_meas=a_meas,
    a_chandra_Rdep=a_chandra_Rdep,  # Plummer DF bracket  [FIX 3]
    a_chandra_Rdep_Maxwell=a_chandra_Rdep_Maxwell,  # Maxwell bracket (comparison)
    Lz_over_L=Lz_over_L,  # [ADD M5]
    X_arr=X_arr,  # Maxwellian X = v_M/(sqrt2*sigma)
    B_X_arr=B_X_arr,  # exact Plummer DF bracket  [FIX 3]
    B_X_Maxwell_arr=B_X_Maxwell_arr,  # Maxwellian bracket (comparison)
    # background time series
    K_bg=K_bg_arr,
    W_bg=W_bg_arr,
    E_tot=E_tot_arr,
    virial=virial_arr,
    sigma_v=sigma_v_arr,
    sigma_vr=sigma_vr_arr,
    sigma_vt=sigma_vt_arr,
    mean_vr=mean_vr_arr,
    r_cm_bg=r_cm_bg_arr,
    v_r_M=v_r_M,
    v_t_M=v_t_M,
    circularity=circularity,
    dE_over_E0=dE_over_E0,
    DK_bg=DK_bg,
    # scalars
    slope_R=np.float64(slope_R),
    slope_L=np.float64(slope_L),
    max_dE=np.float64(energy_err),
    is_good=np.bool_(is_good),
    mean_virial=np.float64(virial_mean),
    std_virial=np.float64(virial_std),
    r_hm_median=np.float64(r_hm_median),
    r_hm_offset=np.float64(r_hm_offset),
    sigma_v_drift=np.float64(sv_drift),
    lnlam_eff_median=np.float64(lnlam_eff_med),
    ln_lam_R0=np.float64(ln_lam_at_R(R_M[0])),
    ln_lam_Rf=np.float64(ln_lam_at_R(R_M[-1])),
    DK_bg_frac=np.float64(DK_bg[-1] / K_bg_arr[0]),
    # goodness-of-fit
    chi2_rho=np.float64(chi2_rho),
    p_rho=np.float64(p_rho),
    ks_stat=np.float64(ks_stat),
    p_ks=np.float64(p_ks),
    chi2_q=np.float64(chi2_q),
    p_q=np.float64(p_q),
    chi2_sigr=np.float64(chi2_sigr),
    p_sigr=np.float64(p_sigr),
    chi2_sigt=np.float64(chi2_sigt),
    p_sigt=np.float64(p_sigt),
    E_final_bg=E_final_bg,
    **lag_keys,
)
print(f"saved: perturber_stats.npz")

#!/usr/bin/env python3
# summary_perturber.py
# Per-run analysis of Plummer sphere + perturber N-body simulation.
# Particle index 0 = perturber; indices 1..N_BG = background Plummer sphere.
#
#   Background analyses, chi2 and KS goodness-of-fit tests at final snapshot.
#   Perturber analyses: orbital decay vs Chandrasekhar (R-dependent ln_Lambda only),
#   angular momentum loss, speed vs v_circ, deceleration (-dL/dt / R) vs theory,
#   energy conservation, energy partitioning, trajectory, Coulomb logarithm.
#
# python3 summary_perturber.py <run_dir> --eta E --R0 R --N N [--eps EPS]

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


def chandrasekhar_bracket(X):
    return math.erf(X) - (2.0 * X / math.sqrt(math.pi)) * math.exp(-(X**2))


def chandrasekhar_decel(R, v_M, M_p_val, ln_lam_val):
    rho_r = plummer_rho(R)
    sig2 = G * M_tot / (6.0 * math.sqrt(R**2 + b**2))
    sigma = math.sqrt(sig2)
    if v_M <= 0.0 or sigma <= 0.0 or ln_lam_val <= 0.0:
        return 0.0
    X = v_M / (math.sqrt(2.0) * sigma)
    B = chandrasekhar_bracket(X)
    if B <= 0.0:
        return 0.0
    return 4.0 * math.pi * G**2 * M_p_val * rho_r * ln_lam_val * B / v_M**2


def ln_lam_at_R(R, v_M=None):
    """
    R-dependent Coulomb logarithm.
    If v_M is given: ln(R * (v_M^2 + sigma^2(R)) / (G*M_p))  -- exact form.
    If v_M is None:  ln(M(<R) / M_p)  -- circular orbit approximation.
    Both floored at ln(1.1).
    """
    if v_M is not None:
        sig2 = G * M_tot / (6.0 * math.sqrt(R**2 + b**2))
        v_rel2 = v_M**2 + sig2
        return math.log(max(R * v_rel2 / (G * M_p), 1.1))
    return math.log(max(plummer_mass_enc(R) / M_p, 1.1))


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
            pos = np.array(
                [[float(x) for x in fh.readline().split()] for _ in range(N_tot)]
            )
            vel = np.array(
                [[float(x) for x in fh.readline().split()] for _ in range(N_tot)]
            )
            phi = np.array([float(fh.readline()) for _ in range(N_tot)])
            yield t, pos, vel, phi


# ============================================================
#  PASS 1 — MAIN STREAMING PASS
# ============================================================

times = []
R_M_arr = []
v_M_arr = []
Lz_M_arr = []
L_M_arr = []
E_orb_M_arr = []
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
    E_orb_M_arr.append(E_orb_p)
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
E_orb_M = np.array(E_orb_M_arr)
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
dt_snap = float(np.median(np.diff(times))) if n_snaps > 1 else 1.0
dR_dt = np.gradient(R_M, dt_snap)

slope_R, *_ = stats.linregress(times, R_M)
slope_L, *_ = stats.linregress(times, L_M)

# R-dependent Coulomb log along the actual trajectory
ln_lam_R_arr = np.array([ln_lam_at_R(R_M[i], v_M[i]) for i in range(n_snaps)])

# Measured deceleration: -dL/dt / R
# Smooth L first to remove N-body noise, then differentiate.
L_M_smooth = gaussian_filter1d(L_M, sigma=4.0)
dL_dt = np.gradient(L_M_smooth, dt_snap)
with np.errstate(divide="ignore", invalid="ignore"):
    a_meas = np.where(R_M > 0.05, -dL_dt / R_M, np.nan)

# Theory deceleration: R-dependent ln_Lambda only
a_chandra_Rdep = np.array(
    [
        chandrasekhar_decel(R_M[i], v_M[i], M_p, ln_lam_at_R(R_M[i], v_M[i]))
        for i in range(n_snaps)
    ]
)

# Effective Coulomb log: a_meas / a_DF(ln=1)
a_chandra_lam1 = np.array(
    [chandrasekhar_decel(R_M[i], v_M[i], M_p, 1.0) for i in range(n_snaps)]
)
with np.errstate(divide="ignore", invalid="ignore"):
    lnlam_eff = np.where(
        (a_chandra_lam1 > 1e-12) & np.isfinite(a_meas), a_meas / a_chandra_lam1, np.nan
    )


# Chandrasekhar ODE: R-dependent ln_Lambda only
def _chandra_ode_Rdep():
    Rc = np.empty(n_snaps)
    Rc[0] = R_M[0]
    for i in range(1, n_snaps):
        R_prev = Rc[i - 1]
        if R_prev < 0.01:
            Rc[i] = R_prev
            continue
        vc = plummer_vcirc(R_prev)
        a_fc = chandrasekhar_decel(R_prev, max(vc, 0.01), M_p, ln_lam_at_R(R_prev))
        dLdR = plummer_dLdR(R_prev)
        dR = -a_fc * R_prev / max(dLdR, 1e-10) * dt_snap
        Rc[i] = max(0.01, R_prev + dR)
    return Rc


R_chandra_Rdep = _chandra_ode_Rdep()

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

# BG3: density profile initial / mid / final
r_theory = np.logspace(-1.5, 1.5, 300)
rho_th_plot = np.array([plummer_rho(r) for r in r_theory])

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, pos_s, label in [
    (axes[0], pos_bg_initial, "t=0"),
    (axes[1], pos_bg_mid, f"t={t_mid_actual:.2f}"),
    (axes[2], pos_bg_last, f"t={times[-1]:.2f}"),
]:
    r_m, rho_m, cnts = bin_density_profile(pos_s)
    if r_m is not None:
        log_sig = 1.0 / np.sqrt(cnts)
        ax.errorbar(
            r_m,
            rho_m,
            yerr=[rho_m * (1.0 - 10.0 ** (-log_sig)), rho_m * (10.0**log_sig - 1.0)],
            fmt="o",
            ms=3,
            color="k",
            label="N-body",
        )
    ax.plot(r_theory, rho_th_plot, "r--", label="Plummer theory")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("r")
    ax.set_title(label)
    ax.legend(fontsize=7)
axes[0].set_ylabel("density")
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
    axes[1].plot(
        times, lagr_arr[:, k] / lagr_arr[0, k], lw=1.0, label=f"r_{int(frac * 100)}%"
    )
axes[0].axhline(r_hm, color="k", ls=":", lw=0.8)
axes[0].set_xlabel("t")
axes[0].set_ylabel("Lagrangian radius")
axes[0].set_title("absolute  (dashed = theory)")
axes[0].legend(fontsize=7)
axes[1].axhline(1.0, color="k", ls="--", lw=0.8, label="= 1")
axes[1].set_xlabel("t")
axes[1].set_ylabel("r / r(t=0)")
axes[1].set_title("normalised  (= 1 means stable)")
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
axes[2].set_title("mean radial velocity  (should be ~0)")
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

# P1: orbital decay — 4 panels
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

ax = axes[0, 0]
ax.plot(times, R_M, color="k", lw=0.8, label="R_M(t)")
ax.plot(
    times, R_chandra_Rdep, color="b", lw=1.5, ls="-.", label="Chandra  ln(M(<R)/M_p)"
)
ax.axhline(r_hm, color="k", ls=":", lw=0.8, label="r_hm")
ax.set_xlabel("t")
ax.set_ylabel("R_M")
ax.set_title("orbital radius decay")
ax.legend(fontsize=7)

ax = axes[0, 1]
ax.plot(times, L_M, color="k", lw=0.8, label="|L_M|")
ax.plot(times, Lz_M, color="0.5", lw=0.6, alpha=0.6, label="L_z")
ax.set_xlabel("t")
ax.set_ylabel("specific angular momentum")
ax.set_title("angular momentum loss")
ax.legend(fontsize=7)

ax = axes[1, 0]
ax.plot(times, v_M, color="k", lw=0.8, label="|v_M|")
vc_theory_arr = np.array([plummer_vcirc(r) for r in R_M])
ax.plot(times, vc_theory_arr, color="r", lw=1.2, ls="--", label="v_circ theory")
ax.set_xlabel("t")
ax.set_ylabel("speed")
ax.set_title("speed vs circular velocity")
ax.legend(fontsize=7)

ax = axes[1, 1]
a_Rdep_plot = np.where(a_chandra_Rdep > 0, a_chandra_Rdep, np.nan)
ax.plot(times, a_meas, color="k", lw=0.8, alpha=0.7, label=r"$-\dot{L}/R$  (measured)")
ax.plot(times, a_Rdep_plot, color="b", lw=1.5, ls="-.", label="Chandra  ln(M(<R)/M_p)")
valid_pos = np.concatenate(
    [
        a_meas[a_meas > 0] if np.any(a_meas > 0) else np.array([]),
        a_Rdep_plot[np.isfinite(a_Rdep_plot)],
    ]
)
if len(valid_pos):
    ax.set_ylim(valid_pos.min() * 0.3, valid_pos.max() * 3.0)
ax.set_xlabel("t")
ax.set_ylabel("deceleration")
ax.set_title("friction deceleration  (-dL/dt / R)")
ax.set_yscale("log")
ax.legend(fontsize=7)

plt.tight_layout()
fname = os.path.join(run_dir, "plot_orbital_decay.pdf")
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
ax.plot(times, E_orb_M - E_orb_M[0], color="k", lw=1.2, label="dE_orb perturber")
ax.plot(times, DK_bg, color="0.5", lw=1.2, label="DK_bg background")
ax.plot(
    times, (E_orb_M - E_orb_M[0]) + DK_bg, color="0.7", lw=1.0, ls="--", label="sum"
)
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("t")
ax.set_ylabel("energy change")
ax.set_title("energy partitioning")
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
    label="measured  (-dL/dt / R) / a_DF(ln=1)",
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
print(f"  ln_Lambda_eff (median): {lnlam_eff_med:.3f}")
print(f"  ln_Lambda(R_M(0)):      {ln_lam_at_R(R_M[0]):.3f}  [= ln(M(<R0)/M_p)]")
print(f"  ln_Lambda(R_M(f)):      {ln_lam_at_R(R_M[-1]):.3f}  [at final radius]")
print(f"  DK_bg / K_bg(0):        {DK_bg[-1] / K_bg_arr[0]:.4f}")
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
    R_chandra_Rdep=R_chandra_Rdep,
    lnlam_eff=lnlam_eff,
    ln_lam_R_arr=ln_lam_R_arr,
    a_meas=a_meas,
    a_chandra_Rdep=a_chandra_Rdep,
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

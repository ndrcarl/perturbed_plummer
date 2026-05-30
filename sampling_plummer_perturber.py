#!/usr/bin/env python3
# sampling_plummer_perturber.py
# Plummer sphere + perturber initial conditions for the Barnes treecode.
# Code units: G=1, M_tot=1, b=1.
# Perturber (index 0) at R0 with tangential speed vfrac*v_circ(R0).
#
# Changes vs original (marked [FIX 1] and [FIX 2]):
#
#   [FIX 1] v_rel_rp: linear sum -> quadrature sum.
#           Notes write  eps = alpha*G*M_BH / (v_BH^2 + sigma^2),
#           so the characteristic relative speed is v_rel = sqrt(v^2 + sigma^2),
#           not v + sigma.  The old form overestimated v_rel by ~36%, causing
#           eps and dt to be slightly underestimated.
#
#   [FIX 2] tstop lower bound: SIS formula -> Plummer-native Chandrasekhar estimate.
#           The SIS formula  t_DF = R0^2*v_circ / (0.8*G*M_p*ln_Lambda)  assumes
#           v_circ = const and uses ln_Lambda = ln(1/eta), neither of which holds
#           for a Plummer sphere.
#
#           The handwritten notes (Images 1 & 2) derive an analytic integral for
#           the inspiral time, but the full prefactor K is not written out in the
#           images we have, so F(R0)/K cannot be assembled.
#
#           Instead, we use a simple Plummer-native estimate:
#               t_DF_Plummer = R0 * v_circ(R0) / (2 * a_DF(R0))
#           where a_DF uses the full Chandrasekhar formula with Plummer rho, sigma,
#           and ln_Lambda = ln(M_enc(R0)/M_p).  This is ~7.5 code units at default
#           parameters, still well below 5*T_orb=66, so tstop_lower remains
#           dominated by the orbital timescale in all tested cases.
#
#           The SIS estimate and its ln_lambda are removed entirely.

import numpy as np
import random
import math
import argparse
import sys
import json

parser = argparse.ArgumentParser()
parser.add_argument('--eta',    type=float, default=0.05)
parser.add_argument('--R0',     type=float, default=None)
parser.add_argument('--vfrac',  type=float, default=1.0)
parser.add_argument('--N',      type=int,   default=10000)
parser.add_argument('--seed',   type=int,   default=None)
parser.add_argument('--out',    type=str,   default='plummer_perturber.txt')
parser.add_argument('--params', type=str,   default='params_recommended.json')
parser.add_argument('--dtout',  type=float, default=1/4,
                    help='output time interval (code units)')
args = parser.parse_args()

# ---- code units ----
N      = args.N
M_tot  = 1.0
b      = 1.0
G      = 1.0
mass_i = M_tot / N
eta    = args.eta
M_pert = eta * M_tot
vfrac  = args.vfrac
dtout  = args.dtout

if not (0.0 < eta < 1.0):
    sys.exit(f"--eta must be in (0,1), got {eta}")
if not (0.0 < vfrac <= 1.0):
    sys.exit(f"--vfrac must be in (0,1], got {vfrac}")
if N < 100:
    sys.exit(f"--N must be >= 100, got {N}")
if dtout <= 0.0:
    sys.exit(f"--dtout must be positive, got {dtout}")

# ---- structural quantities ----
r_hm  = b / math.sqrt(2.0**(2.0/3.0) - 1.0)   # half-mass radius ~ 1.305 b
R0    = args.R0 if args.R0 is not None else r_hm
if R0 <= 0.0:
    sys.exit(f"--R0 must be positive, got {R0}")

rho_c = 3.0 * M_tot / (4.0 * math.pi * b**3)
t_dyn = math.sqrt(3.0 * math.pi / (16.0 * G * rho_c))   # = pi/2 in code units

V_hm   = (4.0 / 3.0) * math.pi * r_hm**3
n_hm   = (N / 2.0) / V_hm
d_mean = n_hm**(-1.0 / 3.0)                              # mean interparticle separation

t_relax = (N / (8.0 * math.log(N))) * t_dyn              # two-body relaxation time

# ---- perturber at R0 ----
M_enc_R0  = M_tot * R0**3 / (R0**2 + b**2)**1.5
v_circ_R0 = math.sqrt(G * M_enc_R0 / R0)
v_pert    = vfrac * v_circ_R0
sigma2_R0 = G * M_tot / (6.0 * math.sqrt(R0**2 + b**2))  # Jeans 1D dispersion^2
sigma_R0  = math.sqrt(sigma2_R0)
T_orb     = 2.0 * math.pi * R0 / v_circ_R0

# ---- pericenter (= R0 for circular orbit) ----
r_peri    = R0
sigma2_rp = sigma2_R0
v_tang_rp = v_circ_R0

# [FIX 1] v_rel: quadrature sum, not linear sum.
# Notes write  eps = alpha * G*M_BH / (v_BH^2 + sigma^2),
# so the characteristic relative speed satisfies  v_rel^2 = v_M^2 + sigma^2.
# The old  v_rel = v_M + sigma  overestimated by ~36%, causing eps and dt to
# be slightly underestimated.
v_rel_rp = math.sqrt(v_tang_rp**2 + sigma2_rp)           # [FIX 1]

# ---- eps recommendation ----
# Notes: eps ~ min(alpha*d_mean, alpha*r_inf) with r_inf = G*M_p/(v^2+sigma^2)
alpha_eps      = 0.1
eps_background = 0.1 * d_mean
r_inf_peri     = G * M_pert / (v_tang_rp**2 + sigma2_rp)
eps_recommended = max(eps_background, alpha_eps * r_inf_peri)

# ---- dtime recommendation ----
# Notes: dt = eta_acc * min(t_2body, t_potential)
eta_acc        = 0.05
t_2body        = eps_recommended / v_rel_rp               # softened crossing time
t_potential    = T_orb / (2.0 * math.pi)                  # orbital timescale / 2pi
dt_recommended = eta_acc * min(t_2body, t_potential)

def nearest_power2_dt(dt_val):
    k = math.ceil(math.log2(1.0 / dt_val))
    k = max(9, min(k, 13))
    return 2**k, 1.0 / 2**k

dtime_denom, dt_value = nearest_power2_dt(dt_recommended)

# ---- tstop recommendation — Plummer-native DF estimate              [FIX 2] ----
#
# Coulomb logarithm: Plummer derivation.
#   b_min = G*M_p/(v_circ^2 + sigma^2) = G*M_p*R0/M_enc(R0)
#   b_max ~ R0
#   => ln_Lambda = ln(M_enc(R0) / M_p)   [v_M = v_circ, leading order]
# This replaces the SIS approximation ln_Lambda = ln(1/eta).
ln_lambda_R0 = math.log(max(M_enc_R0 / M_pert, 1.1))

# Chandrasekhar bracket B(X0) at R0 with v_M = v_circ(R0).
# B(X) = erf(X) - (2X/sqrt(pi))*exp(-X^2),  X = v_M / (sqrt(2)*sigma)
X0 = v_circ_R0 / (math.sqrt(2.0) * sigma_R0)
B0 = math.erf(X0) - (2.0 * X0 / math.sqrt(math.pi)) * math.exp(-X0**2)
B0 = max(B0, 1e-6)

# Plummer density at R0
rho_R0 = 3.0 * M_tot / (4.0 * math.pi * b**3) * (1.0 + (R0 / b)**2)**(-2.5)

# DF deceleration at R0 (full Chandrasekhar formula, Plummer quantities)
a_DF_R0 = 4.0 * math.pi * G**2 * M_pert * rho_R0 * ln_lambda_R0 * B0 / v_circ_R0**2

# Simple Plummer DF inspiral timescale: t ~ R0*v_circ / (2*a_DF)
# This is a lower bound (assumes a_DF constant, correct at order of magnitude).
# The notes (Images 1&2) derive an analytic integral for the exact inspiral time,
# but the full prefactor K is not yet extracted from those pages; when it is,
# t_DF_Plummer can be replaced by F(R0)/K where
#   F(R0) = (1/2)*ln(1+R0^2) + 1/(1+R0^2) - 1/(4*(1+R0^2)^2) - 3/4
# (antiderivative of r^5/(1+r^2)^3 evaluated from 0 to R0).
t_DF_Plummer = R0 * v_circ_R0 / (2.0 * max(a_DF_R0, 1e-30))

tstop_lower       = max(5.0 * T_orb, 3.0 * t_DF_Plummer)
tstop_upper       = t_relax / 5.0
tstop_raw         = min(tstop_lower, tstop_upper)
tstop_recommended = max(10.0 * round(tstop_raw / 10.0), 10.0)

# ---- random seed ----
random.seed(42)
np.random.seed(42)

# ---- print summary ----
print(f"N={N}  eta={eta}  R0={R0:.4f}  eps={eps_recommended:.4f}  dtime={dt_recommended:.6f}")
print(f"tstop_lower={tstop_lower:.1f}  (5*T_orb={5*T_orb:.1f}, 3*t_DF={3*t_DF_Plummer:.1f})")
print(f"  [Plummer DF: a_DF={a_DF_R0:.4e}  X0={X0:.3f}  B(X0)={B0:.3f}  lnL={ln_lambda_R0:.3f}]")
print(f"tstop_upper={tstop_upper:.1f}  (t_relax/5)")
print(f"tstop_recommended={tstop_recommended:.0f}  dtout={dtout}")

# ---- write params JSON ----
params = {
    "eta"            : eta,
    "R0"             : R0,
    "vfrac"          : vfrac,
    "N_bg"           : N,
    "eps"            : round(eps_recommended, 6),
    "dtime_denom"    : dtime_denom,
    "dtime_str"      : f"1/{dtime_denom}",
    "tstop"          : float(tstop_recommended),
    "dtout"          : dtout,
    "theta"          : 0.50,
    "n_snapshots"    : int(tstop_recommended / dtout),
    "t_DF_Plummer"   : round(t_DF_Plummer, 2),      # replaces t_DF_SIS
    "a_DF_R0"        : round(a_DF_R0, 6),
    "X0"             : round(X0, 4),                 # Chandrasekhar argument at R0
    "B0"             : round(B0, 4),                 # Chandrasekhar bracket at R0
    "ln_lambda_R0"   : round(ln_lambda_R0, 4),       # replaces ln_lambda
    "T_orb"          : round(T_orb, 4),
    "t_relax"        : round(t_relax, 1),
    "r_peri"         : round(r_peri, 6),
}

with open(args.params, 'w') as f:
    json.dump(params, f, indent=2)
print(f"params written: {args.params}")

# ---- helpers ----

def isotropic_vec(mag):
    u = random.random()
    w = random.random()
    theta = math.acos(1.0 - 2.0 * u)
    phi   = 2.0 * math.pi * w
    return (mag * math.sin(theta) * math.cos(phi),
            mag * math.sin(theta) * math.sin(phi),
            mag * math.cos(theta))

def get_q():
    # Rejection sampling for the Plummer DF speed distribution:
    # f(q) propto q^2 * (1 - q^2)^(7/2),  q = v / v_esc  in [0, 1)
    g_max = 0.093
    while True:
        q = random.random()
        g = (q**2) * ((1.0 - q**2)**3.5)
        if random.random() * g_max < g:
            return q

# ---- sample background ----
print("sampling background...", flush=True)
positions  = []
velocities = []

for _ in range(N):
    # Position: inversion sampling of Plummer CDF M(<r)/M_tot = r^3/(r^2+b^2)^(3/2)
    X   = random.random()
    r   = b / math.sqrt(X**(-2.0 / 3.0) - 1.0)
    pos = isotropic_vec(r)
    positions.append(list(pos))

    # Velocity: rejection sample Plummer DF via q = v/v_esc
    Psi   = G * M_tot / math.sqrt(r**2 + b**2)
    v_esc = math.sqrt(2.0 * Psi)
    q     = get_q()
    v     = q * v_esc
    vel   = isotropic_vec(v)
    velocities.append(list(vel))

positions  = np.array(positions)
velocities = np.array(velocities)

# ---- perturber position ----
pos_pert = np.array([R0, 0.0, 0.0])

# ---- CM correction (two-stage) ----
M_total_system = M_tot + M_pert
pos_cm_bg  = np.mean(positions, axis=0)
pos_cm_tot = (M_tot * pos_cm_bg + M_pert * pos_pert) / M_total_system
positions -= pos_cm_tot
pos_pert  -= pos_cm_tot

# Recompute v_circ at the CM-corrected perturber radius
R0_actual     = float(np.linalg.norm(pos_pert))
M_enc_actual  = M_tot * R0_actual**3 / (R0_actual**2 + b**2)**1.5
v_circ_actual = math.sqrt(G * M_enc_actual / R0_actual)

r_hat   = pos_pert / R0_actual
z_hat   = np.array([0.0, 0.0, 1.0])
phi_hat = np.cross(z_hat, r_hat)
phi_hat /= np.linalg.norm(phi_hat)
vel_pert = vfrac * v_circ_actual * phi_hat

# Stage 2: velocity CM
vel_cm_bg  = np.mean(velocities, axis=0)
vel_cm_tot = (M_tot * vel_cm_bg + M_pert * vel_pert) / M_total_system
velocities -= vel_cm_tot
vel_pert   -= vel_cm_tot

# ---- write treecode input file ----
N_total = N + 1
print(f"writing {args.out}  (N_total={N_total})", flush=True)

with open(args.out, 'w') as f:
    f.write(f"{N_total}\n3\n0\n")
    f.write(f"{M_pert:.8f}\n")
    for _ in range(N):
        f.write(f"{mass_i:.8f}\n")
    f.write(f"{pos_pert[0]:.8f} {pos_pert[1]:.8f} {pos_pert[2]:.8f}\n")
    for p in positions:
        f.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
    f.write(f"{vel_pert[0]:.8f} {vel_pert[1]:.8f} {vel_pert[2]:.8f}\n")
    for v in velocities:
        f.write(f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")

print("done.")

#!/usr/bin/env python3
# sampling_plummer_perturber.py
# Plummer sphere + perturber initial conditions for the Barnes treecode.
# Code units: G=1, M_tot=1, b=1.
# Perturber (index 0) at R0 with tangential speed vfrac*v_circ(R0).

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
parser.add_argument('--dtout',  type=float, default=1/16,
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
r_hm  = b / math.sqrt(2.0**(2.0/3.0) - 1.0)   # ~ 1.305
R0    = args.R0 if args.R0 is not None else r_hm
if R0 <= 0.0:
    sys.exit(f"--R0 must be positive, got {R0}")

rho_c = 3.0 * M_tot / (4.0 * math.pi * b**3)
t_dyn = math.sqrt(3.0 * math.pi / (16.0 * G * rho_c))   # = pi/2

V_hm    = (4.0 / 3.0) * math.pi * r_hm**3
n_hm    = (N / 2.0) / V_hm
d_mean  = n_hm**(-1.0 / 3.0)

t_relax = (N / (8.0 * math.log(N))) * t_dyn

# ---- perturber at R0 ----
M_enc_R0  = M_tot * R0**3 / (R0**2 + b**2)**1.5
v_circ_R0 = math.sqrt(G * M_enc_R0 / R0)
v_pert    = vfrac * v_circ_R0
sigma2_R0 = G * M_tot / (6.0 * math.sqrt(R0**2 + b**2))
sigma_R0  = math.sqrt(sigma2_R0)
T_orb     = 2.0 * math.pi * R0 / v_circ_R0

# ---- pericenter ----
r_peri    = R0
sigma2_rp = sigma2_R0
v_tang_rp = v_circ_R0

v_rel_rp = v_tang_rp + math.sqrt(sigma2_rp)

# ---- eps recommendation ----
alpha_eps       = 0.1
eps_background  = 0.1 * d_mean 
r_inf_peri      = G * M_pert / (v_tang_rp**2 + sigma2_rp)
eps_recommended = max(eps_background, alpha_eps * r_inf_peri)

# ---- dtime recommendation ----
eta_acc      = 0.05
t_2body      = eps_recommended / v_rel_rp
t_potential  = T_orb / (2.0 * math.pi)
dt_recommended = eta_acc * min(t_2body, t_potential)

def nearest_power2_dt(dt_val):
    k = math.ceil(math.log2(1.0 / dt_val))
    k = max(9, min(k, 13))
    return 2**k, 1.0 / 2**k

dtime_denom, dt_value = nearest_power2_dt(dt_recommended)

# ---- tstop recommendation ----
ln_lambda = math.log(1.0 / eta) 
t_DF_SIS  = R0**2 * v_circ_R0 / (0.8 * G * M_pert * ln_lambda)
tstop_lower = max(5.0 * T_orb, 3.0 * t_DF_SIS)
tstop_upper = t_relax / 5.0
tstop_raw   = min(tstop_lower, tstop_upper)
tstop_recommended = max(10.0 * round(tstop_raw / 10.0), 10.0)

# ---- random seed ----
random.seed(42)
np.random.seed(42)

# ---- print summary ----
print(f"N={N}  eta={eta}  R0={R0:.4f}  eps={eps_recommended:.4f}  dtime={dt_recommended:.6f}")
print(f"tstop_lower={tstop_lower:.1f}  (5*T_orb={5*T_orb:.1f}, 3*t_DF={3*t_DF_SIS:.1f})")
print(f"tstop_upper={tstop_upper:.1f}  (t_relax/5)")
print(f"tstop_recommended={tstop_recommended:.0f}  dtout={dtout}")

# ---- write params JSON ----
params = {
    "eta"         : eta,
    "R0"          : R0,
    "vfrac"       : vfrac,
    "N_bg"        : N,
    "eps"         : round(eps_recommended, 6),
    "dtime_denom" : dtime_denom,
    "dtime_str"   : f"1/{dtime_denom}",
    "tstop"       : float(tstop_recommended),
    "dtout"       : dtout,
    "theta"       : 0.50,
    "n_snapshots" : int(tstop_recommended / dtout),
    "t_DF_SIS"    : round(t_DF_SIS, 2),
    "T_orb"       : round(T_orb, 4),
    "t_relax"     : round(t_relax, 1),
    "ln_lambda"   : round(ln_lambda, 4),
    "r_peri"      : round(r_peri, 6),
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
    X   = random.random()
    r   = b / math.sqrt(X**(-2.0 / 3.0) - 1.0)
    pos = isotropic_vec(r)
    positions.append(list(pos))

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

# recompute v_circ at corrected radius
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

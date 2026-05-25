#!/usr/bin/env python3
# ============================================================
#  sampling_plummer_perturber.py
#
#  Generates equilibrium initial conditions for a Plummer sphere
#  (background) PLUS a single massive perturber particle.
#
#  [BACKGROUND]
#  N equal-mass particles are sampled from the Plummer
#  distribution (m=5 polytrope) using:
#    - Inverse Transform Sampling  for positions
#    - Rejection Sampling          for velocities
#  This produces an exact sample of the Eddington DF
#  f(eps) proportional to eps^(7/2), with g(q) = q^2*(1-q^2)^(7/2).
#
#  [PERTURBER]
#  One particle of mass M = eta * M_tot is appended as particle
#  index 0 (the first particle in the output file).  It is placed
#  at Cartesian position (R0, 0, 0) with velocity (0, v0, 0),
#  where v0 = vfrac * v_circ(R0).
#
#    vfrac = 1.0  -->  circular orbit   (R0 is the orbital radius)
#    vfrac < 1.0  -->  eccentric orbit  (R0 is the apocenter)
#
#  The circular speed at R0 in the Plummer potential is:
#    v_circ(R0) = sqrt( G * M_tot * R0^2 / (b^3*(1+R0^2/b^2)^(3/2)) )
#               = sqrt( M(<R0) / R0 )    [code units: G=M_tot=b=1]
#  where M(<R0) = R0^3 / (R0^2 + 1)^(3/2).
#
#  [CM CORRECTION]
#  After assembling the full system (N background + 1 perturber),
#  the centre-of-mass position and velocity of the TOTAL system
#  are computed (weighted by mass) and subtracted from every
#  particle.  This guarantees exact CM = 0 in the treecode frame.
#
#  The perturber's mass M is included in the CM calculation.
#  Failing to include it would leave a residual CM drift of order
#  M * R0 / (M_tot + M), which for eta=0.05 and R0=1.305 gives
#  a CM offset ~ 0.062 b -- large enough to corrupt long-term
#  Lagrangian radii measurements.
#
#  [OUTPUT FORMAT]  treecode (Barnes) input format:
#    Line 1:   N_total = N + 1
#    Line 2:   3         (dimensions)
#    Line 3:   0         (time = 0)
#    Lines 4   .. N_total+3         : one mass per line
#    Lines N_total+4 .. 2*N_total+3 : one position (x y z) per line
#    Lines 2*N_total+4..3*N_total+3 : one velocity (vx vy vz) per line
#
#  Particle ordering in the file:
#    index 0          : PERTURBER  (mass M)
#    indices 1 .. N   : BACKGROUND (mass m each)
#  summary_perturber.py identifies the perturber by index 0.
#
#  [DIAGNOSTICS]
#  All key physical and numerical parameters are printed to stdout
#  so the run log captures them for later reference.
#
#  Usage:
#    python3 sampling_plummer_perturber.py [options]
#
#  Options:
#    --eta    FLOAT   perturber mass ratio M/M_tot       (default: 0.05)
#    --R0     FLOAT   initial orbital radius [code units](default: r_hm)
#    --vfrac  FLOAT   v_tangential / v_circ(R0)          (default: 1.0)
#    --N      INT     number of background particles      (default: 10000)
#    --seed   INT     random seed (omit for random)       (default: None)
#    --out    STR     output filename                     (default: plummer_perturber.txt)
# ============================================================

import numpy as np
import random
import math
import argparse
import sys

# ============================================================
#  COMMAND-LINE ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description='Generate Plummer sphere + perturber initial conditions '
                'for the Barnes treecode.')
parser.add_argument('--eta',   type=float, default=0.05,
                    help='Perturber mass ratio M/M_tot (default: 0.05)')
parser.add_argument('--R0',    type=float, default=None,
                    help='Initial orbital radius in code units '
                         '(default: r_hm = b/sqrt(2^(2/3)-1) ~ 1.305)')
parser.add_argument('--vfrac', type=float, default=1.0,
                    help='Tangential speed as fraction of v_circ(R0). '
                         '1.0 = circular orbit, <1.0 = eccentric '
                         '(R0 becomes apocenter). (default: 1.0)')
parser.add_argument('--N',     type=int,   default=10000,
                    help='Number of background particles (default: 10000)')
parser.add_argument('--seed',  type=int,   default=None,
                    help='Random seed for reproducibility. '
                         'Omit for a new random realisation each run.')
parser.add_argument('--out',   type=str,   default='plummer_perturber.txt',
                    help='Output filename (default: plummer_perturber.txt)')
args = parser.parse_args()

# ============================================================
#  PHYSICAL PARAMETERS  (code units: G=1, M_tot=1, b=1)
# ============================================================

N      = args.N
M_tot  = 1.0          # total background mass
b      = 1.0          # Plummer scale radius
G      = 1.0
mass_i = M_tot / N    # individual background particle mass

eta    = args.eta
M_pert = eta * M_tot  # perturber mass
vfrac  = args.vfrac

# --- validate inputs ---
if not (0.0 < eta < 1.0):
    sys.exit(f"Error: --eta must be in (0, 1), got {eta}")
if not (0.0 < vfrac <= 1.0):
    sys.exit(f"Error: --vfrac must be in (0, 1], got {vfrac}")
if N < 100:
    sys.exit(f"Error: --N must be >= 100, got {N}")

# --- derived structural quantities ---
# Half-mass radius: M(<r_hm) = M_tot/2  =>  r_hm = b / sqrt(2^(2/3) - 1)
r_hm = b / math.sqrt(2.0**(2.0/3.0) - 1.0)          # ~ 1.305 b

# Default R0 = r_hm if not specified
R0 = args.R0 if args.R0 is not None else r_hm
if R0 <= 0.0:
    sys.exit(f"Error: --R0 must be positive, got {R0}")

# Central density and dynamical time
rho_c = 3.0 * M_tot / (4.0 * math.pi * b**3)         # = 3/(4*pi)
t_dyn = math.sqrt(3.0 * math.pi / (16.0 * G * rho_c)) # = pi/2

# Mean interparticle separation near r_hm (half-mass sphere)
V_hm    = (4.0 / 3.0) * math.pi * r_hm**3
n_hm    = (N / 2.0) / V_hm                            # number density inside r_hm
d_mean  = n_hm**(-1.0 / 3.0)                          # mean separation

# ============================================================
#  PERTURBER ORBITAL PARAMETERS
# ============================================================

# Enclosed background mass at R0:  M(<R0) = M_tot * R0^3 / (R0^2+b^2)^(3/2)
M_enc_R0 = M_tot * R0**3 / (R0**2 + b**2)**1.5

# Circular speed at R0 (from background potential only):
#   v_circ^2(R0) = G * M(<R0) / R0
v_circ_R0 = math.sqrt(G * M_enc_R0 / R0)

# Perturber tangential speed
v_pert = vfrac * v_circ_R0

# Background velocity dispersion at R0 (isotropic Jeans solution):
#   sigma^2(R0) = G*M_tot / (6*sqrt(R0^2 + b^2))
sigma2_R0 = G * M_tot / (6.0 * math.sqrt(R0**2 + b**2))
sigma_R0  = math.sqrt(sigma2_R0)

# Orbital period (circular orbit):  T_orb = 2*pi*R0 / v_circ
T_orb = 2.0 * math.pi * R0 / v_circ_R0 if v_circ_R0 > 0.0 else float('inf')

# -------------------------------------------------------------------
# Epsilon recommendation (professor's formula, Task 4B):
#   eps = min_R [ alpha * G*M / (v_BH^2 + sigma^2(R)) ]
# The denominator is maximised (eps minimised) at R ~ r_hm for a
# circular orbit in the Plummer sphere (verified in the audit section).
# We evaluate at R0 (the starting orbit), which equals r_hm by default.
# For an eccentric orbit this gives an upper bound; the true minimum
# epsilon should be evaluated at the pericenter.
# -------------------------------------------------------------------
alpha_eps         = 0.2                                       # typical value
v_pert_sq         = v_pert**2
denom_eps         = v_pert_sq + sigma2_R0                     # v_BH^2 + sigma^2(R0)
r_inf_corrected   = G * M_pert / denom_eps                    # velocity-corrected influence radius
eps_recommended   = alpha_eps * r_inf_corrected               # recommended softening
eps_background    = d_mean / 10.0                             # background collisionality limit

# Physical influence radius (sigma only):
r_inf_sigma = G * M_pert / sigma2_R0

# -------------------------------------------------------------------
# dt recommendation (professor's formula, Task 4C):
#   dt = min(t_2body, t_potential)
#   t_2body     = eps / v_rel      [encounter timescale at softening scale]
#   t_potential = T_orb / (2*pi)   [orbital timescale in background field]
# -------------------------------------------------------------------
eta_acc       = 0.05                                          # accuracy parameter
v_rel         = v_pert + sigma_R0                             # upper bound on relative speed
t_2body       = max(eps_recommended, eps_background) / v_rel  # use actual eps to be used
t_potential   = T_orb / (2.0 * math.pi)
dt_min        = min(t_2body, t_potential)
dt_recommended = eta_acc * dt_min

# snap to nearest power-of-two denominator  (1/512 ... 1/8192)
def nearest_power2_dt(dt_val):
    """Return (n, 1/n) where n = 2^k is the smallest power of two
    such that 1/n <= dt_val."""
    k = math.ceil(math.log2(1.0 / dt_val))
    k = max(9, min(k, 13))   # clamp to [1/512, 1/8192]
    return 2**k, 1.0 / 2**k

dtime_denom, dt_value = nearest_power2_dt(dt_recommended)

# -------------------------------------------------------------------
# Dynamical-friction timescale estimate (SIS lower bound, Task 4A):
#   t_DF ~ R0^2 * v_circ(R0) / (0.8 * G * M * ln(Lambda))
# ln(Lambda) ~ ln(M_tot / M) = ln(1/eta)
# -------------------------------------------------------------------
ln_lambda = math.log(1.0 / eta) if eta < 1.0 else 1.0
if ln_lambda > 0:
    t_DF_SIS = R0**2 * v_circ_R0 / (0.8 * G * M_pert * ln_lambda)
else:
    t_DF_SIS = float('inf')

# Orbital type label
if abs(vfrac - 1.0) < 1e-6:
    orbit_type = 'circular'
else:
    orbit_type = f'eccentric (apocenter, vfrac={vfrac:.3f})'

# ============================================================
#  RANDOM SEED
# ============================================================

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)

# ============================================================
#  PRINT DIAGNOSTICS
# ============================================================

print("=" * 65)
print("  PLUMMER SPHERE + PERTURBER — INITIAL CONDITIONS")
print("=" * 65)
print(f"  Code units: G=1, M_tot=1, b=1")
print()
print(f"  [BACKGROUND]")
print(f"    N particles         : {N}")
print(f"    particle mass m     : {mass_i:.6e}")
print(f"    scale radius b      : {b:.4f}")
print(f"    half-mass radius    : {r_hm:.4f}")
print(f"    central density     : {rho_c:.4f}")
print(f"    dynamical time t_dyn: {t_dyn:.4f}  (= pi/2)")
print(f"    mean separation     : {d_mean:.4f}  (near r_hm)")
print()
print(f"  [PERTURBER]")
print(f"    mass ratio eta      : {eta:.4f}")
print(f"    perturber mass M    : {M_pert:.6f}  ({eta:.4f} * M_tot)")
print(f"    M / m               : {M_pert/mass_i:.1f}  background masses")
print(f"    initial radius R0   : {R0:.4f}")
print(f"    orbit type          : {orbit_type}")
print(f"    enclosed mass M(<R0): {M_enc_R0:.4f}")
print(f"    v_circ(R0)          : {v_circ_R0:.4f}")
print(f"    v_perturber         : {v_pert:.4f}  ({vfrac:.3f} * v_circ)")
print(f"    sigma(R0)           : {sigma_R0:.4f}")
print(f"    orbital period T_orb: {T_orb:.4f}  ({T_orb/t_dyn:.2f} t_dyn)")
print()
print(f"  [INFLUENCE RADII]")
print(f"    r_inf  (sigma only) : {r_inf_sigma:.4f}  = G*M/sigma^2(R0)")
print(f"    r_inf  (corrected)  : {r_inf_corrected:.4f}  = G*M/(v^2+sigma^2)")
print(f"    r_inf / d_mean      : {r_inf_corrected/d_mean:.3f}")
print(f"    r_inf / eps_bg      : {r_inf_corrected/eps_background:.3f}")
print()
print(f"  [PARAMETER RECOMMENDATIONS  (from analytical derivations)]")
print(f"    eps (background)    : {eps_background:.4f}  ~ d_mean/10")
print(f"    eps (perturber)     : {eps_recommended:.4f}  = alpha*r_inf_corr  (alpha={alpha_eps})")
print(f"    eps to USE          : {max(eps_background, eps_recommended):.4f}  (max of the two)")
print(f"    t_2body (enc. time) : {t_2body:.6f}  = eps/v_rel")
print(f"    t_potential (orb)   : {t_potential:.4f}  = T_orb/(2*pi)")
print(f"    dt_min              : {dt_min:.6f}  = min(t_2body, t_potential)")
print(f"    dt recommended      : {dt_recommended:.6f}  = eta_acc * dt_min  (eta_acc={eta_acc})")
print(f"    dtime (treecode)    : 1/{dtime_denom}  = {dt_value:.6f}")
print()
print(f"  [TIMESCALE HIERARCHY]")
print(f"    t_dyn               : {t_dyn:.2f}")
print(f"    T_orb               : {T_orb:.2f}  = {T_orb/t_dyn:.1f} t_dyn")
print(f"    t_DF (SIS lower bd) : {t_DF_SIS:.2f}  = {t_DF_SIS/t_dyn:.1f} t_dyn")
print(f"    ln(Lambda) used     : {ln_lambda:.3f}  ~ ln(1/eta)")
print(f"    Collisionless regime: t_relax >> t_DF  (check t_stop < t_relax/5)")
if args.seed is not None:
    print(f"    Random seed         : {args.seed}")
else:
    print(f"    Random seed         : not fixed (new realisation)")
print("=" * 65)

# ============================================================
#  HELPER FUNCTIONS
# ============================================================

def isotropic_vec(mag):
    """Generate a random 3D vector with a given magnitude,
    uniformly distributed over the unit sphere."""
    u = random.random()
    w = random.random()
    theta = math.acos(1.0 - 2.0 * u)
    phi   = 2.0 * math.pi * w
    return (mag * math.sin(theta) * math.cos(phi),
            mag * math.sin(theta) * math.sin(phi),
            mag * math.cos(theta))


def get_q():
    """Rejection sampling for the dimensionless velocity q = v/v_esc.
    The target PDF is g(q) = q^2 * (1 - q^2)^(7/2)  for q in [0, 1].
    Peak at q* = sqrt(2/9), g(q*) = (2/9)*(7/9)^(7/2) ~ 0.0923.
    Envelope g_max = 0.093 is a small upward rounding that guarantees
    the envelope lies strictly above g(q) everywhere in [0, 1]."""
    g_max = 0.093
    while True:
        q = random.random()
        g = (q**2) * ((1.0 - q**2)**3.5)
        if random.random() * g_max < g:
            return q

# ============================================================
#  SAMPLE BACKGROUND PARTICLES
# ============================================================

print("Sampling background particles...", flush=True)

positions  = []
velocities = []

for _ in range(N):
    # --- Position: Inverse Transform Method ---
    # Plummer CDF: M(<r)/M_tot = r^3/(r^2+b^2)^(3/2) = X
    # Inversion:   r = b * (X^(-2/3) - 1)^(-1/2)
    X = random.random()
    r = b / math.sqrt(X**(-2.0 / 3.0) - 1.0)
    pos = isotropic_vec(r)
    positions.append(list(pos))

    # --- Velocity: Rejection Sampling ---
    # Local escape speed: v_esc = sqrt(2 * |Phi(r)|) = sqrt(2*G*M_tot/sqrt(r^2+b^2))
    Psi   = G * M_tot / math.sqrt(r**2 + b**2)   # = |Phi(r)| (specific binding energy)
    v_esc = math.sqrt(2.0 * Psi)
    q     = get_q()                               # dimensionless speed
    v     = q * v_esc
    vel   = isotropic_vec(v)
    velocities.append(list(vel))

positions  = np.array(positions)
velocities = np.array(velocities)

# ============================================================
#  CONSTRUCT PERTURBER PARTICLE
# ============================================================
# Placed at (R0, 0, 0) with velocity (0, v_pert, 0).
# This puts the initial angular momentum vector along +z.

pos_pert = np.array([R0, 0.0, 0.0])
vel_pert = np.array([0.0, v_pert, 0.0])

# ============================================================
#  CENTRE-OF-MASS CORRECTION  (full system: background + perturber)
# ============================================================
# The background alone has CM near zero (finite-N scatter).
# Adding the perturber at (R0, 0, 0) shifts the total CM significantly.
# We must subtract the total CM (mass-weighted) from all particles.

M_total_system = M_tot + M_pert  # total mass of the full system

# Background CM (unweighted mean, since all background masses are equal)
pos_cm_bg  = np.mean(positions,  axis=0)
vel_cm_bg  = np.mean(velocities, axis=0)

# Full-system CM (mass-weighted):
#   r_CM = (M_tot * r_CM_bg + M_pert * r_pert) / (M_tot + M_pert)
pos_cm_tot = (M_tot * pos_cm_bg + M_pert * pos_pert) / M_total_system
vel_cm_tot = (M_tot * vel_cm_bg + M_pert * vel_pert) / M_total_system

# Shift all particles
positions  -= pos_cm_tot
velocities -= vel_cm_tot
pos_pert   -= pos_cm_tot
vel_pert   -= vel_cm_tot

# Verify CM is now at origin (should be < 1e-12 after subtraction)
pos_cm_bg_check  = np.mean(positions, axis=0)
vel_cm_bg_check  = np.mean(velocities, axis=0)
pos_cm_sys_check = (M_tot * np.mean(positions, axis=0) + M_pert * pos_pert) / M_total_system
vel_cm_sys_check = (M_tot * np.mean(velocities, axis=0) + M_pert * vel_pert) / M_total_system

print(f"\n  CM correction applied.")
print(f"    CM shift (position) : {pos_cm_tot}")
print(f"    CM shift (velocity) : {vel_cm_tot}")
print(f"    Residual |CM_pos|   : {np.linalg.norm(pos_cm_sys_check):.2e}  (should be < 1e-12)")
print(f"    Residual |CM_vel|   : {np.linalg.norm(vel_cm_sys_check):.2e}  (should be < 1e-12)")

# Corrected perturber radius (after CM shift, slightly different from R0)
R0_actual = np.linalg.norm(pos_pert)
v0_actual = np.linalg.norm(vel_pert)
print(f"    Perturber R0 (post-CM): {R0_actual:.6f}  (was {R0:.6f})")
print(f"    Perturber v0 (post-CM): {v0_actual:.6f}  (was {v_pert:.6f})")

# ============================================================
#  QUICK SANITY CHECKS ON THE BACKGROUND ICS
# ============================================================

r_bg   = np.linalg.norm(positions, axis=1)
r_bg_s = np.sort(r_bg)

# Measured half-mass radius
idx_hm   = N // 2
r_hm_sim = r_bg_s[idx_hm - 1]

# Velocity dispersion
v_bg    = np.linalg.norm(velocities, axis=1)
sigma_v = math.sqrt(np.mean(v_bg**2))

# Theoretical 3D rms velocity from virial theorem: <v^2> = 3*pi/(32) * G*M/b
sigma_v_theory = math.sqrt(3.0 * math.pi / 32.0)   # ~ 0.543

print(f"\n  Background IC checks:")
print(f"    r_hm (sampled)      : {r_hm_sim:.4f}  (theory: {r_hm:.4f})")
print(f"    sigma_v (sampled)   : {sigma_v:.4f}  (theory: {sigma_v_theory:.4f})")
print(f"    N background        : {N}")
print(f"    N total (incl. pert): {N+1}")

# ============================================================
#  WRITE TREECODE INPUT FILE
# ============================================================
# Format (Barnes treecode):
#   N_total           <- total particle count
#   3                 <- spatial dimensions
#   0                 <- simulation start time
#   [N_total masses]  <- one per line (perturber first)
#   [N_total pos]     <- one x y z per line (perturber first)
#   [N_total vel]     <- one vx vy vz per line (perturber first)

outfile = args.out

print(f"\nWriting initial conditions to: {outfile}")

N_total = N + 1   # background + perturber

with open(outfile, 'w') as f:
    # Header
    f.write(f"{N_total}\n3\n0\n")

    # Masses: perturber first, then N background particles
    f.write(f"{M_pert:.8f}\n")
    for _ in range(N):
        f.write(f"{mass_i:.8f}\n")

    # Positions: perturber first
    f.write(f"{pos_pert[0]:.8f} {pos_pert[1]:.8f} {pos_pert[2]:.8f}\n")
    for p in positions:
        f.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")

    # Velocities: perturber first
    f.write(f"{vel_pert[0]:.8f} {vel_pert[1]:.8f} {vel_pert[2]:.8f}\n")
    for v in velocities:
        f.write(f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")

print(f"Done. File written: {outfile}")
print(f"  Total particles in file : {N_total}")
print(f"  Particle 0 (perturber)  : M={M_pert:.6f}, "
      f"r=({pos_pert[0]:.4f},{pos_pert[1]:.4f},{pos_pert[2]:.4f}), "
      f"v=({vel_pert[0]:.4f},{vel_pert[1]:.4f},{vel_pert[2]:.4f})")
print("=" * 65)

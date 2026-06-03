#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import math

# ---- Code Units / Parameters ----
G, M_tot, b, eta = 1.0, 1.0, 1.0, 0.01
M_p = eta * M_tot
N = 30000
mass_i = M_tot / N

def f_integrand(q):
    return (q**2) * ((1.0 - q**2)**3.5)

# ==========================================
# 1. TRAPEZOIDAL TABLE 
# ==========================================
q_fine = np.linspace(0, 1, 1000)
f_fine = f_integrand(q_fine)

# Visible trapezoidal grid (40 bins)
q_trap = np.linspace(0, 1, 41)
f_trap = f_integrand(q_trap)

# Exact cumulative area for the 10,000 array
q_real = np.linspace(0, 1, 10000)
f_real = f_integrand(q_real)
I_real = np.zeros(10000)
I_real[1:] = np.cumsum(0.5 * (f_real[:-1] + f_real[1:]) * (q_real[1] - q_real[0]))

# [FIX] Match the sampling script by normalizing with the numerical total area
exact_area = I_real[-1] 
B_cdf = I_real / exact_area

# ==========================================
# 2. EULER METHOD & t_DF
# ==========================================
def dRdt(R):
    # [FIX] Match the Coulomb logarithm from the sampling script
    Me = M_tot * R**3 / (R**2 + b**2)**1.5
    lnL = math.log(max(Me / M_p, 1.1))
    
    v_c = math.sqrt(G * Me / R) if R > 0 else 0.0
    v_e = math.sqrt(2.0 * G * M_tot / math.sqrt(R**2 + b**2))
    q_val = min(v_c / v_e, 0.9999) if v_e > 0 else 0.0
    
    B = np.interp(q_val, q_real, B_cdf)
    
    return -6.0 * (M_p + mass_i) * lnL * B / R**2 * (1.0 + R**2)**0.75 / (4.0 + R**2)

# [FIX] Match R0 exactly to the half-mass radius used in the sampling script
R0 = b / math.sqrt(2.0 ** (2.0 / 3.0) - 1.0)
R_low = 0.05
dt = 0.01

t_euler, R_euler = [0.0], [R0]

# [FIX] Increased the time limit from 200 to 2000 to allow the inspiral to finish
while R_euler[-1] > R_low and t_euler[-1] < 2000:
    # [FIX] Added max(0.001, ...) to match the sampling script's safety bound
    R_next = max(0.001, R_euler[-1] + dRdt(R_euler[-1]) * dt)
    t_euler.append(t_euler[-1] + dt)
    R_euler.append(R_next)

t_DF_true = t_euler[-1]

# ==========================================
# TERMINAL OUTPUT
# ==========================================
print("\n--- 1. TRAPEZOIDAL INTEGRATION ---")
print(f"Computed Area (10k)   : {I_real[-1]:.6f}")

print("\n--- 2. EULER METHOD ---")
print(f"Initial Radius (R0)   : {R0:.4f}")
print(f"Stopping Bound (R_low): {R_low:.3f}")
print(f"Time Step (dt)        : {dt}")
print(f"Euler Steps Taken     : {len(t_euler) - 1}")
print(f"Final Calculated t_DF : {t_DF_true:.2f}")

# ==========================================
# PLOTTING
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# PANEL 1: Trapezoidal Integration
ax = axes[0]
ax.plot(q_fine, f_fine, 'k-', lw=1.5, label=r"$f(q)$")
ax.fill_between(q_trap, 0, f_trap, facecolor='orange', edgecolor='black', 
                alpha=0.5, lw=1.0, label="Trapezoids")
ax.set_title("Trapezoidal Integration")
ax.set_xlabel(r"$q$")
ax.set_ylabel(r"$f(q)$")
ax.legend(frameon=False)
ax.grid(True, alpha=0.3)

# PANEL 2: Euler t_DF
ax = axes[1]
ax.plot(t_euler, R_euler, 'b-', lw=2, label=r"$R(t)$")
ax.axhline(R_low, color='r', ls='--', lw=1.5, label=r"$R_{low}$")
ax.axvline(t_DF_true, color='k', ls=':', lw=1.5)
ax.plot(t_DF_true, R_low, 'ko', markersize=5, zorder=5)
ax.text(t_DF_true + 3, R_low + 0.05, f"$t_{{DF}} = {t_DF_true:.1f}$", fontsize=11)
ax.set_title("Euler Integration")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$R$")
ax.legend(frameon=False)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig("plot_methods.pdf")
print("saved 'plot_methods.pdf'\n")

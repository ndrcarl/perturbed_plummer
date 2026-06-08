#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import math

# ---- Code Units / Parameters ----
G, M_tot, b, eta = 1.0, 1.0, 1.0, 0.01
M_p    = eta * M_tot
N      = 30000
mass_i = M_tot / N

# ==========================================
# 1. PLUMMER BRACKET — trapezoidal CDF table
# ==========================================
def f_integrand(q):
    return (q**2) * ((1.0 - q**2)**3.5)

# Fine grid for the CDF table (10 000 points)
q_real = np.linspace(0, 1, 10000)
f_real = f_integrand(q_real)
I_real = np.zeros(10000)
I_real[1:] = np.cumsum(0.5 * (f_real[:-1] + f_real[1:]) * (q_real[1] - q_real[0]))
exact_area = I_real[-1]          # = 7*pi/512 ~ 0.04294
B_plum_cdf = I_real / exact_area # normalised CDF: B_Plummer(q_M)

# Coarse grid for the visible trapezoids (40 bins)
q_trap = np.linspace(0, 1, 41)
f_trap = f_integrand(q_trap)

# Fine grid for the smooth curve overlay
q_fine = np.linspace(0, 1, 1000)
f_fine = f_integrand(q_fine)

# ==========================================
# 2. MAXWELLIAN BRACKET — trapezoidal CDF table
#    Integrand: u^2 * exp(-u^2),  u = v / (sqrt(2) * sigma)
#    B_Maxwell(X) = (4/sqrt(pi)) * integral_0^X u^2 exp(-u^2) du
#    Table truncated at u=5 (B(5) = 1 to < 1e-10).
# ==========================================
u_real  = np.linspace(0, 5, 50000)
g_real  = u_real**2 * np.exp(-u_real**2)
I_maxw  = np.zeros(50000)
I_maxw[1:] = np.cumsum(0.5 * (g_real[:-1] + g_real[1:]) * (u_real[1] - u_real[0]))
I_maxw_full = I_maxw[-1]         # = sqrt(pi)/4 ~ 0.44311
B_maxw_cdf  = I_maxw / I_maxw_full  # normalised CDF: B_Maxwell(X)

# ==========================================
# 3. ODE / EULER METHOD
#    Uses lnL floored at ln(1.0) = 0, which cleanly switches off friction
#    when M(<R) <= M_p  (Chandrasekhar formula breaks down at that scale).
# ==========================================
def dRdt(R):
    Me  = M_tot * R**3 / (R**2 + b**2)**1.5
    lnL = math.log(max(Me / M_p, 1.0))   # floor at ln(1) = 0, not 1.1
    if lnL <= 0.0:
        return 0.0
    v_c   = math.sqrt(G * Me / R) if R > 0 else 0.0
    v_e   = math.sqrt(2.0 * G * M_tot / math.sqrt(R**2 + b**2))
    q_val = min(v_c / v_e, 0.9999) if v_e > 0 else 0.0
    B     = float(np.interp(q_val, q_real, B_plum_cdf))
    return -6.0*(M_p + mass_i)*lnL*B / R**2 * (1.0 + R**2)**0.75 / (4.0 + R**2)

R0    = b / math.sqrt(2.0**(2.0/3.0) - 1.0)
# Stall radius: M(<R_stall) = M_p — the point where Chandrasekhar friction
# switches off (lnL -> 0).  Physical stopping condition for the Euler loop.
# Solved by bisection on M(<r) = r^3/(r^2+b^2)^{3/2} = M_p.
_lo, _hi = 1e-4, R0
for _ in range(100):
    _mid = 0.5 * (_lo + _hi)
    if M_tot * _mid**3 / (_mid**2 + b**2)**1.5 < M_p:
        _lo = _mid
    else:
        _hi = _mid
R_stall = 0.5 * (_lo + _hi)
R_low   = R_stall
dt    = 0.01
t_euler, R_euler = [0.0], [R0]
while R_euler[-1] > R_stall * (1.0 + 1e-4) and t_euler[-1] < 2000:
    R_next = R_euler[-1] + dRdt(R_euler[-1]) * dt
    t_euler.append(t_euler[-1] + dt)
    R_euler.append(max(R_stall, R_next))
t_DF_true = t_euler[-1]

# ---- operating point at R0 ----
Me_0  = M_tot * R0**3 / (R0**2 + b**2)**1.5
vc_0  = math.sqrt(G * Me_0 / R0)
ve_0  = math.sqrt(2.0 * G * M_tot / math.sqrt(R0**2 + b**2))
sig_0 = math.sqrt(G * M_tot / (6.0 * math.sqrt(R0**2 + b**2)))
q_M0  = min(vc_0 / ve_0, 0.9999)
X_0   = vc_0 / (math.sqrt(2.0) * sig_0)
B_P0  = float(np.interp(q_M0, q_real,  B_plum_cdf))
B_M0  = float(np.interp(X_0,  u_real,  B_maxw_cdf))

# ==========================================
# TERMINAL OUTPUT
# ==========================================
print("\n--- 1. TRAPEZOIDAL INTEGRATION ---")
print(f"Plummer area  (10k):  {exact_area:.6f}   (exact 7pi/512 = {7*math.pi/512:.6f})")
print(f"Maxwell norm  (50k):  {I_maxw_full:.6f}  (exact sqrt(pi)/4 = {math.pi**0.5/4:.6f})")
print("\n--- 2. OPERATING POINT AT R0 ---")
print(f"R0         = {R0:.4f}")
print(f"q_M(R0)    = {q_M0:.4f}   (v_c / v_esc)")
print(f"X(R0)      = {X_0:.4f}   (v_c / sqrt(2)*sigma)")
print(f"B_Plummer  = {B_P0:.4f}")
print(f"B_Maxwell  = {B_M0:.4f}")
print("\n--- 3. EULER METHOD ---")
print(f"R_stall    = {R_stall:.4f}  (M(<R_stall) = M_p, friction off)")
print(f"R_low      = {R_low:.4f}  (= R_stall)")
print(f"dt         = {dt}")
print(f"t_DF       = {t_DF_true:.2f}")

# ==========================================
# PLOTTING — 1x3 figure
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# ---------- Panel 1: trapezoidal integration (Plummer) ----------
ax = axes[0]
ax.plot(q_fine, f_fine / exact_area, "k-", lw=1.5,
        label=r"$f(q) = q^2(1-q^2)^{7/2}\,/\,I_\mathrm{full}$")
ax.fill_between(q_trap, 0, f_integrand(q_trap) / exact_area,
                facecolor="orange", edgecolor="black", alpha=0.5, lw=1.0,
                label=f"trapezoids  (40 bins, area$={exact_area:.4f}$)")
ax.axvline(q_M0, color="b", ls="--", lw=1.2,
           label=rf"$q_M(R_0) = {q_M0:.2f}$  (circ. orbit)")
ax.set_title("Plummer speed distribution")
ax.set_xlabel(r"$q = v\,/\,v_\mathrm{esc}(r)$")
ax.set_ylabel(r"normalised PDF")
ax.legend(frameon=False, fontsize=8)
ax.grid(True, alpha=0.3)

# ---------- Panel 2: bracket CDFs ----------
ax = axes[1]
ax.plot(q_real, B_plum_cdf, color="b", lw=1.5,
        label=r"$B_\mathrm{Plummer}(q)$,  $q = v/v_\mathrm{esc}$")
ax.plot(u_real[u_real <= 3.0], B_maxw_cdf[u_real <= 3.0],
        color="r", lw=1.5, ls="--",
        label=r"$B_\mathrm{Maxwell}(X)$,  $X = v/\sqrt{2}\,\sigma$")
ax.axvline(q_M0, color="b", ls=":", lw=0.9, alpha=0.8)
ax.axvline(X_0,  color="r", ls=":", lw=0.9, alpha=0.8)
ax.scatter([q_M0, X_0], [B_P0, B_M0], color=["b", "r"], s=50, zorder=5,
           label=rf"at $R_0$:  $B_P={B_P0:.2f}$,  $B_M={B_M0:.2f}$")
ax.axvline(1.0, color="b", ls="--", lw=0.7, alpha=0.4,
           label=r"$q=1$: Plummer support ends")
ax.set_title("bracket CDFs")
ax.set_xlabel(r"velocity parameter  ($q$  or  $X$)")
ax.set_ylabel(r"$B$  (fraction of slower particles)")
ax.set_xlim(0, 3.0)
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, fontsize=8)
ax.grid(True, alpha=0.3)

# ---------- Panel 3: Euler inspiral ----------
ax = axes[2]
ax.plot(t_euler, R_euler, "b-", lw=2, label=r"$R(t)$  (Plummer DF)")
ax.axhline(R_low, color="r", ls="--", lw=1.5, label=r"$R_\mathrm{low}$")
ax.axvline(t_DF_true, color="k", ls=":", lw=1.5)
ax.plot(t_DF_true, R_low, "ko", markersize=5, zorder=5)
ax.text(t_DF_true + 5, R_low + 0.05, f"$t_{{DF}} = {t_DF_true:.1f}$", fontsize=11)
ax.set_title("Euler inspiral")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$R$")
ax.legend(frameon=False, fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig("plot_methods_clean.pdf", bbox_inches="tight")
print("\nsaved 'plot_methods_clean.pdf'")

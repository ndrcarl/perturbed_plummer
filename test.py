#!/usr/bin/env python3
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Safe for remote servers like Argo
import matplotlib.pyplot as plt
import os
import argparse
import sys

def parse_treecode_out(filepath):
    """
    Correctly parses the Barnes treecode .out file, which has 4 blocks per snapshot:
    Mass, Position, Velocity, and Potential (phi).
    """
    snapshots = []
    
    if not os.path.exists(filepath):
        print(f"ERROR: File '{filepath}' not found.")
        return snapshots
        
    print(f"Opening {filepath} and reading all snapshots...")
    with open(filepath, "r") as fh:
        while True:
            line = fh.readline()
            if not line:
                break  # End of file
            
            line = line.strip()
            if not line:
                continue
            
            # 1. Read N
            try:
                N_tot = int(line)
            except ValueError:
                continue
                
            # 2. Read ndim and time
            ndim = int(fh.readline().strip())
            t_now = float(fh.readline().strip())
            
            # 3. Read arrays
            masses = np.array([float(fh.readline().strip()) for _ in range(N_tot)])
            pos = np.array([[float(x) for x in fh.readline().split()] for _ in range(N_tot)])
            vel = np.array([[float(x) for x in fh.readline().split()] for _ in range(N_tot)])
            phi = np.array([float(fh.readline().strip()) for _ in range(N_tot)])  # Must read phi to advance lines
            
            snapshots.append({
                'time': t_now,
                'm': masses,
                'pos': pos,
                'vel': vel,
                'phi': phi
            })
            
    print(f"Successfully loaded {len(snapshots)} snapshots!")
    return snapshots


# ---------- Density binning function (identical to summary_perturber.py) ----------
def bin_density_profile(pos, m_bg, n_bins=35):
    """
    Compute radial density profile from particle positions.
    Returns (r_mid, density, counts) for bins where count > 0.
    """
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Path to the plummer_perturber.out file")
    args = parser.parse_args()

    snapshots = parse_treecode_out(args.input)

    if len(snapshots) < 2:
        print("Not enough snapshots to plot a trajectory. Did the simulation finish?")
        sys.exit(1)

    times = []
    radii = []
    circularities = []
    reflex_distances = []
    v_radials = []
    v_tangentials = []
    x_traj = []
    y_traj = []

    a_plummer = 1.0  # b=1.0 in the generator script
    M_tot = 1.0      # total mass of background (code units)
    
    for snap in snapshots:
        # PERTURBER IS INDEX 0
        pos_p_abs = snap['pos'][0]
        vel_p_abs = snap['vel'][0]
        
        # BACKGROUND STARS ARE INDICES 1 TO N
        mass_stars = snap['m'][1:]
        pos_stars = snap['pos'][1:]
        vel_stars = snap['vel'][1:]
        
        # Compute instantaneous barycenter of host background stars
        cm_stars = np.average(pos_stars, axis=0, weights=mass_stars)
        cm_vel_stars = np.average(vel_stars, axis=0, weights=mass_stars)
        
        # Center perturber kinematics onto the live host frame
        pos_p = pos_p_abs - cm_stars
        vel_p = vel_p_abs - cm_vel_stars
        
        r_mag = np.linalg.norm(pos_p)
        v_mag = np.linalg.norm(vel_p)
        
        # Specific angular momentum and target circular profile
        J_vec = np.cross(pos_p, vel_p)
        J_mag = np.linalg.norm(J_vec)
        v_c = np.sqrt(r_mag**2 / (r_mag**2 + a_plummer**2)**1.5)
        J_circ = r_mag * v_c
        eta = J_mag / J_circ if J_circ > 0 else 0
        
        # Velocity space decomposition
        v_r = np.dot(vel_p, pos_p) / r_mag if r_mag > 0 else 0
        v_t = np.sqrt(np.maximum(0, v_mag**2 - v_r**2))
        
        # Track core displacement (reflex motion)
        r_reflex = np.linalg.norm(cm_stars)
        
        times.append(snap['time'])
        radii.append(r_mag)
        circularities.append(eta)
        v_radials.append(v_r)
        v_tangentials.append(v_t)
        reflex_distances.append(r_reflex)
        x_traj.append(pos_p[0])
        y_traj.append(pos_p[1])

    # Convert to numpy arrays
    times = np.array(times)
    radii = np.array(radii)
    circularities = np.array(circularities)
    reflex_distances = np.array(reflex_distances)
    v_radials = np.array(v_radials)
    v_tangentials = np.array(v_tangentials)
    x_traj = np.array(x_traj)
    y_traj = np.array(y_traj)
        
    # ---------- Density profile from the final snapshot (using summary_perturber.py method) ----------
    final = snapshots[-1]
    pos_bg = final['pos'][1:]            # exclude perturber
    mass_bg = final['m'][1]              # single particle mass (all equal)
    N_bg = len(pos_bg)
    
    # Compute density profile using the exact binning function from summary_perturber.py
    r_mid, rho_sim, cnts = bin_density_profile(pos_bg, mass_bg, n_bins=35)
    
    # Theoretical Plummer density (M_tot = 1, b = 1)
    rho_th = (3.0 * M_tot) / (4.0 * np.pi * a_plummer**3) * (1 + (r_mid / a_plummer)**2)**(-2.5)
    
    print("Generating plots...")
    # --- PLOTTING ANALYSIS PANELS (3 rows, 3 columns) ---
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))

    # (0,0) Orbital decay
    axs[0, 0].plot(times, radii, 'b-', lw=2.5, label='Simulation')
    axs[0, 0].axhline(y=a_plummer, color='gray', linestyle=':', label='Core Scale Radius $a$')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel(r'Orbital Separation $r_p(t)$')
    axs[0, 0].set_title('Orbital Decay via Dynamical Friction')
    axs[0, 0].legend()
    axs[0, 0].grid(True, ls='--')

    # (0,1) Circularity
    axs[0, 1].plot(times, circularities, 'g-', lw=2.5)
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel(r'Circularity $\eta = J / J_{\mathrm{circ}}$')
    axs[0, 1].set_title('Orbital Radialization')
    axs[0, 1].grid(True, ls='--')

    # (0,2) Velocity components vs time
    axs[0, 2].plot(times, v_tangentials, 'k-', label='$v_t$', lw=2)
    axs[0, 2].plot(times, v_radials, 'k--', label='$v_r$', lw=2)
    axs[0, 2].set_xlabel('t')
    axs[0, 2].set_ylabel('v')
    axs[0, 2].set_title('v vs t')
    axs[0, 2].legend()
    axs[0, 2].grid(True, ls='--')

    # (1,0) Reflex motion
    axs[1, 0].plot(times, reflex_distances, 'm-', lw=2.5)
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel(r'Core Displacement $|R_{\mathrm{cm}}|$')
    axs[1, 0].set_title('Global Host Response: Reflex Motion')
    axs[1, 0].grid(True, ls='--')

    # (1,1) Kinematic velocity phase space
    axs[1, 1].plot(v_radials, v_tangentials, color='purple', lw=2)
    axs[1, 1].scatter(v_radials[0], v_tangentials[0], color='green', s=100, label='Initial', zorder=5)
    axs[1, 1].scatter(v_radials[-1], v_tangentials[-1], color='red', s=100, label='Final', zorder=5)
    axs[1, 1].set_xlabel(r'Radial Velocity $v_r$')
    axs[1, 1].set_ylabel(r'Tangential Velocity $v_t$')
    axs[1, 1].set_title('Kinematic Velocity Phase Space')
    axs[1, 1].legend()
    axs[1, 1].grid(True, ls='--')

    # (1,2) Perturber orbit (x-y)
    ax_traj = axs[1, 2]
    sc = ax_traj.scatter(x_traj, y_traj, c=times, s=3, cmap='viridis')
    ax_traj.set_aspect('equal')
    ax_traj.set_xlabel('x')
    ax_traj.set_ylabel('y')
    ax_traj.set_title('Perturber Orbit')
    plt.colorbar(sc, ax=ax_traj, label='Time')
    ax_traj.grid(True, ls='--', alpha=0.5)

    # (2,1) Density profile (final snapshot) – using same style as summary_perturber.py
    ax_dens = axs[2, 1]   # bottom center
    ax_dens.errorbar(
        r_mid, rho_sim,
        yerr=[rho_sim * (1.0 - 10.0 ** (-1.0 / np.sqrt(cnts))),
              rho_sim * (10.0 ** (1.0 / np.sqrt(cnts)) - 1.0)],
        fmt='o', ms=3, color='k', label='N-body (final)'
    )
    ax_dens.loglog(r_mid, rho_th, 'r--', label='Plummer theory')
    ax_dens.set_xlabel('r')
    ax_dens.set_ylabel(r'$\rho(r)$')
    ax_dens.set_title('Density')
    ax_dens.legend()
    ax_dens.grid(True, ls='--', alpha=0.5)

    # Hide empty subplots (2,0) and (2,2) for a cleaner look
    axs[2, 0].axis('off')
    axs[2, 2].axis('off')

    plt.tight_layout()
    out_img = "plots.pdf"
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"Done! Plots saved as {out_img}")

if __name__ == "__main__":
    main()

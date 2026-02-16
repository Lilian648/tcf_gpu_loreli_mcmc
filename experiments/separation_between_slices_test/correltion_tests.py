import sys
print("Which env am i in?", sys.executable)

import os
import re
import time
import copy as cp
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
# matplotlib.rcParams['text.usetex'] = False          # don’t call system LaTeX
# matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # nice default font

# import seaborn as sns
import pandas as pd
# import zeus

from scipy.interpolate import PchipInterpolator, CubicSpline


import warnings
from types import SimpleNamespace
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
from catwoman.shelter import Cat

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

base_dir = '/data/cluster/emc-brid/Datasets/LoReLi'
path_sim = '/data/cluster/emc-brid/Datasets/LoReLi/simcubes'

# ---- settings ----
sim_names = ['10038','10501','11557','12091','16995','17338','18886']

N = 256
L = 296.0
deltax = L / N
x = np.arange(N//2) * deltax

# ---- load sims ----
sim_list = []
for name in sim_names:
    sim = Cat(
        name,
        skip_early=False, path_spectra='spectra', path_sim=path_sim, base_dir=base_dir,
        load_params=False, load_spectra=True, just_Pee=True,
        reinitialise_spectra=False, save_spectra=False,
        load_density_cubes=False, load_xion_cubes=False,
        load_T21cm_cubes=True, use_LoReLi_xe=False, verbose=True
    )
    sim_list.append(sim)

# ---- determine common redshift index range ----
n_cubes_each = [len(sim.T21cm) for sim in sim_list]
max_common_idx = min(n_cubes_each) - 1

# choose indices every 5, but only up to what all sims have
idx_step = 5
idx_list = list(range(0, max_common_idx + 1, idx_step))

print("T21cm cube counts per sim:", dict(zip(sim_names, n_cubes_each)))
print("Max common idx:", max_common_idx)
print("Using idx_list:", idx_list)

# ---- compute correlations ----
total_corr = np.zeros((len(sim_list), len(idx_list), N//2), dtype=np.float32)

for isim, sim in enumerate(sim_list):
    print(f"\n===== Sim {sim_names[isim]} ({isim+1}/{len(sim_list)}) =====")

    for iz, idx in enumerate(idx_list):
        print(f"  -> idx {idx} ({iz+1}/{len(idx_list)})")

        cube = sim.T21cm[idx].astype(np.float64, copy=False)
        cube = cube - np.mean(cube)

        corr = np.zeros(N//2, dtype=np.float64)
        for n in range(N//2):
            diag = np.array([cube[j] * np.take(cube, j+n, axis=0, mode='wrap') for j in range(N)])
            corr[n] = np.mean(diag)

        total_corr[isim, iz, :] = corr.astype(np.float32)

# ---- save ----
out_fn = "/home/lcrascal/Code/TCF/TCF_completed_code/TCF_python_loreli/DecisionCatalogue_results/separation_between_slices/slice_corr_LoReLi_T21cm.npz"
np.savez(
    out_fn,
    sim_names=np.array(sim_names),
    idx_list=np.array(idx_list),
    x=x,
    L=L,
    N=N,
    total_corr=total_corr
)
print(f"\nSaved to {out_fn}")

# ----------------------------------- #
# --------------- plot -------------- #
# ----------------------------------- #
d = np.load(out_fn, allow_pickle=True)

x = d["x"]
idx_list = d["idx_list"]
total_corr = d["total_corr"]  # (nsims, nidx, ns)

# total_corr shape: (nsims, nidx, ns)
V0 = total_corr[:, :, 0]                 # (nsims, nidx)

rho_all = total_corr / (V0[:, :, None] )   # (nsims, nidx, ns)

rho_mean = np.mean(rho_all, axis=0)             # (nidx, ns)
rho_std  = np.std(rho_all, axis=0, ddof=1)      # (nidx, ns)

fig, ax = plt.subplots(figsize=(9, 8))

norm = Normalize(vmin=idx_list.min(), vmax=idx_list.max())
cmap = plt.cm.viridis
sm = ScalarMappable(norm=norm, cmap=cmap)

for iz, idx in enumerate(idx_list):
    color = cmap(norm(idx))
    y = rho_mean[iz]         # percent
    yerr = rho_std[iz]  # percent (approx)

    ax.plot(x, y, lw=2, color=color)
    ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)

ax.axhline(0, color="k", ls="--", lw=1)
ax.axhline(1, color="k", ls="--", lw=1) 
ax.axhline(0.05, color="k", ls="-.", lw=1, label="<5% correlation") 
ax.set_xlabel(r"Slice separation $s$ [Mpc]", fontsize=18)
ax.set_ylabel(r"Correlation $\rho(s)$ [%]", fontsize=18)
ax.set_xlim(0, 148)
#ax.set_ylim(0, 1)

cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Redshift index", fontsize=12)

fig.tight_layout()
plt.legend()
fig.savefig("slice_correlations_plot.png", dpi=200, bbox_inches="tight")
plt.show()
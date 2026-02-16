from TCF_pipeline_GPU import pyTCF_of_2Dslice_GPU
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
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import PchipInterpolator, CubicSpline
import warnings
from types import SimpleNamespace
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
from catwoman.shelter import Cat

from pathlib import Path
import re, subprocess
import math
import matplotlib.patheffects as pe
import pickle as pkl
import time
import h5py
import json
import warnings

import tools21cm as t2c
import tools21cm.cosmo as cm

# gpu imports
from batch_tcf import generate_kq_pairs_batches_gpu, compute_gpu_norms, tcf_partial_vectorized_gpu, estimate_bispectrum_gpu
import cupy as cp


r_defs = [
    {"r_min": 2, "r_max": 50, "Nr": 49},
    {"r_min": 2, "r_max": 50, "Nr": 25},
    {"r_min": 5, "r_max": 50, "Nr": 49},
    {"r_min": 8, "r_max": 50, "Nr": 49},
    {"r_min": 2, "r_max": 35, "Nr": 49},
    {"r_min": 8, "r_max": 35, "Nr": 49},
    {"r_min": 8, "r_max": 35, "Nr": 13},
]
# # to check
# r_defs = [{"r_min": 10, "r_max": 20, "Nr": 10}]

r_list = [
    np.linspace(d["r_min"], d["r_max"], d["Nr"])
    for d in r_defs
]


# Open sim slice

slice2D = np.loadtxt("clean_sim_slice_256x256.txt")
L = 296 # Mpc


# compute TCF of a single slice

sr_output = []
r_used_output = []
times_list = []

for index, rvals in enumerate(r_list):

    print(f"\nRunning GPU TCF for r-set {index}: "
          f"[{rvals.min():.1f}, {rvals.max():.1f}], Nr={len(rvals)}")

    time_begin = time.time()
    sr, r_used = pyTCF_of_2Dslice_GPU(slice2D, L, rvals=rvals)
    time_end = time.time()

    time_taken = time_end - time_begin

    sr_output.append(sr)
    r_used_output.append(r_used)
    times_list.append(time_taken)

    print(f"  Time taken: {time_taken:.2f} s")
    print("r_used", r_used)


# save results 
sr_output = np.array(sr_output, dtype=object)
r_used_output = np.array(r_used_output, dtype=object)

# table
rows = []
for i, d in enumerate(r_defs):
    rows.append([
        i,
        d["r_min"],
        d["r_max"],
        d["Nr"],
        f"{times_list[i]:.2f}"
    ])


col_labels = ["Set", "r_min", "r_max", "Nr", "Time [s]"]
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 0.6 * len(rows)))
ax.axis("off")

table = ax.table(
    cellText=rows,
    colLabels=col_labels,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

plt.tight_layout()
plt.savefig("tcf_gpu_rvals_timing_table.png", dpi=300, bbox_inches="tight")
plt.close()



# save results
np.savez(
    "tcf_gpu_single_slice_tests.npz",
    L=float(L),
    slice_shape=np.array(slice2D.shape),
    r_defs=np.array(r_defs, dtype=object),
    r_used=r_used_output,
    sr=sr_output,
    times=np.array(times_list, dtype=float),
    allow_pickle=True
)



# # load dictionary later
# data = np.load("tcf_gpu_single_slice_tests.npz", allow_pickle=True)

# sr = data["sr"]
# r_used = data["r_used"]
# times = data["times"]





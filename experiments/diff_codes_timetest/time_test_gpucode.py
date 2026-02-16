
import os
import re, subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt


slice_path = Path("loreli_slice.txt")
slice2D = np.loadtxt(slice_path) 
L = 296
N = np.shape(slice2D)[0]
nbins = 49
rmin = 2
rmax = 50
rvals = np.linspace(2, 50, 49)


from TCF_pipeline_GPU_timetest import pyTCF_of_2Dslice_GPU

t0 = time.time()
nmodes, sr, r_used = pyTCF_of_2Dslice_GPU(slice2D, L, rvals=rvals, outfile=None)
t1 = time.time()

time_total = t1-t0
print("time_total", time_total)


# ---- save results to txt ----
out_txt = Path("tcf_res_GPU.txt")

header = (
    f"TCF of single 2D slice\n"
    f"time_taken_sec = {t1 - t0:.3f}\n"
    f"L = {L}, N = {N}, nbins = {nbins}, rmin = {rmin}, rmax = {rmax}\n"
    f"columns: r  Re_s_r\n"
)

data = np.column_stack([
    r_used,
    sr,
])

np.savetxt(out_txt, data, header=header, comments="# ")

print(f"Saved TCF results to {out_txt.resolve()}")

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



################ python code ################

from Compute_TCF_Pipeline import pyTCF_of_2Dslice

t0 = time.time()
nmodes, sr_vals, rvals = pyTCF_of_2Dslice(slice2D, L, rvals, outfile=None)
t1 = time.time()

time_total = t1-t0

print("time_total", time_total)


# ---- save results to txt ----
out_txt = Path("tcf_res_python.txt")

header = (
    f"TCF of single 2D slice\n"
    f"time_taken_sec = {t1 - t0:.3f}\n"
    f"L = {L}, N = {N}, nbins = {nbins}, rmin = {rmin}, rmax = {rmax}\n"
    f"columns: r  Re_s_r\n"
)

data = np.column_stack([
    rvals,
    sr_vals,
])

np.savetxt(out_txt, data, header=header, comments="# ")

print(f"Saved TCF results to {out_txt.resolve()}")


# import sys
# from pathlib import Path
# # Add project root to Python path
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# sys.path.insert(0, str(PROJECT_ROOT))

# import os
# import re, subprocess
# import pandas as pd
# import numpy as np
# import time
# import matplotlib.pyplot as plt
# import argparse
# from scipy.interpolate import PchipInterpolator, CubicSpline
# import warnings
# from types import SimpleNamespace
# from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
# import matplotlib.patheffects as pe
# import pickle as pkl
# import h5py
# import json
# import tools21cm as t2c
# import tools21cm.cosmo as cm

# ########### Global Parameters + Data ###############################################################################################################

# # --- loreli file  --- #
# tcf_code_dir = Path("/home/lcrascal/Code/TCF/TCF_completed_code/TCF_required_files")          # folder with Makefile, SC.h, SC_2d.o target
# output_dir   = Path("./")

# slice_path = Path("loreli_slice.txt")
# slice2D = np.loadtxt(slice_path)
# plt.imshow(slice2D)
# plt.colorbar()
# plt.show()
# plt.savefig("loreli_slice.png")
# plt.close()

# # --- parameters --- #
# L = 296
# N = np.shape(slice2D)[0]

# # for rvals
# nbins    = 49
# rmin     = 2
# rmax     = 50
# rvals = np.linspace(2, 50, 49)
# ndim = 2

# ####################################################################################################################################################
# ########### 1. C++ code ############################################################################################################################
# ####################################################################################################################################################

# # print("%%%%%%%%%% C++ Code %%%%%%%%%")
# # import TCF_Class
# # #from tcf.TCF_Class import Compute_TCF



# # # ---  TCF params --- #
# # nthreads = 5

# # # --- run --- #
# # t1 = time.time()

# # tcf = TCF_Class.Compute_TCF(
# #     tcf_code_dir=str(tcf_code_dir),
# #     L=L, DIM=N,
# #     nthreads=nthreads, nbins=nbins, rmin=rmin, rmax=rmax
# # )


# # cplus_df = tcf.compute_TCF_of_single_field(str(slice_path))
# # sr_cplus = cplus_df["Re_s_r"]

# # t2 = time.time()
# # time_taken_cplus = t2-t1

# # print(f"time taken c++ = {time_taken_cplus}")

# ####################################################################################################################################################
# ########### 2. python code #########################################################################################################################
# ####################################################################################################################################################
# # print("%%%%%%%%%% python Code %%%%%%%%%")
# # INEED the CAtWOMAN ENVIRONMENT

# # from Compute_TCF_Pipeline import pyTCF_of_2Dslice


# # t1 = time.time()
# # nmodes, sr_py, rvals = pyTCF_of_2Dslice(slice2D, L, rvals, outfile=None)
# # t2 = time.time()
# # time_taken_py = t2-t1

# # print(f"time taken py = {time_taken_py}")

# ####################################################################################################################################################
# ########### 3. GPU py code #########################################################################################################################
# ####################################################################################################################################################

# # print("%%%%%%%%%% GPU Code %%%%%%%%%")

# # from TCF_pipeline_GPU import pyTCF_of_2Dslice_GPU
# # from batch_tcf import generate_kq_pairs_batches_gpu, compute_gpu_norms, tcf_partial_vectorized_gpu, estimate_bispectrum_gpu
# # import cupy as cp

# # t1 = time.time()
# # nmodes, sr_gpu, rvals = pyTCF_of_2Dslice_GPU(slice2D, L, rvals=rvals, outfile=None)
# # t2 = time.time()
# # time_taken_gpu = t2-t1

# # print(f"time taken gpu = {time_taken_gpu}")

# ####################################################################################################################################################
# ########### 4. CPU py code #########################################################################################################################
# ####################################################################################################################################################

# print("%%%%%%%%%% CPU Code %%%%%%%%%")


# # Compute the TCF with batch approach to avoid memory overload

# t1 = time.time()

# field_gpu = np.array(slice2D, dtype=np.float32)
# field_k = np.fft.fftn(field_gpu)
# m = np.abs(field_k) < 2. * np.pi / L
# abs_field_k = np.abs(field_k)
# epsilon_k = np.where(m, 0, field_k / abs_field_k)
# kcoord = np.fft.fftfreq(N, d=L/N/2./np.pi)
# # batches
# n_samples = n**(ndim*2)
# batch_size = 1000000
# print("batch_size", batch_size)
# tcf_sums = np.zeros_like(rlin)
# u = 0
# for pairs in tqdm(generate_kq_pairs_batches_gpu(N, ndim, batch_size=batch_size), total=n_samples//batch_size):
#     Bk_batch = estimate_bispectrum_gpu(epsilon_k, N, pairs, ndim)
#     k_norms, q_norms, p_norms = compute_gpu_norms(pairs, ndim, kcoord)
#     s_v = tcf_partial_vectorized_gpu(rlin, Bk_batch.real, k_norms, q_norms, p_norms, L, ndim)
#     tcf_sums += s_v
#     u += 1
    
# sr_cpu = tcf_sums 

# t2 = time.time()
# time_taken_cpu = t2-t1
# print(f"time taken cpu = {time_taken_cpu}")


# ####################################################################################################################################################
# ########### 4. Save All Results ####################################################################################################################
# ####################################################################################################################################################

# # save in one file the sr_cpu, sr_gpu, sr_py, sr_cplus
# # plot a graph of rvals vs sr_cpu, sr_gpu, sr_py, sr_cplus and save
# # save in another file the 4 times takens, time_taken_cpu, time_taken_gpu, time_taken_py, time_taken_cplus
# rvals = np.linspace(2, 49, 50)
# sr_cpu = np.linspace(1, 50, 50)
# sr_gpu = np.linspace(1, 50, 50)
# sr_py = np.linspace(1, 50, 50)
# sr_cplus = np.linspace(1, 50, 50)

# time_taken_cpu = 1.2
# time_taken_gpu = 1.2
# time_taken_py = 1.2
# time_taken_py = 1.2
# time_taken_cplus = 1.2

# results = {
#     "rvals": rvals,
#     "sr_cpu": np.asarray(sr_cpu),
#     "sr_gpu": np.asarray(sr_gpu),
#     "sr_py":  np.asarray(sr_py),
#     "sr_cplus": np.asarray(sr_cplus),
# }

# timings = {
#     "cpu": time_taken_cpu,
#     "gpu": time_taken_gpu,
#     "python": time_taken_py,
#     "c++": time_taken_cplus,
# }


# np.savez(
#     "tcf_comparison_results.npz",
#     **results
# )

# np.savez(
#     "tcf_timings.npz",
#     **timings
# )

# # plot
# plt.figure(figsize=(7, 5))
# plt.axhline(0, color="k", ls=":", lw=1)

# plt.plot(rvals, results["sr_cplus"], lw=2.5, label="C++", color="C0")
# plt.plot(rvals, results["sr_py"],    lw=2.0, label="Python", color="C1")
# plt.plot(rvals, results["sr_cpu"],   lw=2.0, ls="--", label="CPU batch", color="C2")
# plt.plot(rvals, results["sr_gpu"],   lw=2.0, ls="--", label="GPU", color="C3")

# plt.xlabel(r"$r$ [Mpc]")
# plt.ylabel(r"TCF $s(r)$")
# plt.title("TCF comparison: C++ vs Python vs CPU vs GPU")
# plt.legend()
# plt.tight_layout()

# plt.savefig("tcf_comparison.png", dpi=200)
# plt.close()


# labels = list(timings.keys())
# values = [timings[k] for k in labels]

# plt.figure(figsize=(6, 4))
# plt.bar(labels, values, color="steelblue")
# plt.ylabel("Wall time [s]")
# plt.title("TCF runtime comparison")
# plt.tight_layout()
# plt.savefig("tcf_timing_comparison.png", dpi=200)
# plt.close()

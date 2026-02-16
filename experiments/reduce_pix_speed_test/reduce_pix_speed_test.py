import sys
print("Which env am i in?", sys.executable)

# catwoman required imports
import os
import re
import time
import copy as cp
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
import pandas as pd
from scipy.interpolate import PchipInterpolator, CubicSpline
import warnings
from types import SimpleNamespace
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
from catwoman.shelter import Cat

# other imports
from pathlib import Path
import re, subprocess
import math
import matplotlib.patheffects as pe
import pickle as pkl
import time
import h5py

import tools21cm as t2c
from TCFpy import tcf, get_bispec

import tools21cm.cosmo as cm

from TCF_pipeline_GPU import compute_tcf_for_slices_list, extract_LoReLi_slices_every_dMpc


################################################################################################################################################
# open sim at a given z
################################################################################################################################################
sn = "16382"
sim = Cat("16382",
    redshift_range=[6.45,6.55],
    skip_early=False,
    path_spectra='spectra',
    path_sim='/data/cluster/emc-brid/Datasets/LoReLi/simcubes',
    base_dir='/data/cluster/emc-brid/Datasets/LoReLi',
    load_params=False,
    load_spectra=False,
    just_Pee=True,
    reinitialise_spectra=False,
    save_spectra=False,
    load_density_cubes=True,
    load_xion_cubes=False,
    load_T21cm_cubes=True,
    verbose=True,
    debug=False)

print(sim.z)



################################################################################################################################################
# extract slices
################################################################################################################################################

cube_3d = sim.T21cm[0]
box_size_mpc = sim.box_size
delta_mpc = 40


slices_meta = []   # list of dicts: axis, idx, r_mpc
clean_slices = []

for ax in ["x", "y", "z"]:
    slices_ax, idx_ax, r_ax = extract_LoReLi_slices_every_dMpc(cube_3d, box_size_mpc, delta_mpc=delta_mpc, axis=ax, demean=True,
                                                                   save=False, output_dir=None, verbose=True)      

    # append slices and metadata
    for sl, ii, rr in zip(slices_ax, idx_ax, r_ax):
        clean_slices.append(sl)
        slices_meta.append({"axis": ax, "slice_idx": int(ii), "r_mpc": float(rr)})

slices = np.asarray(clean_slices, dtype=np.float32)  # (nslices_total, N, N)

print(np.shape(slices))

################################################################################################################################################
# crop slices
################################################################################################################################################

def crop_slices(slices, new_size=128):
    """
    Crop a stack of 2D slices to the top-left new_size x new_size region.

    """
    
    # Safety check
    if slices.shape[1] < new_size or slices.shape[2] < new_size:
        raise ValueError("new_size is larger than the slice dimensions.")

    return slices[:, :new_size, :new_size]

cropped_slices = crop_slices(slices, 128)
print("shape of cropped slices", np.shape(cropped_slices))



################################################################################################################################################
# reduce slice resolution
################################################################################################################################################


def block_average_slices(slices, bin_factor=2):
    """
    Downsample 2D slices by block-averaging.

    Parameters
    ----------
    slices : ndarray
        Array of shape (N_slices, Ny, Nx)
    bin_factor : int
        Factor by which to reduce resolution (e.g. 2 for 256->128)

    Returns
    -------
    downsampled : ndarray
        Array of shape (N_slices, Ny/bin_factor, Nx/bin_factor)
    """

    N, Ny, Nx = slices.shape

    if Ny % bin_factor != 0 or Nx % bin_factor != 0:
        raise ValueError("Slice dimensions must be divisible by bin_factor.")

    new_Ny = Ny // bin_factor
    new_Nx = Nx // bin_factor

    # reshape and average
    downsampled = slices.reshape(
        N,
        new_Ny, bin_factor,
        new_Nx, bin_factor
    ).mean(axis=(2,4))

    return downsampled


binned_slices = block_average_slices(slices, bin_factor=2)
print("shape of binned slices", np.shape(binned_slices))



################################################################################################################################################
# plot slices to check
################################################################################################################################################


plt.imshow(slices[0])
plt.title(f"Full Sim (L={box_size_mpc})")
plt.xlabel("x pix")
plt.ylabel("y pix")
cbar = plt.colorbar()
cbar.set_label(r"$\delta T_{21}$")  
plt.savefig("/home/lcrascal/Code/TCF/TCF_completed_code/TCF_gpu_loreli/reduce_pix_speed_test/full_sim_img.png")
plt.show()
plt.close()

plt.imshow(cropped_slices[0])
plt.title(f"Cropped Sim (L={box_size_mpc/2})")
plt.xlabel("x pix")
plt.ylabel("y pix")
cbar = plt.colorbar()
cbar.set_label(r"$\delta T_{21}$")  
plt.savefig("/home/lcrascal/Code/TCF/TCF_completed_code/TCF_gpu_loreli/reduce_pix_speed_test/cropped_sim_img.png")
plt.show()
plt.close()

plt.imshow(binned_slices[0])
plt.title(f"Reduced Resolution Sim (L={box_size_mpc})")
plt.xlabel("x pix")
plt.ylabel("y pix")
cbar = plt.colorbar()
cbar.set_label(r"$\delta T_{21}$")  
plt.savefig("/home/lcrascal/Code/TCF/TCF_completed_code/TCF_gpu_loreli/reduce_pix_speed_test/binned_sim_img.png")
plt.show()
plt.close()



################################################################################################################################################
# function to save data
################################################################################################################################################


def save_tcf_summary_txt(filename, rvals, mean_sr, std_sr, time_total_sec, num_slices):
    """
    Save:
      col0 = r
      col1 = mean s(r)
      col2 = std s(r)
    """

    filename = Path(filename)

    if not (len(rvals) == len(mean_sr) == len(std_sr)):
        raise ValueError("rvals, mean_sr, std_sr must have same length.")

    data = np.column_stack([rvals, mean_sr, std_sr])

    header = (
        "TCF summary (mean and std over slices)\n"
        f"time_total_sec = {time_total_sec:.6f}\n"
        f"time_per_slice_sec = {time_total_sec/num_slices:.6f}\n" 
        "columns: r  mean_s_r  std_s_r"
    )

    np.savetxt(filename, data, header=header, comments="# ")
    return filename



################################################################################################################################################
# compute and save TCF
################################################################################################################################################
# 0. parameters
rvals = np.linspace(2, 50, 49)

# ------- 1. cropped sim ------- #
print("%%%%%%%% CROPPED SIM TEST %%%%%%%%")
L_cropped = box_size_mpc/2
N_slices_cropped = np.shape(cropped_slices)[0]

t0_cropped = time.time()
nmodes_results, sr_results_cropped, r_used_results_cropped = compute_tcf_for_slices_list(cropped_slices, L_cropped, rvals, verbose=True)
t1_cropped = time.time()

time_total_all_slices_cropped = t1_cropped-t0_cropped
time_per_slice_cropped = time_total_all_slices_cropped/(np.shape(cropped_slices)[0])
print("time total", time_total_all_slices_cropped)
print("time per slice", time_per_slice_cropped)


mean_sr_cropped = np.mean(sr_results_cropped, axis=0)   # average over slices
std_sr_cropped  = np.std(sr_results_cropped, axis=0)    # std over slices
print("shape mean sr cropped", np.shape(mean_sr_cropped))
print("shape std sr cropped", np.shape(std_sr_cropped))
print("shape rused cropped", np.shape(r_used_results_cropped))

out_txt_cropped = "/home/lcrascal/Code/TCF/TCF_completed_code/TCF_gpu_loreli/reduce_pix_speed_test/cropped_sim_tcf_res.txt"
save_tcf_summary_txt(
    out_txt_cropped,
    r_used_results_cropped[0],
    mean_sr_cropped,
    std_sr_cropped,
    time_total_all_slices_cropped,
    N_slices_cropped
)


# ------- 2. binned sim ------- #
print("%%%%%%%% BINNED SIM TEST %%%%%%%%")
L_binned = box_size_mpc
N_slices_binned = np.shape(binned_slices)[0]

t0_binned = time.time()
nmodes_results, sr_results_binned, r_used_results_binned = compute_tcf_for_slices_list(binned_slices, L_binned, rvals, verbose=True)
t1_binned = time.time()

time_total_all_slices_binned = t1_binned-t0_binned
time_per_slice_binned = time_total_all_slices_binned/(np.shape(binned_slices)[0])
print("time total", time_total_all_slices_binned)
print("time per slice", time_per_slice_binned)


mean_sr_binned = np.mean(sr_results_binned, axis=0)   # average over slices
std_sr_binned  = np.std(sr_results_binned, axis=0)    # std over slices

out_txt_binned = "/home/lcrascal/Code/TCF/TCF_completed_code/TCF_gpu_loreli/reduce_pix_speed_test/binned_sim_tcf_res.txt"
save_tcf_summary_txt(
    out_txt_binned,
    r_used_results_binned[0],
    mean_sr_binned,
    std_sr_binned,
    time_total_all_slices_binned,
    N_slices_binned
)





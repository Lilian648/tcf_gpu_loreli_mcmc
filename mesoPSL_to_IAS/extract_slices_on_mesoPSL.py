
from pathlib import Path
from datetime import datetime

import numpy as np
import h5py
from pathlib import Path
from datetime import datetime, timezone

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





# THESE ARE THE VALUES OF THE REDSHIFT CUBE THAT I WANT TO EXTRACT. CHECK THAT THEY ARE CORRECT!
the_z = 10.02347
the_zidx = "021"

##################################################################################################################$
# load data cube from .dat file
##################################################################################################################$

def load_dat_cube(path, N=256):
    """
    This function opens and loads .dat files
    On MesoPSL the sims are stored as .dat files
    """
    path = Path(path)

    # Check file size
    filesize = path.stat().st_size
    nvox = N**3

    # Infer dtype from file size
    if filesize == nvox * 4:
        dtype = np.float32
    elif filesize == nvox * 8:
        dtype = np.float64
    else:
	raise ValueError(
            f"File size {filesize} bytes does not match "
            f"{nvox*4} (float32) or {nvox*8} (float64)."
        )

    print(f"Detected dtype: {dtype}")

    # Load raw binary
    data = np.fromfile(path, dtype=dtype)

    if data.size != nvox:
        raise ValueError(f"Expected {nvox} elements, got {data.size}")

    cube = data.reshape((N, N, N))

    return cube




################################################################################################################################################
# extract slices function
################################################################################################################################################


def extract_LoReLi_slices_every_dMpc(cube_3d, box_size_mpc, delta_mpc=40.0, axis="z", demean=True, save=False, verbose=True):
    """
    Extract 2D slices from a 3D cube at regular physical intervals.

    Parameters
    ----------
    cube_3d : np.ndarray
        3D array (Nx, Ny, Nz) (or equivalent ordering; slicing uses `axis`).
    box_size_mpc : float
        Physical size of the simulation box in Mpc (assumed cubic).
    delta_mpc : float
        Desired spacing between slices in Mpc (e.g. 10 Mpc).
    axis : {"x","y","z",0,1,2}
        Axis along which to take slices.
    demean : bool
        If True, subtract the mean of each 2D slice (slice-by-slice demeaning).
    verbose : bool
        If True, print diagnostic messages.

    Returns
    -------
    slices : np.ndarray or list
        Extracted 2D slices.
    slice_indices : np.ndarray
        Indices along the slicing axis used for each slice.
    slice_r_mpcs : np.ndarray
        Physical positions (Mpc) of each slice from the origin.
    """
    cube_3d = np.asarray(cube_3d)
    if cube_3d.ndim != 3:
        raise ValueError(f"cube_3d must be 3D, got shape {cube_3d.shape}")

    # Map axis if given as string
    if isinstance(axis, str):
        axis_map = {"x": 0, "y": 1, "z": 2}
        a = axis.lower()
        if a not in axis_map:
            raise ValueError(f"axis must be 'x','y','z',0,1,2; got {axis!r}")
        axis = axis_map[a]
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0,1,2; got {axis}")


    N = cube_3d.shape[axis]
    cell_size_mpc = box_size_mpc / N
    step_cells = max(int(round(delta_mpc / cell_size_mpc)), 1)

    slice_indices = np.arange(0, N, step_cells, dtype=int)
    slice_r_mpcs = slice_indices * cell_size_mpc

    if verbose:
        print(f"Box size: {box_size_mpc} Mpc")
        print(f"N cells along axis {axis}: {N}")
        print(f"Cell size: {cell_size_mpc:.4f} Mpc")
        print(f"Requested spacing: {delta_mpc} Mpc")
        print(f"Using step of {step_cells} cells (~{step_cells*cell_size_mpc:.3f} Mpc)")
        print(f"Extracting {len(slice_indices)} slices.")

    slices = []
    for idx, r in zip(slice_indices, slice_r_mpcs):
        if axis == 0:
            slice_2d = cube_3d[idx, :, :]
        elif axis == 1:
            slice_2d = cube_3d[:, idx, :]
        else:
            slice_2d = cube_3d[:, :, idx]

        slice_2d = slice_2d.astype(np.float64, copy=False)
        if demean:
            slice_2d = slice_2d - slice_2d.mean()

        slices.append(slice_2d)
        if verbose:
            print(f"  ?^?^? Extracted idx={idx}, r?^?^?{r:.2f} Mpc")

    slices = np.stack(slices, axis=0)  # (nslices, Ny, Nx)

    if verbose:
        print("All slices extracted.")

    return slices, slice_indices, slice_r_mpcs

######################################################################################################################$
# smooth slices function
######################################################################################################################$

def block_average_slices(slices, bin_factor=2):
    """
    Downsample 2D slices by block-averaging. This effectively reduces the resolution, 
    while keeping the physical size of the sim the same.

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
    downsampled_slices = slices.reshape(
        N,
	new_Ny, bin_factor,
        new_Nx, bin_factor
    ).mean(axis=(2,4))

    return downsampled_slices




######################################################################################################################$
# save h5 file of a single sim
######################################################################################################################$


def z_to_group_name(z, ndp=3):
    """
    Converts a float redshift to a string (without the .) to use as a filename
    """
    # 6.457 -> "zsim_6p457"
    fmt = f"{{:.{ndp}f}}".format(float(z))
    return "zsim_" + fmt.replace(".", "p")

def save_slices_h5(h5_path, sim_name, z, box_size_mpc, delta_mpc, slices, slices_meta, bin_factor=None, demean=True, overwrite_group=False):
    """
    Save extracted 2D slices of a 3D simulation cube into an HDF5 file.
    If the file already exists, it is opened in append mode.
    The data are stored under a group named according to the redshift, e.g. "zsim_7p431"

    Parameters
    ----------
    h5_path : str or Path
        Output HDF5 file path. Parent directories are created if needed.

    sim_name : str or int
        Simulation identifier stored as a file-level attribute.

    z : float
        Redshift of the cube.

    box_size_mpc : float
        Simulation box size in comoving Mpc.

    delta_mpc : float
        Mpc separation between slices.

    slices : array-like, shape (n_slices, Ny, Nx)
        3D array containing 2D slice data.

    slices_meta : list of dict
        Metadata for each slice. Each dictionary must contain:
            - "axis"      : str
            - "slice_idx" : int
            - "r_mpc"     : float

    bin_factor : int, optional
        Spatial binning factor applied before saving.

    demean : bool, default=True
        Whether slices were mean-subtracted prior to saving.

    overwrite_group : bool, default=False
        If True, an existing redshift group will be deleted and replaced.
        If False, a ValueError is raised if the group already exists.

    Returns
    -------
    h5_path : str
        Path to the saved HDF5 file.

    gname : str
        Name of the redshift group created inside the file.

    Notes
    -----
    - Data are stored in float32 format.
    - The file is opened in append mode ("a"), so multiple redshift
      groups can coexist in the same file.

    """

    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    slices = np.asarray(slices)
    if slices.ndim != 3:
        raise ValueError(f"slices must be 3D (n_slices, Ny, Nx). Got {slices.shape}")

    n_slices, Ny, Nx = slices.shape

    # metadata arrays
    axis_arr = np.array([m["axis"] for m in slices_meta], dtype="S1")  # store as bytes
    idx_arr  = np.array([m["slice_idx"] for m in slices_meta], dtype=np.int32)
    r_arr    = np.array([m["r_mpc"] for m in slices_meta], dtype=np.float32)

    if len(axis_arr) != n_slices:
        raise ValueError("slices_meta length does not match number of slices.")

    gname = z_to_group_name(z, ndp=3)

    with h5py.File(h5_path, "a") as f:
        # file-level attrs (set once / overwrite harmlessly)
        f.attrs["sim_name"] = str(sim_name)
        f.attrs["box_size_mpc"] = float(box_size_mpc)
        f.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()

        # handle group existence

        if gname in f:
            if overwrite_group:
                del f[gname]
            else:
                raise ValueError(f"Group {gname} already exists in {h5_path}. Set overwrite_group=True to replace.")

        g = f.create_group(gname)
        g.attrs["z"] = float(z)
        g.attrs["delta_mpc"] = float(delta_mpc)
        g.attrs["Ny"] = int(Ny)
        g.attrs["Nx"] = int(Nx)
        g.attrs["demean"] = bool(demean)
        if bin_factor is not None:
            g.attrs["bin_factor"] = int(bin_factor)

        sg = g.create_group("slices")

        # per-slice metadata
        sg.create_dataset("axis", data=axis_arr)
        sg.create_dataset("slice_idx", data=idx_arr)
        sg.create_dataset("r_mpc", data=r_arr)

        # main data
        sg.create_dataset(
            "data",
            data=slices.astype(np.float32, copy=False))

    print(f"sim {sim_name} saved to {h5_path}")

    return str(h5_path), gname


######################################################################################################################$
# pipeline for a single sim
######################################################################################################################$


def pipeline_single_sim(sim_name):
    """
    Full run through for a single sim
    0. get parameters
    1. open sim from .dat file
    2. extract 2D slices
    3. reduce resolution
    4. save as h5 file (in my mesoPSL travail folder, will have to move to IAS afterwards)
    """
    
    # ----- 0. get parameters ----- #
    path = Path(
        f"/loreli/rmeriot/simus_loreli/simu{sim_name}/runtime_parameters_simulation_{sim_name}"
    )

    with open(path, "r") as f:
        for line in f:
            if "/Box size in kpc/" in line:
                # Extract the first number in the line
                value_str = line.split("/")[0].strip()

                # Sometimes it looks like "296735,90504451"
                # Replace comma with dot if needed
                value_str = value_str.replace(",", ".")

                box_size_kpc = float(value_str)
                box_size_mpc = box_size_kpc / 1000.0

                 
    print("box_size_mpc", box_size_mpc)

    # ----- 1. open sim from file ----- #

    print("loading sim")

    path = f"/loreli/rmeriot/simus_loreli/simu{sim_name}/postprocessing/cubes/dtb/dtb_tp_hi_256_nocorrection_out{the_zidx}.dat"
    sim_cube = load_dat_cube(path, N=256)

    z = the_z
    print("redshfit", z, the_zidxcd)

    print("sim loaded")

    # 2. ----- extract slices ----- #
    delta_mpc = 40 # Mpc

    slices_meta = []   # list of dicts: axis, idx, r_mpc
    slices = []

    for ax in ["x", "y", "z"]:
        slices_ax, idx_ax, r_ax = extract_LoReLi_slices_every_dMpc(cube_3d=sim_cube, box_size_mpc=box_size_mpc, delta_mpc=delta_mpc, axis=ax,
                                                                   demean=True,verbose=True)

        # append slices and metadata
        for sl, ii, rr in zip(slices_ax, idx_ax, r_ax):
            slices.append(sl)
            slices_meta.append({"axis": ax, "slice_idx": int(ii), "r_mpc": float(rr)})

    slices = np.asarray(slices, dtype=np.float32)  # (nslices_total, N, N)



    # 3. ----- downsample ----- #
    print("downdsampling")
    downsampled_slices = block_average_slices(slices, bin_factor=2)

    # 4. ----- save as h5 file ----- #
    print("saving file")
    h5_path = Path(f"/travail/lcrascall-kennedy/LoReLi_sim_slices/{sim_name}_slices.h5")
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path, gname = save_slices_h5(h5_path, sim_name, the_z, box_size_mpc, delta_mpc, downsampled_slices, slices_meta, bin_factor=None,
                                    demean=True, overwrite_group=False)




    # plt.imshow(downsampled_slices[0])
    # plt.title(f"mean pix = {np.mean(downsampled_slices[0])}")
    # plt.savefig(f"FULL_PIPE_tests/{sim_name}_slices_plot_check.png")
    # plt.close()



######################################################################################################################$
# Run through all LoReLi sims
######################################################################################################################$

def get_sim_list(base_dir="/loreli/rmeriot/simus_loreli"):
    """
    get list of all sims, .dat files
    """
    base = Path(base_dir)

    sim_list = []

    for p in base.iterdir():
        if p.is_dir() and p.name.startswith("simu"):
            sim_number = p.name.replace("simu", "")
            sim_list.append(sim_number)

    # Sort numerically (important!)
    sim_list = sorted(sim_list, key=int)

    return sim_list

sim_list = get_sim_list()


print("Found", len(sim_list), "sims")
print("First 10:", sim_list[:10])


# I want to keep note of which sims fail
# A specific sim will fail if it has no cube at a specific redshift (becuase reionisation has already ended) or if ther eis just no data for some reason
# Below I create a list of the names of all the sim that failed. This list is then saved as a txt file (in the same place as the h5 files)

failed = []

log_file = Path(f"/travail/lcrascall-kennedy/LoReLi_sim_slices/failed_sims_z{the_z}.txt")

for idx, sn in enumerate(sim_list):
    try:
        print(f"sim: {sn}, {idx+1}/{len(sim_list)}")
        # try to run full pipeline for this sim
        pipeline_single_sim(sn)

    except Exception as e:
        print(f"Skipping sim {sn} due to error:")
        print(e)
        # if pipeline fails, record failed sim name 
        failed.append((sn, str(e)))
        continue


# ---- Save failures to file ----
if failed:
    with open(log_file, "w") as f:
        f.write(f"Failed sims log\n")
        f.write(f"Created: {datetime.now()}\n\n")

        for sn, err in failed:
            f.write(f"{sn}\n")
            f.write(f"    {err}\n\n")

    print(f"\nSaved failure log to {log_file}")


else:
    print("\nNo failed sims ?^?^?^?")

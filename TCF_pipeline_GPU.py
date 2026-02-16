"""
This is the complete (or what will be the complete) pipeline for computing a TCF of the LoReLi sims
(computing TCF with TCFpy GPU code)

1. input list of sims
2. add SKA noise (optional) to each sim
3. extract 2D slices 
4. compute TCF and save

+ checks 
"""

import sys
print("Which env am i in?", sys.executable)

import os
import re
import copy as copy
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
#from catwoman.shelter import Cat

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



################################################################################################################################################
#################################### Functions #################################################################################################
################################################################################################################################################

def extract_LoReLi_slices_every_dMpc(cube_3d, box_size_mpc, delta_mpc=40.0, axis="z", demean=True, save=False, output_dir=None, verbose=True):
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
    save : bool
        If True, save each slice as a text file.
    output_dir : str or Path or None
        Directory where slices will be saved if `save=True`. Ignored if `save=False`.
    fmt : str
        Format string for np.savetxt if saving.
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
    saved_files : list of str
        Only returned if save=True. Filepaths of saved slices.
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

    saved_files = []

    if save:
        if output_dir is None:
            raise ValueError("output_dir must be provided when save=True")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

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

        if save:
            r_int = int(round(r))
            fname = output_dir / f"slice_axis{axis}_idx{idx}_r{r_int}Mpc.txt"
            np.savetxt(fname, slice_2d, fmt="%.8e")
            saved_files.append(str(fname))
            if verbose:
                print(f"  ✔ Extracted and saved idx={idx}, r≈{r:.2f} Mpc -> {fname}")
        else:
            if verbose:
                print(f"  ✔ Extracted idx={idx}, r≈{r:.2f} Mpc")

    slices = np.stack(slices, axis=0)  # (nslices, Ny, Nx)

    if verbose:
        print("All slices extracted.")

    if save:
        return slices, slice_indices, slice_r_mpcs, saved_files
    return slices, slice_indices, slice_r_mpcs



def pyTCF_of_2Dslice_GPU(field, L, rvals=None, outfile=None):
    """
    Compute the 2D TCF using the GPU batch estimator.

    Parameters
    ----------
    field : array_like (n, n)
        2D slice.
    L : float
        Box size Mpc (same units as r).
    rvals : array_like or None
        Radii at which to compute TCF. If None, uses r = dx..L/2 in steps of dx.

    Returns
    -------
    r_used : np.ndarray (Nr,)
    sr : np.ndarray (Nr,)
    """
    print(" !!!!!!!!!!! Using GPU python code !!!!!!!!!!!")
    
    ndim = field.ndim
    n = field.shape[0]
    assert field.shape[0] == field.shape[1], "field must be square (n,n) for 2D."

    dx = L / n
    if rvals is None:
        r_used = np.arange(dx, L/2, dx, dtype=np.float64)
    else:
        r_used = np.asarray(rvals, dtype=np.float64)

    # GPU precompute
    field_gpu = cp.asarray(field, dtype=cp.float32)
    field_k = cp.fft.fftn(field_gpu)

    # phase-only field epsilon_k
    abs_field_k = cp.abs(field_k)
    m = abs_field_k < (2.0 * cp.pi / L)
    epsilon_k = cp.where(m, 0, field_k / abs_field_k)

    kcoord = cp.fft.fftfreq(n, d=dx / (2.0 * cp.pi))

    # batching
    n_samples = n ** (ndim * 2)
    batch_size = 100000
    n_batches = n_samples // batch_size + int(n_samples % batch_size != 0)

    tcf_sums = cp.zeros(r_used.shape, dtype=cp.float32)

    time0 = time.time()
    for pairs in tqdm(generate_kq_pairs_batches_gpu(n, ndim, batch_size=batch_size), total=n_batches):
        Bk_batch = estimate_bispectrum_gpu(epsilon_k, n, pairs, ndim)
        k_norms, q_norms, p_norms = compute_gpu_norms(pairs, ndim, kcoord)
        s_v = tcf_partial_vectorized_gpu(r_used, Bk_batch.real, k_norms, q_norms, p_norms, L, ndim)
        tcf_sums += s_v.astype(cp.float32)

    sr = cp.asnumpy(tcf_sums)
    time1 = time.time()
    print(f"Time taken for batch TCF (n={n}, ndim={ndim}, batch={batch_size}): {time1 - time0:.2f} s")

    # Optional saving to text file
    if outfile is not None:
        print("saving")
        header = (
            f"dim {ndim}\n"
            f"s(r)   r")
        data = np.column_stack((sr, r_used))
        np.savetxt(outfile, data, header=header, comments="# ")

    # fake nmodes (to match output of normal python function)
    nmodes = np.zeros_like(np.asarray(sr))

    return nmodes, sr, r_used



def compute_tcf_for_slices_list(slices, L, rvals, verbose=True):
    """
    Compute TCF for each 2D slice.

    Parameters
    ----------
    slices : array-like
        Either a list of (Ny, Nx) arrays or a stacked array (nslices, Ny, Nx).
    L : float
        Physical size (Mpc) of the 2D slice.
    rvals : array-like
        Radii at which to compute the TCF.
    verbose : bool
        Print progress/timing.

    Returns
    -------
    sr_results : np.ndarray
        Shape (nslices, len(rvals)) 
    rvals : array-like
        Radii at which to compute the TCF (same as inputted param,  for clarity).
        shape len(rvals)
    """
    t0 = time.time()

    slices = np.asarray(slices)
    nslices = len(slices)

    sr_results = []
    r_used_results = []

    for i, sl in enumerate(slices):
        if verbose:
            print(f"➤ {i+1}/{nslices}")
        tstart = time.time()

        nmodes, sr, r_used = pyTCF_of_2Dslice_GPU(sl, L, rvals, outfile=None)

        sr_results.append(sr)
        r_used_results.append(r_used)

        if verbose:
            print(f"  ✓ time: {time.time() - tstart:.1f}s")

    if verbose:
        print(f"all TCFs for this list/sim computed in {time.time() - t0:.1f}s")

    # fake nmodes (to match python tcf code)
    nmodes_results = np.zeros_like(np.asarray(sr_results))

    return np.array(nmodes_results), np.array(sr_results), r_used_results




################################################################################################################################################
#################################### Adding Noise + Smoothing to Clean Slice ###################################################################
################################################################################################################################################




def add_noise_smoothing(clean_xy, uvmap_filename, sim_params, noise_params, depth_mhz):
    """
    Add SKA noise and smoothing to a single sim slice at one redshift using tools21cm

    Parameters
    ----------
    clean_xy : array (shape (NxN))
        clean sim slice
    uvmap_filename : str or Path
        Cache file for UV maps.
    sim_params : dict
        Must contain: 
        - redshift (float), 
        - box_length_Mpc (float), 
        - box_dim (int)
    noise_params : dict
        Must contain: 
        - obs_time, 
        - total_int_time, 
        - int_time,
        - declination,
        - subarray_type,
        - bmax_km
        Optional: "
        - uv_weighting,
        - verbose,
        - njobs,
        - checkpoint
    depth_mhz : float
        Frequency channel width (MHz) used to set the thermal noise amplitude.

    Returns
    -------
    noise_xy : ndarray, shape (N,N)
        2D noise realisation in mK (same convention as noise_lightcone output slices).
    noisy_xy : ndarray, shape (N,N)
        2D sim slice = clean + noise
    obs_xy : ndarray, shape (N,N)
        2D sim slice = clean + noise + smoothing
    """
    # unpack parameters
    z = float(sim_params["redshift"])
    N = int(sim_params["box_dim"])
    boxsize = float(sim_params["box_length_Mpc"])

    obs_time = float(noise_params["obs_time"])
    total_int_time = float(noise_params["total_int_time"])
    int_time = float(noise_params["int_time"])
    declination = float(noise_params["declination"])
    subarray_type = noise_params["subarray_type"]
    bmax_km = float(noise_params["bmax_km"])

    uv_weighting = noise_params.get("uv_weighting", "natural")
    verbose = bool(noise_params.get("verbose", False))
    njobs = int(noise_params.get("njobs", 1))
    checkpoint = noise_params.get("checkpoint", 16)

    uvpath = Path(uvmap_filename)
    uvpath.parent.mkdir(parents=True, exist_ok=True)
    print(f"{'Reusing' if uvpath.exists() else 'Will save'} UV map at: {uvpath}")

    # 1. Build/load the uv map for this redshift
    uvs = t2c.get_uv_map_lightcone(
        N, 
        np.array([z], dtype=float),
        subarray_type=subarray_type,
        total_int_time=total_int_time,
        int_time=int_time,
        boxsize=boxsize,
        declination=declination,
        save_uvmap=str(uvpath),
        n_jobs=njobs,
        verbose=verbose,
        checkpoint=checkpoint,
    )
    N_ant = uvs.get("Nant") or uvs.get("N_ant")  # version differences
    uv_map = uvs[f"{z:.3f}"]

    # 2. Generate 2D noise (Jy)
    noise2d_jy = t2c.noise_map(
        N, z, depth_mhz,
        obs_time=obs_time,
        subarray_type=subarray_type,
        boxsize=boxsize,
        total_int_time=total_int_time,
        int_time=int_time,
        declination=declination,
        uv_map=uv_map,
        N_ant=N_ant,
        uv_weighting=uv_weighting,
        verbose=False,
    )

    # Convert to Kelvin/mK like noise_lightcone does
    noise_xy = t2c.jansky_2_kelvin(noise2d_jy, z, boxsize=boxsize)

    # 3. Add nosie to clean sim
    noisy_xy = clean_xy + noise_xy

    # 4. Smoothing
    dtheta = (1.0 + z) * 21e-5 / bmax_km  # radians-ish small-angle factor used by tools21cm
    ang_res_mpc = dtheta * t2c.cm.z_to_cdist(z)   # comoving Mpc

    # convert to sigma in pixels
    fwhm = ang_res_mpc * N / boxsize

    # apply Gaussian smoothing
    obs_xy = t2c.smooth_gauss(noisy_xy, fwhm=fwhm)

    return noise_xy, noisy_xy, obs_xy


# testing


# obs_time = 1000.                      # total observation hours
# total_int_time = 6.                   # hours per day
# int_time = 10.                        # seconds
# declination = -30.0                   # declination of the field in degrees
# subarray_type = "AA4"
# bmax_km = 2. #* units.km # km needed for smoothibg

# verbose = True
# uvmap_filename = "tests/uvmap_mock.h5"
# njobs = 1
# checkpoint = 16

# sim_params = {
#     "redshift": 6.0,
#     "box_length_Mpc": 296.0/4.0,  # 1/4 of the full sim size
#     "box_dim": 64,
# }
# print(type(sim_params["redshift"]))

# depth_mhz = 0.07 # no idea if this is correct or not

# noise_params = {
#     "obs_time": 1000.0,         # total observation hours
#     "total_int_time": 6.0,      # hours per day
#     "int_time": 10.0,           # seconds
#     "declination": -30.0,       # degrees
#     "subarray_type": "AA4",
#     "verbose": True,
#     "njobs": 1,
#     "checkpoint": 16,
#     "bmax_km": bmax_km
# }

# clean_xy = np.load("tests/mock_LoReLi_sim_N64.npy")[:, :, 0]
# noise_xy, noisy_xy, obs_xy = add_noise_smoothing(clean_xy, uvmap_filename, sim_params, noise_params, depth_mhz)

# np.savetxt("tests/mock_noise_slice.txt", noise_xy)
# np.savetxt("tests/mock_noisy_slice.txt", noisy_xy)
# np.savetxt("tests/mock_obs_slice.txt", obs_xy)



################################################################################################################################################
#################################### Saving TCF Results as h5 File #############################################################################
################################################################################################################################################


def save_tcf_results_h5(h5_path, results, delta_mpc, noise_params, uvmap_filename, compression="gzip"):
    """
    Save TCF pipeline outputs into an HDF5 file (append mode).

    Redshift handling
    -----------------
    Results are stored under a redshift group named like:
        zsim_7p50
    If that group already exists, the function creates a versioned group (with warning):
        zsim_7p50_v2, zsim_7p50_v3, ...

    Stored content (per redshift group)
    -----------------------------------
    - attrs: z, z_idx, delta_mpc, uvmap_filename, noise_params_json
    - /slices: axis, slice_idx, r_mpc
    - /tcf/clean: sr, nmodes
    - /tcf/noise: sr, nmodes
    - /tcf/obs:   sr, nmodes

    Stored content (file-level)
    ---------------------------
    - attrs: sim_name, L_Mpc, N, created_utc
    - /rvals: array of TCF radii (created once; checked for consistency on subsequent writes)

    Returns
    -------
    saved_group : str
        The HDF5 group name actually used (e.g. "zsim_7p50" or "zsim_7p50_v2").
    """

    def _get_versioned_group_name(h5file, base_name: str) -> str:
        if base_name not in h5file:
            return base_name
        i = 2
        while f"{base_name}_v{i}" in h5file:
            i += 1
        return f"{base_name}_v{i}"

    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- required metadata ----
    sim_name = str(results["sim_name"])
    z = float(results["z"])
    z_idx = int(results["z_idx"])
    L = float(results["L"])
    N = int(results["N"])
    rvals = np.asarray(results["rvals"])

    # ---- arrays to store ----
    clean_sr     = np.asarray(results["clean_sr"])
    clean_nmodes = np.asarray(results["clean_nmodes"])
    
    noise_sr     = results.get("noise_sr", None)
    noise_nmodes = results.get("noise_nmodes", None)
    
    obs_sr       = results.get("obs_sr", None)
    obs_nmodes   = results.get("obs_nmodes", None)
    
    # Convert to arrays only if present
    if noise_sr is not None:
        noise_sr = np.asarray(noise_sr)
        noise_nmodes = np.asarray(noise_nmodes)
    
    if obs_sr is not None:
        obs_sr = np.asarray(obs_sr)
        obs_nmodes = np.asarray(obs_nmodes)


    nslices = clean_sr.shape[0]

    # ---- slice metadata ----
    meta = results["slices_meta"]
    axis_arr = np.array([m["axis"] for m in meta], dtype="S1")
    slice_idx_arr = np.array([m["slice_idx"] for m in meta], dtype=np.int32)
    r_mpc_arr = np.array([m["r_mpc"] for m in meta], dtype=np.float32)

    # ---- consistency checks ----
    Nr = rvals.shape[0]

    def _check_pair(sr, nmodes, label):
        if sr.ndim != 2:
            raise ValueError(f"{label}_sr must be 2D (nslices, Nr), got shape {sr.shape}")
        if nmodes.shape != sr.shape:
            raise ValueError(f"{label}_nmodes has shape {nmodes.shape}, expected {sr.shape}")
        if sr.shape[1] != Nr:
            raise ValueError(f"{label}_sr has Nr={sr.shape[1]}, but len(rvals)={Nr}")
    
    # always check clean
    _check_pair(clean_sr, clean_nmodes, "clean")
    
    # check noise/obs only if provided
    if noise_sr is not None:
        _check_pair(noise_sr, noise_nmodes, "noise")
    if obs_sr is not None:
        _check_pair(obs_sr, obs_nmodes, "obs")
    
    # If both optional blocks exist, ensure shapes match clean
    if noise_sr is not None and noise_sr.shape != clean_sr.shape:
        raise ValueError(f"noise_sr shape {noise_sr.shape} does not match clean_sr shape {clean_sr.shape}")
    if obs_sr is not None and obs_sr.shape != clean_sr.shape:
        raise ValueError(f"obs_sr shape {obs_sr.shape} does not match clean_sr shape {clean_sr.shape}")


    # ---- group naming ----
    z_tag_base = f"zsim_{z:.2f}".replace(".", "p")

    with h5py.File(h5_path, "a") as f:
        # ---------- file-level metadata ----------
        f.attrs["sim_name"] = sim_name
        f.attrs["L_Mpc"] = L
        f.attrs["N"] = N
        if "created_utc" not in f.attrs:
            f.attrs["created_utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

        # ---------- store rvals once (and verify consistency) ----------
        if "rvals" not in f:
            f.create_dataset("rvals", data=rvals, compression=compression)
        else:
            existing = f["rvals"][:]
            if existing.shape != rvals.shape or not np.allclose(existing, rvals):
                raise ValueError(
                    "rvals in file do not match rvals being saved. "
                    "Refusing to append inconsistent results."
                )

        # ---------- create (versioned) redshift group ----------
        z_tag = _get_versioned_group_name(f, z_tag_base)
        if z_tag != z_tag_base:
            warnings.warn(
                f"Redshift group '{z_tag_base}' already exists in {h5_path}. "
                f"Saving results under '{z_tag}' instead."
            )

        g = f.create_group(z_tag)
        g.attrs["z"] = z
        g.attrs["z_idx"] = z_idx
        g.attrs["delta_mpc"] = float(delta_mpc)
        g.attrs["nslices"] = int(nslices)
        g.attrs["Nr"] = int(Nr)

        if uvmap_filename is not None:
            g.attrs["uvmap_filename"] = str(uvmap_filename)
        if noise_params is not None:
            g.attrs["noise_params_json"] = json.dumps(noise_params)

        # ---------- slice metadata ----------
        sg = g.create_group("slices")
        sg.create_dataset("axis", data=axis_arr)
        sg.create_dataset("slice_idx", data=slice_idx_arr)
        sg.create_dataset("r_mpc", data=r_mpc_arr)

        # ---------- TCF results ----------
        tg = g.create_group("tcf")

        def _write_block(parent, name, sr, nmodes):
            bg = parent.create_group(name)
            bg.create_dataset("sr", data=sr, compression=compression)
            bg.create_dataset("nmodes", data=nmodes, compression=compression)

        _write_block(tg, "clean", clean_sr, clean_nmodes)
        if noise_sr is not None:
            _write_block(tg, "noise", noise_sr, noise_nmodes)
        if obs_sr is not None:
            _write_block(tg, "obs",   obs_sr,   obs_nmodes)


    print("Everythin saved!")



################################################################################################################################################
#################################### Main: Run All #############################################################################################
################################################################################################################################################


def TCFpipeline_single_sim(sim_name, sim_cube, z_idx, z, L, rvals, noise_params, uvmap_filename, delta_mpc, compute_noise=True):
    """
    Run the LoReLi → extract slices → adding noise → compute TCF pipeline for a single simulation at a single redshift.

    Parameters
    ----------
    sim_name : str
        Simulation identifier (e.g. "10038").
    sim_cube : np.ndarray
        3D simulation cube at a single redshift with shape (N, N, N).
    z_idx : int
        Index of the redshift slice used in the original simulation/lightcone.
    z : float
        Redshift corresponding to `sim_cube`.
    L : float
        Physical side length of the simulation box in Mpc.
    rvals : array_like
        1D array of triangle side lengths (in Mpc) at which to compute the TCF.
    noise_params : dict
        Dictionary containing parameters required for noise generation.
        These are passed directly to the noise model.
        Expected keys are:
        - "obs_time" : float, Total observing time in hours (e.g. 1000.0).
        - "total_int_time" : float, Total integration time per pointing in hours.
        - "int_time" : float, Integration time per visibility sample in seconds.
        - "declination" : float, Declination of the observed field in degrees.
        - "subarray_type" : str, SKA-Low subarray configuration (e.g. "AA4").
        - "bmax_km" : float, Maximum baseline length in kilometres.
        - "verbose" : bool, If True, print detailed information during noise generation.
        - "njobs" : int, Number of parallel jobs used for noise generation.
    uvmap_filename : str
        Path to the UV-coverage map used for generating interferometric noise.
    delta_mpc : float
        Physical spacing (in Mpc) between consecutive extracted slices
        along each axis. Default is 10.0 Mpc.

    Returns
    -------
    results : dict
        Dictionary containing all pipeline outputs and metadata with keys:
        - "sim_name" : str
            Simulation identifier.
        - "z" : float
            Redshift of the processed cube.
        - "z_idx" : int
            Redshift index of the processed cube.
        - "L" : float
            Physical box size in Mpc.
        - "N" : int
            Number of pixels per dimension in the cube.
        - "rvals" : np.ndarray
            r values at which the TCF was evaluated.
        - "slices_meta" : list of dict
            Metadata for each extracted slice. Each entry contains:
            {"axis", "slice_idx", "r_mpc"}.
        - "clean_sr" : np.ndarray, shape: (N_slices, len(rvals))
            TCF values for clean slices,
            shape (nslices, len(rvals)).
        - "clean_nmodes" : np.ndarray, shape: (N_slices, len(rvals))
            Number of contributing triangle configurations for clean slices,
            same shape as `clean_sr`.
        - "noise_sr" : np.ndarray, shape: (N_slices, len(rvals))
            TCF values for noise-only slices.
        - "noise_nmodes" : np.ndarray, shape: (N_slices, len(rvals))
            Number of modes for noise-only slices.
        - "obs_sr" : np.ndarray, shape: (N_slices, len(rvals))
            TCF values for observed (signal + noise) slices.
        - "obs_nmodes" : np.ndarray, shape: (N_slices, len(rvals))Compute_TCF_Pipeline
            Number of modes for observed slices.

    """

    
    it_begins = time.time()
    print("%%%%%% IT BEGINS:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(it_begins)))

    N = sim_cube.shape[0]

    # ----------------------------
    # 1) Extract slices along x,y,z
    # ----------------------------
    slices_meta = []   # list of dicts: axis, idx, r_mpc
    clean_slices = []

    for ax in ["x", "y", "z"]:
        slices_ax, idx_ax, r_ax = extract_LoReLi_slices_every_dMpc(cube_3d=sim_cube, box_size_mpc=L, delta_mpc=delta_mpc, axis=ax, demean=True,
                                                                   save=False, output_dir=None, verbose=True)

        # append slices and metadata
        for sl, ii, rr in zip(slices_ax, idx_ax, r_ax):
            clean_slices.append(sl)
            slices_meta.append({"axis": ax, "slice_idx": int(ii), "r_mpc": float(rr)})

    clean_slices = np.asarray(clean_slices, dtype=np.float32)  # (nslices_total, N, N)


    # ----------------------------
    # 2) Noise + smoothing per slice
    # ----------------------------

    noise_slices = None
    obs_slices = None

    if compute_noise:
        depth_mhz = (cm.z_to_nu(cm.cdist_to_z(cm.z_to_cdist(z) - L/2)) - cm.z_to_nu(cm.cdist_to_z(cm.z_to_cdist(z) + L/2))) / N

        sim_params = {"redshift": float(z), "box_length_Mpc": float(L), "box_dim": int(N)}

        noise_slices = np.empty_like(clean_slices, dtype=np.float32)
        obs_slices   = np.empty_like(clean_slices, dtype=np.float32)

        for i, sl in enumerate(clean_slices):
            print(f"Adding noise + smoothing to {i+1}/{len(clean_slices)}")
            noise_xy, noisy_xy, obs_xy = add_noise_smoothing(
                sl, uvmap_filename, sim_params, noise_params, depth_mhz
            )
            noise_slices[i] = noise_xy
            obs_slices[i]   = obs_xy


    # ----------------------------
    # 3) Compute TCFs
    # ----------------------------
    print(" %%%%% Compute TCF of CLEAN slices %%%%%%% ")
    clean_nmodes, clean_sr, _ = compute_tcf_for_slices_list(clean_slices, L, rvals, verbose=True)
    
    noise_nmodes = None
    noise_sr = None
    obs_nmodes = None
    obs_sr = None

    if compute_noise:
        noise_nmodes, noise_sr, _ = compute_tcf_for_slices_list(noise_slices, L, rvals, verbose=True)
        obs_nmodes, obs_sr, _     = compute_tcf_for_slices_list(obs_slices,   L, rvals, verbose=True)


    it_ends = time.time()
    print("%%%%%% IT ENDS:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(it_ends)))
    print(f"%%%%%% TOTAL: {it_ends - it_begins:.1f} s")

    return {
        "sim_name": sim_name,
        "z": float(z),
        "z_idx": int(z_idx),
        "L": float(L),
        "N": int(N),
        "rvals": np.asarray(rvals),
        "slices_meta": slices_meta,

        "clean_sr": np.asarray(clean_sr),
        "clean_nmodes": np.asarray(clean_nmodes),

        "noise_sr": None if noise_sr is None else np.asarray(noise_sr),
        "noise_nmodes": None if noise_nmodes is None else np.asarray(noise_nmodes),

        "obs_sr": None if obs_sr is None else np.asarray(obs_sr),
        "obs_nmodes": None if obs_nmodes is None else np.asarray(obs_nmodes),
    }


################## TESTING ##################

# testing for full cube


# # ----------------------------
# # 1. Load sim
# # ----------------------------
# base_dir = '/data/cluster/emc-brid/Datasets/LoReLi' # where Lisa keeps the LoReLi info/sims
# sim_name = '10038'

# print("loading sim")
# sim = Cat(sim_name, redshift_range=[5.5, 6.5], skip_early=True, path_spectra='spectra', path_sim='/data/cluster/emc-brid/Datasets/LoReLi/simcubes', 
#             base_dir=base_dir, load_params=False, load_spectra=False, just_Pee=True, reinitialise_spectra=False, save_spectra=False, 
#             load_density_cubes=False, load_xion_cubes=False, load_T21cm_cubes=True, verbose=True, debug=False)

# print("sim loaded")
# # sim params
# L = sim.box_size # Mpc (check units?)

# # ----------------------------
# # 2. Choose redshift cube 
# # (eg, choosing cube closest to z=6)
# # ---------------------------- 
# z_target = 6
# z_idx = np.abs(sim.z - z_target).argmin()
# z_used = float(sim.z[z_idx])
    
# sim_cube = sim.T21cm[z_idx]  # expected shape (N,N,N)

# # ----------------------------
# # other parameters
# # ----------------------------

# # small r range for speed
# rvals = np.linspace(2, 50, 49)

# noise_params = {
#     "obs_time": 1000.0,        # hours
#     "total_int_time": 6.0,     # hours
#     "int_time": 10.0,          # seconds
#     "declination": -30.0,
#     "subarray_type": "AA4",
#     "bmax_km": 2.0,
#     "verbose": True,
#     "njobs": 1,
# }

# uvmap_filename = "/home/lcrascal/Code/TCF/TCF_completed_code/TCF_python_loreli/tests/uvmap_mock_fullsim.h5"
# #uvmap_filename = "/home/lcrascal/Code/TCF/TCF_completed_code/TCF_python_loreli/tests/uvmap_mock.h5" # this one is for the mock 64x64 cube

# # ---------------------------- 
# # 3. run function
# # ---------------------------- 
# delta_mpc = 20
# #sim_cube = np.load("/home/lcrascal/Code/TCF/TCF_completed_code/TCF_python_loreli/tests/mock_LoReLi_sim_N64.npy") # this one is for the mock 64x64 cube

# res_dict = TCFpipeline_single_sim(sim_name, sim_cube, z_idx, z_used, L, rvals, noise_params, uvmap_filename, delta_mpc)


# # ---------------------------- 
# # 4. save results
# # ---------------------------- 
# h5_path = "mock_TCF_results_256x256.h5"
# save_tcf_results_h5(h5_path, res_dict, delta_mpc, noise_params, uvmap_filename, compression="gzip")


################################################################################################################################################
####################################  ###################################################################
################################################################################################################################################




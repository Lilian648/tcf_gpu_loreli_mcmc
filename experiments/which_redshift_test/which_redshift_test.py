import sys
print("Which env am i in?", sys.executable)

# catwoman required imports
import os
import re
import time
import copy as copy
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
import pandas as pd
from scipy.interpolate import PchipInterpolator, CubicSpline
import warnings
from types import SimpleNamespace
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline

# other imports
from pathlib import Path
import re, subprocess
import math
import matplotlib.patheffects as pe
import pickle as pkl
import time
import json
import h5py
import warnings

import tools21cm as t2c
from TCFpy import tcf, get_bispec
import tools21cm.cosmo as cm


from TCF_pipeline_GPU import extract_LoReLi_slices_every_dMpc, pyTCF_of_2Dslice_GPU, compute_tcf_for_slices_list, add_noise_smoothing, save_tcf_results_h5, TCFpipeline_single_sim
from catwoman.shelter import Cat



###################################################################################################################################################
################# Open All Sims ###################################################################################################################
###################################################################################################################################################

sims_names = ['10038',  '10501',  '11557',  '12091',  '16382',  '16995',  '17338',  '18886']
base_dir = '/data/cluster/emc-brid/Datasets/LoReLi' # where Lisa keeps the LoReLi info/sims

sims = []
redshift_ranges = []
for sn in sims_names:
    
    sim = Cat(sn,
    redshift_range=[100, 0],
    skip_early=False,
    path_spectra='spectra',
    path_sim='/data/cluster/emc-brid/Datasets/LoReLi/simcubes',
    base_dir=base_dir,
    load_params=False,
    load_spectra=False,
    just_Pee=True,
    reinitialise_spectra=False,
    save_spectra=False,
    load_density_cubes=False,
    load_xion_cubes=False,
    load_T21cm_cubes=True,
    verbose=True,
    debug=False)

    redshift_ranges.append(sim.z)
    sims.append(sim)




###################################################################################################################################################
################# Get cubes at given zs ###########################################################################################################
###################################################################################################################################################


def extract_cubes_at_given_zs(sim_name, sim, chosen_z):
    
    chosen_z = np.asarray(chosen_z, dtype=float)
    print("Chosen redshifts:", chosen_z)

    # get available redshifts
    sim_z = np.asarray(sim.z, dtype=float)
    print("Sim has", len(sim_z), "redshift cubes")
    zmin, zmax = sim_z.min(), sim_z.max()

    # infer cube shape from first real cube
    example_cube = sim.T21cm[0]
    if hasattr(example_cube, "values"):
        example_cube = example_cube.values
    else:
        try:
            example_cube = example_cube[...]
        except Exception:
            pass
    example_cube = np.asarray(example_cube)
    cube_shape = example_cube.shape

    closest_indices = []
    z_used = []
    cubes = []
    is_real_cube = []

    for z in chosen_z:
        # check if z is within the sim's redshift coverage
        if z < zmin or z > zmax:
            # ---- no data: insert zero cube ----
            print(f"requested z={z:.2f}  ->  NO DATA (outside [{zmin:.2f}, {zmax:.2f}]), inserting zero cube")

            closest_indices.append(-1)        # sentinel index
            z_used.append(z)                  # keep the "pretend" redshift
            cubes.append(np.zeros(cube_shape, dtype=example_cube.dtype))
            is_real_cube.append(False)

        else:
            # ---- real cube: find closest ----
            idx = int(np.argmin(np.abs(sim_z - z)))
            z_act = sim_z[idx]

            print(f"requested z={z:.2f}  ->  using z={z_act:.4f}  (index {idx})")

            cube = sim.T21cm[idx]
            if hasattr(cube, "values"):
                cube = cube.values
            else:
                try:
                    cube = cube[...]
                except Exception:
                    pass

            closest_indices.append(idx)
            z_used.append(z_act)
            cubes.append(np.asarray(cube))
            is_real_cube.append(True)

    cube_info = {
        "sim_name": sim_name,
        "z_requested": chosen_z,
        "z_used": np.asarray(z_used),
        "z_index": np.asarray(closest_indices),
        "cubes": cubes,
        "is_real_cube": np.asarray(is_real_cube, dtype=bool),
    }

    return cube_info, cubes


######### 0. parameters #########

L = 296
delta_mpc = 40
rvals = np.linspace(2, 50, 49)
noise_params = None
uvmap_filename = None
chosen_z = np.linspace(10, 5, 11)

######### 1. Loop over all sims #########

out_dir = "which_redshift_test"

times = []

for sim_i, (sim_name, sim) in enumerate(zip(sims_names, sims), start=1):
    
    print(f" XXXXXXX SIM {sim_i}/{len(sims)} XXXXXXX")

    t0 = time.time()

    # 1) get cubes for this sim
    cube_info, _ = extract_cubes_at_given_zs(sim_name, sim, chosen_z)

    # 2) one H5 file per sim
    h5_path = f"{out_dir}/{sim_name}.h5"

    # 3) loop over each chosen z cube
    for cube_i, (z_used, z_idx, cube, is_real) in enumerate(
        zip(
            cube_info["z_used"],
            cube_info["z_index"],
            cube_info["cubes"],
            cube_info["is_real_cube"],
        ),
        start=1
    ):
        
        print(f" XXXXXXX CUBE {cube_i}/{len(cube_info['cubes'])} XXXXXXX")

        t1 = time.time()


        if not is_real:
            
            slices_meta = []
            nslices = 0
            clean_sr = np.zeros((nslices, len(rvals)), dtype=np.float32)
            clean_nmodes = np.zeros((nslices, len(rvals)), dtype=np.int32)
        
            # keep consistent schema (noise/obs None if you want)
            res_dict = {
                "sim_name": str(sim_name),
                "z": float(z_used),
                "z_idx": int(z_idx),
                "L": float(L),
                "N": int(np.asarray(cube).shape[0]),
                "rvals": np.asarray(rvals),
                "slices_meta": slices_meta,
        
                "clean_sr": clean_sr,
                "clean_nmodes": clean_nmodes,
                "noise_sr": None,
                "noise_nmodes": None,
                "obs_sr": None,
                "obs_nmodes": None,
        
                "is_real_cube": False,
            }


        else:
            # compute tcf of cube at this z
            res_dict = TCFpipeline_single_sim(sim_name=sim_name, sim_cube=cube, z_idx=z_idx, z=z_used, L=L, rvals=rvals, noise_params=noise_params,
                                              uvmap_filename=uvmap_filename, delta_mpc=delta_mpc, compute_noise=False)
    
            # include flag in results so it can be stored
            res_dict["is_real_cube"] = bool(is_real)

        # save results
        save_tcf_results_h5( h5_path=h5_path, results=res_dict, delta_mpc=delta_mpc, noise_params=noise_params, 
                            uvmap_filename=uvmap_filename, compression="gzip")

        t2 = time.time()
        time_1cube = t2-t1
        print(" %%%%%%%%%%%%%% time to compute TCF of one cube %%%%%%%%%%%%%% ")
        print(f"{time_1cube}s")

    t3 = time.time()
    time_1sim = t3-t0
    times.append(time_1sim)
    print(" %%%%%%%%%%%%%% time to compute TCFs of one sim %%%%%%%%%%%%%% ")
    print(f"{time_1sim}s")


print("%%%%%%%%%%%%%% All sims done. %%%%%%%%%%%%%%")
print(f"times taken for each sim {times}")

















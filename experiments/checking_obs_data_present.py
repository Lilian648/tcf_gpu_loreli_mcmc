import numpy as np
import h5py
from pathlib import Path

import tools21cm as t2c
from tools21cm import cm
import time

from pathlib import Path
import h5py

folder = Path("/data/cluster/reionisation/brightness_temp_slices")
h5_files_list = sorted(folder.glob("*.h5"))

zgroup = "zsim_7p431"
obs_path = f"{zgroup}/slices/obs_data"
expected_shape = (24, 128, 128)

missing = []
wrong_shape = []
present_correct = []
corrupted = []

print(f"Checking {len(h5_files_list)} files...\n")

for idx, h5_file in enumerate(h5_files_list, start=1):
    t0 = time.time()

    try:
        with h5py.File(h5_file, "r") as f:

            if obs_path not in f:
                print(f"[MISSING] {h5_file.name}")
                missing.append(h5_file.name)
                continue

            shape = f[obs_path].shape

            if shape != expected_shape:
                print(f"[WRONG SHAPE] {h5_file.name}  ->  {shape}")
                wrong_shape.append((h5_file.name, shape))
            else:
                present_correct.append(h5_file.name)

    except Exception as e:
        print(f"[CORRUPTED] {h5_file.name}")
        print("   ", e)
        corrupted.append(h5_file.name)

    t1 = time.time()
    print("time = ",  t1-t0)

print("\n========================================")
print(f"Total files checked: {len(h5_files_list)}")
print(f"Correct shape:       {len(present_correct)}")
print(f"Missing obs_data:    {len(missing)}")
print(f"Wrong shape:         {len(wrong_shape)}")
print(f"Corrupted files:     {len(corrupted)}")
print("========================================")

if wrong_shape:
    print("\nFiles with wrong shape:")
    for name, shape in wrong_shape:
        print(f"  {name} -> {shape}")

if corrupted:
    print("\nCorrupted files:")
    for name in corrupted:
        print(f"  {name}")

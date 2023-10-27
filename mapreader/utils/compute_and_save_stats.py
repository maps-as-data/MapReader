#!/usr/bin/env python
from __future__ import annotations

import os
from glob import glob

from parhugin import multiFunc

from mapreader import load_patches


def save_stats(one_dir):
    mymaps = load_patches(
        os.path.join(one_dir, "patch_50_50", "*png"),
        parent_paths=os.path.join(one_dir, "*png"),
    )

    path2metadata = os.path.join(one_dir, "metadata.csv")
    mymaps.add_metadata(metadata=path2metadata)
    mymaps.add_center_coord()
    mymaps.calc_pixel_stats()
    parent_df, patch_df = mymaps.convertImages()
    parent_df.to_csv(os.path.join(one_dir, "parent_df.csv"), mode="w", header=True)
    patch_df.to_csv(os.path.join(one_dir, "patch_df.csv"), mode="w", header=True)


list_dirs = glob("/maps_large/2021_experiments_1inch/chunks_*")

myproc = multiFunc(num_req_p=10)

list_jobs = []
for one_dir in list_dirs:
    list_jobs.append([save_stats, (one_dir,)])

# and then adding them to myproc
myproc.add_list_jobs(list_jobs)
myproc.run_jobs()

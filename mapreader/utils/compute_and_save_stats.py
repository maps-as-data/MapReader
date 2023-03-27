#!/usr/bin/env python
# -*- coding: utf-8 -*-

from glob import glob
import os

from mapreader import read_patches
from parhugin import multiFunc


def save_stats(one_dir):
    mymaps = read_patches(
        os.path.join(one_dir, "slice_50_50", "*png"),
        parent_paths=os.path.join(one_dir, "*png"),
    )

    path2metadata = os.path.join(one_dir, "metadata.csv")
    mymaps.add_metadata(metadata=path2metadata)
    mymaps.add_center_coord()
    mymaps.calc_pixel_stats()
    pd_parent, pd_child = mymaps.convertImages()
    pd_parent.to_csv(os.path.join(one_dir, "pd_parent.csv"), mode="w", header=True)
    pd_child.to_csv(os.path.join(one_dir, "pd_child.csv"), mode="w", header=True)


list_dirs = glob("/maps_large/2021_experiments_1inch/chunks_*")

myproc = multiFunc(num_req_p=10)

list_jobs = []
for one_dir in list_dirs:
    list_jobs.append([save_stats, (one_dir,)])

# and then adding them to myproc
myproc.add_list_jobs(list_jobs)
myproc.run_jobs()

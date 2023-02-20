#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import glob
import os

from mapreader import read
from parhugin import multiFunc

# Serial version fo the Func to be run in parallel
def patchify_serial(
    path2images_dir,
    patch_size=100,
    patch_method="pixel",
    output_dirname="patches_100_100",
):
    """Patchify images stored in path2images_dir.
    This function is the serial version and will be run in parallel using parhugin

    Parameters
    ----------
    path2images_dir : str
        Path to images
    patch_size : int, optional
        Number of pixels/meters in both x and y to use for slicing, by default 100
    patch_method : str, optional
        Method used to patchify, choices between "pixel" and "meters" or "meter", by default "pixel"
    output_dirname : str, optional
        Directory to save the patches, by default "patches_100_100"
    """

    path2images = os.path.join(path2images_dir, "*png")
    mymaps = read(path2images)
    path2metadata = os.path.join(path2images_dir, "metadata.csv")
    mymaps.add_metadata(metadata=path2metadata)

    # method can also be set to meters
    mymaps.patchifyAll(
        path_save=os.path.join(path2images_dir, output_dirname),
        patch_size=patch_size,
        square_cuts=False,
        verbose=False,
        rewrite=True,
        method=patch_method,
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Run patchifyAll method in parallel.")
    parser.add_argument("--path2dirs", default="/maps_large_03/six_inch_v001/chunks_*")
    arguments = parser.parse_args()

    path2images_all_dirs = glob.glob(arguments.path2dirs)
    patchify_serial(path2images_all_dirs[0], 100, "pixel", "patches_100_100")

    ### myproc = multiFunc(num_req_p=12)

    ### list_jobs = []
    ### for path2images_dir in path2images_all_dirs:
    ###     list_jobs.append([patchify_serial, (path2images_dir, 100, "pixel", "patches_100_100")])

    ### # and then adding them to myproc
    ### myproc.add_list_jobs(list_jobs)
    ### print(myproc)

    ### myproc.run_jobs()

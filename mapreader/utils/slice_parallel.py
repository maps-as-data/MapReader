#!/usr/bin/env python
from __future__ import annotations

import glob
import os
from argparse import ArgumentParser

from mapreader import loader


# Serial version fo the Func to be run in parallel
def slice_serial(
    path2images_dir,
    slice_size=100,
    slice_method="pixel",
    output_dirname="slice_100_100",
):
    """Slice images stored in path2images_dir
    This function is the serial version and will be run in parallel using parhugin
    """

    path2images = os.path.join(path2images_dir, "*png")
    mymaps = loader(path2images)
    path2metadata = os.path.join(path2images_dir, "metadata.csv")
    mymaps.add_metadata(metadata=path2metadata)

    # method can also be set to meters
    mymaps.sliceAll(
        path_save=os.path.join(path2images_dir, output_dirname),
        slice_size=slice_size,
        square_cuts=False,
        verbose=False,
        rewrite=True,
        method=slice_method,
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Run sliceAll method in parallel.")
    parser.add_argument("--path2dirs", default="/maps_large_03/six_inch_v001/chunks_*")
    arguments = parser.parse_args()

    path2images_all_dirs = glob.glob(arguments.path2dirs)
    slice_serial(path2images_all_dirs[0], 100, "pixel", "slice_100_100")

    ### myproc = multiFunc(num_req_p=12)

    ### list_jobs = []
    ### for path2images_dir in path2images_all_dirs:
    ###     list_jobs.append([slice_serial, (path2images_dir, 100, "pixel", "slice_100_100")])

    ### # and then adding them to myproc
    ### myproc.add_list_jobs(list_jobs)
    ### print(myproc)

    ### myproc.run_jobs()

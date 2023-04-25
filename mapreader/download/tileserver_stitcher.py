#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**Stitcher for tileserver**

The main code for the stitcher was sourced from a repository located at https://github.com/stamen/the-ultimate-tile-stitcher, which is licensed under the MIT license. 
The adapted functions were then used to run the scraper via Python modules.
"""

from .tileserver_helpers import input_class

import glob
import os

from PIL import Image
from typing import Union, Optional


def runner(opts: input_class) -> None:
    """
    Stitch together a series of images into a larger image.

    Parameters
    -----------
    opts : input_class
        The options for the runner, of the ``input_class`` type that contains
        the following attributes:

        - ``dir`` (str): The directory containing the input images.
        - ``out_file`` (str): The output file path for the stitched image.
        - ``pixel_closest`` (int, optional): The closest pixel value to round the image size to.

    Raises
    ------
    SystemExit
        If no input files are found in the specified directory.

    Returns
    -------
    None
        The function saves the stitched image to the specified output file
        path.

    Notes
    -----
    This function is usually called through the
    :func:`mapreader.download.tileserver_stitcher.stitcher` function. Refer to
    the documentation of that method for a simpler implementation.
    """
    search_path = os.path.join(opts.dir, "*_*_*.png")

    filepaths = glob.glob(search_path)

    def xy(filepath):
        # base = os.path.basename(filepath)
        z, x, y = filepath.split("_")
        y = os.path.splitext(y)[0]
        return int(x), int(y)

    # yx = lambda filepath: xy(filepath)[::-1]

    filepaths = sorted(filepaths, key=xy)

    if len(filepaths) == 0:
        print("No files found")
        raise SystemExit

    tile_w, tile_h = Image.open(filepaths[0]).size

    xys = list(map(xy, filepaths))
    x_0, y_0 = min(map(lambda x_y: x_y[0], xys)), min(map(lambda x_y: x_y[1], xys))
    x_1, y_1 = max(map(lambda x_y: x_y[0], xys)), max(map(lambda x_y: x_y[1], xys))

    n_x, n_y = x_1 - x_0, y_1 - y_0

    out_w, out_h = n_x * tile_w, n_y * tile_h

    print("output image size:", out_w, out_h, "tile size:", tile_w, tile_h)

    out_img = Image.new("RGBA", (out_w, out_h), (0, 0, 255, 0))
    for iter_f, filepath in enumerate(filepaths):
        for_perc = (iter_f + 1) / len(filepaths) * 100
        if for_perc % 10 == 0:
            print(f"Progress: {for_perc}%\r", end="")
        x, y = xy(filepath)
        x, y = x - x_0, y - y_0
        tile = Image.open(filepath)
        out_img.paste(
            tile,
            box=(x * tile_w, y * tile_h, (x + 1) * tile_w, (y + 1) * tile_h),
        )

    print("\nSaving")

    if opts.pixel_closest is not None:
        basewidth = int(myround(float(out_img.size[0]), opts.pixel_closest))
        wpercent = basewidth / float(out_img.size[0])
        hsize = int(
            myround(
                int((float(out_img.size[1]) * float(wpercent))),
                opts.pixel_closest,
            )
        )
        out_img = out_img.resize((basewidth, hsize), Image.ANTIALIAS)

    out_img.save(opts.out_file)


def myround(x: Union[float, int], base: Optional[int] = 100) -> int:
    """
    Round a number to the nearest multiple of the given base.

    Parameters
    ----------
    x : float or int
        The number to be rounded.
    base : int, optional
        The base to which ``x`` will be rounded. Default is ``100``.

    Returns
    -------
    int
        The rounded number.

    ..
        TODO: Could we make this function private? It's only used by the
        runner above.
    """
    return base * round(x / base)


def stitcher(dir_name: str, out_file: str, pixel_closest: Optional[int] = None) -> None:
    """
    Stitch together multiple images from a directory and save the result to a
    file.

    Parameters
    ----------
    dir_name : str
        The directory containing the images to be stitched.
    out_file : str
        The name of the file to which the stitched image will be saved.
    pixel_closest : int or None
        The distance between the closest neighboring pixels. If ``None``, the
        optimal value will be determined automatically.

    Returns
    -------
    None
    """
    opts = input_class(name="stitcher")
    opts.dir = dir_name
    opts.out_file = out_file
    opts.pixel_closest = pixel_closest

    if not opts.pixel_closest:
        opts.pixel_closest = None

    runner(opts)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stitcher for tileserver

The main-part/most of these codes are from the following repo:

https://github.com/stamen/the-ultimate-tile-stitcher

(released under MIT license)

Here, we adapted the functions to run them via Python modules
"""

import glob
import os
from PIL import Image

from .tileserver_helpers import input_class


# -------
def runner(opts):
    search_path = os.path.join(opts.dir, "*_*_*.png")

    filepaths = glob.glob(search_path)

    def xy(filepath):
        base = os.path.basename(filepath)
        z, x, y = filepath.split("_")
        y = os.path.splitext(y)[0]
        return int(x), int(y)

    yx = lambda filepath: xy(filepath)[::-1]

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
            tile, box=(x * tile_w, y * tile_h, (x + 1) * tile_w, (y + 1) * tile_h)
        )

    print("\nSaving")

    if not opts.pixel_closest == None:
        basewidth = int(myround(float(out_img.size[0]), opts.pixel_closest))
        wpercent = basewidth / float(out_img.size[0])
        hsize = int(
            myround(int((float(out_img.size[1]) * float(wpercent))), opts.pixel_closest)
        )
        out_img = out_img.resize((basewidth, hsize), Image.ANTIALIAS)

    out_img.save(opts.out_file)


# -------
def myround(x, base=100):
    return base * round(x / base)


# -------
def stitcher(dir_name, out_file, pixel_closest):
    opts = input_class(name="stitcher")
    opts.dir = dir_name
    opts.out_file = out_file
    opts.pixel_closest = pixel_closest
    if not opts.pixel_closest:
        opts.pixel_closest = None
    runner(opts)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import PIL

PIL.Image.MAX_IMAGE_PIXELS = 800000000

import PIL.Image as PIL_image

# -------- patchifyByPixel
def patchifyByPixel(
    image_path,
    patch_size,
    path_save="patches",
    square_cuts=False,
    resize_factor=False,
    output_format="PNG",
    rewrite=False,
    verbose=True,
):
    """Patchify an image by pixels

    Parameters
    ----------
    image_path : str
        Path to image
    patch_size : int
        Number of pixels/meters in both x and y to use for slicing
    path_save : str, optional
        Directory to save the patches, by default "patches"
    square_cuts : bool, optional
        If True, all patches will have the same number of pixels in x and y, by default False
    resize_factor : bool, optional
        If True, resize the images before slicing, by default False
    output_format : str, optional
        Format to use when writing image files, by default "png"
    rewrite : bool, optional
        If True, existing patches will be rewritten, by default False
    verbose : bool, optional
        If True, progress updates will be printed throughout, by default False

    Returns
    -------
    list
        patches_info
    """
    # read image using PIL
    im = PIL_image.open(image_path)
    # resize the figure?
    if resize_factor:
        (im_width, im_height) = im.size
        im2patchify = im.resize(
            (int(im_width / resize_factor), int(im_height / resize_factor))
        )
    else:
        im2patchify = im

    (width, height) = im2patchify.size
    patches_info = []
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            max_x = min(x + patch_size, width)
            max_y = min(y + patch_size, height)

            if square_cuts:
                min_x = x - (patch_size - (max_x - x))
                min_y = y - (patch_size - (max_y - y))
            else:
                min_x = x
                min_y = y

            fname = os.path.join(
                path_save,
                "patch-%d-%d-%d-%d-#%s#"
                % (min_x, min_y, max_x, max_y, os.path.basename(image_path)),
            )
            path2save = "{}.{}".format(fname, output_format)

            if (not os.path.isfile(path2save)) or (rewrite):
                if verbose:
                    print(
                        f"Creating: {os.path.basename(fname):>50} -- #pixels in x/y: {max_x - min_x}/{max_y - min_y}"
                    )
                patch = im2patchify.crop((min_x, min_y, max_x, max_y))
                patch.save(path2save, output_format)
            else:
                if verbose:
                    print(f"File already exists: {path2save}")

            patches_info.append(
                [os.path.abspath(path2save), (min_x, min_y, max_x, max_y)]
            )

    return patches_info

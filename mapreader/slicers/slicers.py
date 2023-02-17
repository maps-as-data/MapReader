#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import PIL

PIL.Image.MAX_IMAGE_PIXELS = 800000000

import PIL.Image as PIL_image

# -------- sliceByPixel
def sliceByPixel(
    image_path,
    slice_size,
    path_save="sliced_images",
    square_cuts=False,
    resize_factor=False,
    output_format="png",
    rewrite=False,
    verbose=False,
):
    """Slice an image by pixels

    Parameters
    ----------
    image_path : str
        Path to image
    slice_size : int
        Number of pixels/meters in both x and y to use for slicing
    path_save : str, optional
        Directory to save the sliced image, by default "sliced_images"
    square_cuts : bool, optional
        If True, all sliced images will have the same number of pixels in x and y, by default False
    resize_factor : bool, optional
        If True, resize the images before slicing, by default False
    output_format : str, optional
        Format to use when writing image files, by default "png"
    rewrite : bool, optional
        If True, existing slices will be rewritten, by default False
    verbose : bool, optional
        If True, progress updates will be printed throughout, by default False

    Returns
    -------
    list
        sliced_images_info
    """

    # read image using PIL
    im = PIL_image.open(image_path)
    # resize the figure?
    if resize_factor:
        (im_width, im_height) = im.size
        im2slice = im.resize(
            (int(im_width / resize_factor), int(im_height / resize_factor))
        )
    else:
        im2slice = im

    (width, height) = im2slice.size
    sliced_images_info = []
    for y in range(0, height, slice_size):
        for x in range(0, width, slice_size):
            max_x = min(x + slice_size, width)
            max_y = min(y + slice_size, height)

            if square_cuts:
                min_x = x - (slice_size - (max_x - x))
                min_y = y - (slice_size - (max_y - y))
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
                patch = im2slice.crop((min_x, min_y, max_x, max_y))
                patch.save(path2save, output_format)
            else:
                if verbose:
                    print(f"File already exists: {path2save}")

            sliced_images_info.append(
                [os.path.abspath(path2save), (min_x, min_y, max_x, max_y)]
            )

    return sliced_images_info

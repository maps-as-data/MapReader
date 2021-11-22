#!/usr/bin/env python
# -*- coding: utf-8 -*-

from glob import glob
import numpy as np
import os
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import get_data_window
import subprocess
import distutils.spawn

# ------- preprocess_all
def preprocess_all(image_paths, save_preproc_dir, **kwds):
    """Preprocess a list of images

    Args:
        image_paths (list or path): a path (wildcard accepted) or a list of paths to images to be preprocessed 
        save_preproc_dir (str, path): path to save preprocessed images 
    """
    if not type(image_paths) == list:
        all_paths = glob(image_paths)
    else:
        all_paths = image_paths
    
    saved_paths = []
    for one_path in all_paths:
        print(f"Preprocessing: {one_path}")
        saved_path = preprocess(one_path, save_preproc_dir, **kwds)
        saved_paths.append(saved_path)
    return saved_paths

# -------- preprocess
def preprocess(image_path, save_preproc_dir, dst_crs='EPSG:3857', 
               crop_prefix="preproc_", reproj_prefix="preproc_tmp_", 
               resample_prefix="preproc_resample_", resize_percent=40,
               remove_reproj_file=True):
    """preprocess an image

    Preprocessing has three steps:
    - reproject maps to dst_crs
    - crop images by removing the white borders
    - resanme using resize_percent

    Args:
        image_path (str, path): path to an image to be preprocessed 
        save_preproc_dir (str, path): path to save preprocessed image
        dst_crs (str, optional): target map projection. Defaults to 'EPSG:3857'.
        crop_prefix (str, optional): prefix to cropped image filename. Defaults to "preproc_".
        reproj_prefix (str, optional): prefix to reprojected image filename. Defaults to "preproc_tmp_".
        resample_prefix (str, optional): prefix to resnamed image filename. Defaults to "preproc_resample_".
        resize_percent (int, optional): resize images by to this. Defaults to 40.
        remove_reproj_file (bool, optional): after preprocessing is finished, cleanup the files. Defaults to True.
    """
    
    # make sure that the output dir exists
    if not os.path.isdir(save_preproc_dir):
        os.makedirs(save_preproc_dir)
    
    # paths to reprojected, cropped and resampled images
    path2save_reproj = os.path.join(save_preproc_dir, reproj_prefix + os.path.basename(image_path))
    path2save_crop = os.path.join(save_preproc_dir, crop_prefix + os.path.basename(image_path))
    path2save_resample = os.path.join(save_preproc_dir, resample_prefix + os.path.basename(image_path))

    if os.path.isfile(path2save_resample):
        print(f"{path2save_resample} already exists!")
        return "False"

    with rasterio.open(image_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height})

        # --- reproject
        with rasterio.open(path2save_reproj, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
             
    # --- crop
    # make sure that the file is cropped correctly before proceeding to the next step
    cropped = False
    with rasterio.open(path2save_reproj) as src:
        try:
            window = get_data_window(src.read(1, masked=True))
            # window = Window(col_off=13, row_off=3, width=757, height=711)
    
            kwargs = src.meta.copy()
            kwargs.update({
                'height': window.height,
                'width': window.width,
                'transform': rasterio.windows.transform(window, src.transform)})
    
            with rasterio.open(path2save_crop, 'w', **kwargs) as dst:
                dst.write(src.read(window=window))

            cropped = True
        except Exception as e:
            print(e)

    # if not cropped correctly, clean up and exit
    if not cropped:
        os.remove(path2save_reproj)
        return "False"

    # make sure that the file is resampled correctly before proceeding to the next step
    resampled = False
    try:    
        gdal_exec = distutils.spawn.find_executable("gdal_translate")
        if not gdal_exec:
            err_msg = "gdal_translate could not be found. Refer to https://gdal.org/"
            raise ImportError(err_msg)

        # Run gdal using subprocess, in our experiments, this was faster than using the library
        gdal_command = f"{gdal_exec} -of GTiff -strict -outsize {resize_percent}% {resize_percent}% {path2save_crop} {path2save_resample}"
        # gdalwarp -t_srs EPSG:3857 -crop_to_cutline -overwrite 126517601.27.tif ./preproc/resize_126517601.27.tif
        subprocess.run(gdal_command, shell=True)
        resampled = True
    except Exception as e:
        print(e)
    
    if remove_reproj_file:
        os.remove(path2save_reproj)
        if resampled:
            os.remove(path2save_crop)
    if resampled:
        return path2save_resample
    else:
        return path2save_crop


    """
    # Another way of resampling:
    import rasterio
    from rasterio.enums import Resampling
    
    upscale_factor = 0.1
    
    with rasterio.open("./101201040.27.tif") as dataset:
    
        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * upscale_factor),
                int(dataset.width * upscale_factor)
            ),
            resampling=Resampling.bilinear
        )
    
        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )
    """

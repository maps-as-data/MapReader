#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from geopy.distance import geodesic, great_circle
except ImportError:
    pass
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import pandas as pd
from pylab import cm as pltcm
import pyproj
import rasterio


def extractGeoInfo(image_path, proj1='epsg:3857', proj2='epsg:4326', calc_size_in_m=False):
    """Extract geographic information (coordinates, size in meters) from GeoTiff files

    Args:
        image_path (str) -- Path to image (GeoTiff format)
        proj1 (str) -- Projection from proj1 ---> proj2, here, specify proj1. Defaults to 'epsg:3857'.
        proj2 (str) -- Projection from proj1 ---> proj2, here, specify proj2. Defaults to 'epsg:4326'.
        calc_size_in_m (bool, optional) -- Calculate size of the image (in meters). 
            Options: 'geodesic'; 'gc' or 'great-circle'; False ; Defaults to False.

    Returns:
        xmin, xmax, ymin, ymax, tiff_shape, size_in_m
    """
    # read the image using rasterio
    tiff_src = rasterio.open(image_path)
    tiff_shape = tiff_src.read().shape

    # Coordinate transformation: proj1 ---> proj2
    P1 = pyproj.Proj(proj1)
    P2 = pyproj.Proj(proj2)
    ymax, xmin = pyproj.transform(P1, P2, tiff_src.bounds.left, tiff_src.bounds.top)
    ymin, xmax = pyproj.transform(P1, P2, tiff_src.bounds.right, tiff_src.bounds.bottom)
    print(f"[INFO] Use the following coordinates to compute width/height:")
    print(f"[INFO] lon min/max: {xmin:.4f}/{xmax:.4f}")
    print(f"[INFO] lat min/max: {ymin:.4f}/{ymax:.4f}")
    print(f"[INFO] shape: {tiff_shape}")

    # Calculate the size of image in meters 
    if calc_size_in_m == 'geodesic':
        bottom = geodesic((ymin, xmin), (ymin, xmax)).meters
        right = geodesic((ymin, xmax), (ymax, xmax)).meters
        top = geodesic((ymax, xmax), (ymax, xmin)).meters
        left = geodesic((ymax, xmin), (ymin, xmin)).meters
        size_in_m = (bottom, top, left, right) 
        print(f"[INFO] size (in meters) bottom/top/left/right: {bottom:.2f}/{top:.2f}/{left:.2f}/{right:.2f}")

        mean_width = np.mean([size_in_m[0]/tiff_shape[2], size_in_m[1]/tiff_shape[2]])
        mean_height = np.mean([size_in_m[2]/tiff_shape[1], size_in_m[3]/tiff_shape[1]])
        print(f"Each pixel is ~{mean_width:.3f} X {mean_height:.3f} meters (width x height).")
    elif calc_size_in_m in ['gc', 'great-circle']:
        bottom = great_circle((ymin, xmin), (ymin, xmax)).meters
        right = great_circle((ymin, xmax), (ymax, xmax)).meters
        top = great_circle((ymax, xmax), (ymax, xmin)).meters
        left = great_circle((ymax, xmin), (ymin, xmin)).meters
        size_in_m = (bottom, top, left, right) 
        print(f"[INFO] size (in meters) bottom/top/left/right: {bottom:.2f}/{top:.2f}/{left:.2f}/{right:.2f}")

        mean_width = np.mean([size_in_m[0]/tiff_shape[2], size_in_m[1]/tiff_shape[2]])
        mean_height = np.mean([size_in_m[2]/tiff_shape[1], size_in_m[3]/tiff_shape[1]])
        print(f"Each pixel is ~{mean_width:.3f} x {mean_height:.3f} meters (width x height).")
    else:
        size_in_m = False

    return xmin, xmax, ymin, ymax, tiff_shape, size_in_m

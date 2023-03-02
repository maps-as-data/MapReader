#!/usr/bin/env python
# -*- coding: utf-8 -*-

from geopy.distance import geodesic, great_circle
import rasterio
import numpy as np
from pyproj import Transformer


def extractGeoInfo(image_path, proj2convert="EPSG:4326", calc_size_in_m=False):
    """Extract geographic information (coordinates, size in meters) from GeoTiff files

    Parameters
    ----------
    image_path : str
        Path to image
    proj2convert : str, optional
        Projection to convert coordinates into, by default "EPSG:4326"
    calc_size_in_m : str or bool, optional
        Method to compute pixel widths and heights, choices between "geodesic" and "great-circle" or "gc", by default "great-circle", by default False

    Returns
    -------
    list
        coords, tiff_shape, size_in_m
    """
    # read the image using rasterio
    tiff_src = rasterio.open(image_path)
    h, w = tiff_src.shape
    c = tiff_src.count
    tiff_shape = (h, w, c)

    # check coordinates are present
    if tiff_src.crs != None:
        tiff_proj = tiff_src.crs.to_string()
    else:
        raise ValueError(f"No coordinates found in {image_path}")

    # Coordinate transformation: proj1 ---> proj2
    transformer = Transformer.from_crs(tiff_proj, proj2convert)
    ymax, xmin = transformer.transform(tiff_src.bounds.left, tiff_src.bounds.top)
    ymin, xmax = transformer.transform(tiff_src.bounds.right, tiff_src.bounds.bottom)
    coords = (xmin, xmax, ymin, ymax)

    print(f"[INFO] Use the following coordinates to compute width/height:")
    print(f"[INFO] lon min/max: {xmin:.4f}/{xmax:.4f}")
    print(f"[INFO] lat min/max: {ymin:.4f}/{ymax:.4f}")
    print(f"[INFO] shape: {tiff_shape}")

    # Calculate the size of image in meters
    if calc_size_in_m == "geodesic":
        bottom = geodesic((ymin, xmin), (ymin, xmax)).meters
        right = geodesic((ymin, xmax), (ymax, xmax)).meters
        top = geodesic((ymax, xmax), (ymax, xmin)).meters
        left = geodesic((ymax, xmin), (ymin, xmin)).meters
        size_in_m = (bottom, top, left, right)
        print(
            f"[INFO] size (in meters) bottom/top/left/right: {bottom:.2f}/{top:.2f}/{left:.2f}/{right:.2f}"
        )

        mean_width = np.mean(
            [size_in_m[0] / tiff_shape[2], size_in_m[1] / tiff_shape[2]]
        )
        mean_height = np.mean(
            [size_in_m[2] / tiff_shape[1], size_in_m[3] / tiff_shape[1]]
        )
        print(
            f"Each pixel is ~{mean_width:.3f} X {mean_height:.3f} meters (width x height)."
        )
    elif calc_size_in_m in ["gc", "great-circle"]:
        bottom = great_circle((ymin, xmin), (ymin, xmax)).meters
        right = great_circle((ymin, xmax), (ymax, xmax)).meters
        top = great_circle((ymax, xmax), (ymax, xmin)).meters
        left = great_circle((ymax, xmin), (ymin, xmin)).meters
        size_in_m = (bottom, top, left, right)
        print(
            f"[INFO] size (in meters) bottom/top/left/right: {bottom:.2f}/{top:.2f}/{left:.2f}/{right:.2f}"
        )

        mean_width = np.mean(
            [size_in_m[0] / tiff_shape[2], size_in_m[1] / tiff_shape[2]]
        )
        mean_height = np.mean(
            [size_in_m[2] / tiff_shape[1], size_in_m[3] / tiff_shape[1]]
        )
        print(
            f"Each pixel is ~{mean_width:.3f} x {mean_height:.3f} meters (width x height)."
        )
    else:
        size_in_m = False

    return coords, tiff_shape, size_in_m

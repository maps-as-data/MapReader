#!/usr/bin/env python
# -*- coding: utf-8 -*-

from geopy.distance import geodesic, great_circle
import rasterio
import numpy as np
from pyproj import Transformer


def extractGeoInfo(image_path):
    """Extract geographic information (shape, CRS and coordinates) from GeoTiff files

    Parameters
    ----------
    image_path : str
        Path to image

    Returns
    -------
    list
        shape, CRS, coord
    """
    # read the image using rasterio
    tiff_src = rasterio.open(image_path)
    image_height, image_width = tiff_src.height, tiff_src.width
    image_channels = tiff_src.count
    tiff_shape = (image_height, image_width, image_channels)

    # check coordinates are present
    if isinstance(tiff_src.crs, type(None)):
        raise ValueError(f"No coordinates found in {image_path}")
    else:
        tiff_proj = tiff_src.crs.to_string()
        tiff_coord = tuple(tiff_src.bounds)

    print(f"[INFO] Shape: {tiff_shape}. \n[INFO] CRS: {tiff_proj}.")
    print("[INFO] Coordinates: %.4f %.4f %.4f %.4f" % tiff_coord)

    return tiff_shape, tiff_proj, tiff_coord


def reprojectGeoInfo(image_path, proj2convert="EPSG:4326", calc_size_in_m=False):
    """Extract geographic information from GeoTiff files and reproject to specified CRS (`proj2convert`).

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
        shape, old CRS, new CRS, reprojected coord, size in meters
    """
    tiff_shape, tiff_proj, tiff_coord = extractGeoInfo(image_path)

    # Coordinate transformation: proj1 ---> proj2
    transformer = Transformer.from_crs(tiff_proj, proj2convert)
    ymin, xmin = transformer.transform(
        tiff_coord[0], tiff_coord[1]
    )
    ymax, xmax = transformer.transform(
        tiff_coord[2], tiff_coord[3]
    )
    coord = (xmin, ymin, xmax, ymax)

    print(f"[INFO] New CRS: {proj2convert}")
    print("[INFO] Reprojected coordinates: %.4f %.4f %.4f %.4f" % coord)

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

    return tiff_shape, tiff_proj, proj2convert, coord, size_in_m

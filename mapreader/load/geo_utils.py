#!/usr/bin/env python
from __future__ import annotations

import geopandas as gpd
import numpy as np
import rasterio
from geopy.distance import geodesic, great_circle
from pyproj import Transformer
from rasterio.features import geometry_mask
from shapely.geometry import box


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
    print("[INFO] Coordinates: {:.4f} {:.4f} {:.4f} {:.4f}".format(*tiff_coord))

    return tiff_shape, tiff_proj, tiff_coord


def reproject_geo_info(image_path, target_crs="EPSG:4326", calc_size_in_m=False):
    """Extract geographic information from GeoTiff files and reproject to specified CRS (`target_crs`).

    Parameters
    ----------
    image_path : str
        Path to image
    target_crs : str, optional
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
    transformer = Transformer.from_crs(tiff_proj, target_crs, always_xy=True)
    coord = transformer.transform_bounds(*tiff_coord)
    print(f"[INFO] New CRS: {target_crs}")
    print("[INFO] Reprojected coordinates: {:.4f} {:.4f} {:.4f} {:.4f}".format(*coord))

    height, width, _ = tiff_shape

    # Calculate the size of image in meters
    xmin, ymin, xmax, ymax = coord
    if calc_size_in_m:
        if calc_size_in_m in ["geodesic", "gd"]:
            bottom = geodesic((ymin, xmin), (ymin, xmax)).meters
            right = geodesic((ymin, xmax), (ymax, xmax)).meters
            top = geodesic((ymax, xmax), (ymax, xmin)).meters
            left = geodesic((ymax, xmin), (ymin, xmin)).meters

        elif calc_size_in_m in ["gc", "great-circle", "great_circle"]:
            bottom = great_circle((ymin, xmin), (ymin, xmax)).meters
            right = great_circle((ymin, xmax), (ymax, xmax)).meters
            top = great_circle((ymax, xmax), (ymax, xmin)).meters
            left = great_circle((ymax, xmin), (ymin, xmin)).meters

        else:
            raise NotImplementedError(
                f'[ERROR] ``calc_size_in_m`` must be one of "great-circle", "great_circle", "gc", "geodesic" or "gd", not: {calc_size_in_m}'
            )

        size_in_m = (left, bottom, right, top)  # anticlockwise order

        mean_pixel_height = np.mean([right / height, left / height])
        mean_pixel_width = np.mean([bottom / width, top / width])

        print(
            f"[INFO] Size in meters of left/bottom/right/top: {left:.2f}/{bottom:.2f}/{right:.2f}/{top:.2f}"
        )
        print(
            f"Each pixel is ~{mean_pixel_height:.3f} X {mean_pixel_width:.3f} meters (height x width)."
        )  # noqa

    else:
        size_in_m = None

    return tiff_shape, tiff_proj, target_crs, coord, size_in_m


def apply_mask_to_raster(
    input_tif: str,
    gdf: gpd.GeoDataFrame,
    output_tif: str,
    mask_color: str = "white",
    buffer_distance: float = 0,
):
    """
    Apply a geospatial mask to a GeoTIFF file based on a GeoDataFrame, clipping geometries to the raster bounds.

    Parameters:
    - input_tif (str): Path to the input GeoTIFF file.
    - gdf (geopandas.GeoDataFrame): GeoDataFrame containing geometries to mask.
    - output_tif (str): Path to save the masked GeoTIFF.
    - mask_color (str): Color of the mask, either "white" or "black". Defaults to "white".
    - buffer_distance (float): Buffer distance in meters for polylines. Defaults to 0 (no buffering).

    Returns:
    - None
    """
    if mask_color not in ["white", "black"]:
        raise ValueError("mask_color must be either 'white' or 'black'")

    with rasterio.open(input_tif) as src:
        gdf = gdf.to_crs(src.crs)

        # Clip geometries to raster bounds
        raster_bounds = box(*src.bounds)
        raster_bbox = gpd.GeoDataFrame({"geometry": [raster_bounds]}, crs=src.crs)
        gdf = gpd.clip(gdf, raster_bbox)

        # Buffer polylines if applicable
        if buffer_distance > 0:
            gdf["geometry"] = gdf.geometry.apply(
                lambda geom: geom.buffer(buffer_distance)
                if geom.geom_type in ["LineString", "MultiLineString"]
                else geom
            )

        # Read raster data
        raster_data = src.read()
        transform = src.transform
        out_meta = src.meta.copy()

        # Create the mask
        mask = geometry_mask(
            [geom for geom in gdf.geometry if geom.is_valid],
            transform=transform,
            invert=True,
            out_shape=(src.height, src.width),
        )

        # Apply the mask
        masked_raster = np.copy(raster_data)
        fill_value = 255 if mask_color == "white" else 0
        for band in range(masked_raster.shape[0]):
            masked_raster[band][mask] = fill_value

        # Save the masked raster
        with rasterio.open(output_tif, "w", **out_meta) as dst:
            dst.write(masked_raster)

    print(f"Masked GeoTIFF saved to {output_tif}")

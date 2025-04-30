from __future__ import annotations

import io
import os
import pathlib

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import requests
from bs4 import BeautifulSoup
from piffle.iiif_dataclasses.presentation2 import IIIFPresentation2
from piffle.iiif_dataclasses.presentation3 import (
    GeoreferenceAnnotation3,
    IIIFPresentation3,
)
from piffle.load_iiif import load_iiif_image, load_iiif_presentation
from PIL import Image, ImageDraw
from rasterio import transform
from rasterio.plot import reshape_as_raster
from shapely import Polygon
from tqdm.auto import tqdm


class IIIFDownloader:
    def __init__(
        self,
        iiif: str
        | IIIFPresentation3
        | IIIFPresentation2
        | list[str | IIIFPresentation3 | IIIFPresentation2],
        iiif_versions: int | float | str | list[int | float | str] | None = None,
        iiif_uris: str | list[str] | None = None,
    ):
        """A class to download images from IIIF presentation jsons.

        Parameters
        ----------
        iiif : str | IIIFPresentation3 | IIIFPresentation2 | list[str  |  IIIFPresentation3  |  IIIFPresentation2]
            The IIIF url or IIIF presentation API object(s) containing the maps to download
        iiif_versions : int | float | str | list[int | float | str] | None
            The IIIF version(s) of the IIIF object(s). Can only be None if `iiif` is/are already IIIF object(s). Default is None.
        iiif_uris : str | list[str] | None
            The URI(s) of the IIIF object(s), needed if the IIIF object is missing an 'id' field. Default is None.

            If only one IIIF object is missing an `id`, create a list with None values for the other objects.
            e.g. iiif_uri=[None, "https://example.com/iiif/manifest.json", None]
        """
        if not isinstance(iiif, list):
            iiif = [iiif]
        if iiif_versions is None:
            iiif_versions = [None] * len(iiif)
        if not isinstance(iiif_versions, list):
            iiif_versions = [iiif_versions]
        if len(iiif) != len(iiif_versions):
            raise ValueError(
                "[ERROR] Length of `iiif` must match length of `iiif_versions`."
            )

        iiif_objs = []
        for iiif_obj, iiif_version in zip(iiif, iiif_versions):
            if isinstance(iiif_obj, str):
                iiif_obj = load_iiif_presentation(iiif_obj, iiif_version)
            if isinstance(iiif_obj, (IIIFPresentation3, IIIFPresentation2)):
                iiif_objs.append(iiif_obj)
            else:
                raise ValueError("[ERROR] Each `iiif` must be a string or IIIF object.")

        if iiif_uris is None:
            iiif_uris = [None] * len(iiif_objs)
        if not isinstance(iiif_uris, list):
            iiif_uris = [iiif_uris]
        if len(iiif_uris) != len(iiif_objs):
            raise ValueError(
                "[ERROR] Length of `iiif_uris` must match length of `iiif`."
            )

        for iiif_obj, iiif_uri in zip(iiif_objs, iiif_uris):
            if not hasattr(iiif_obj, "id"):
                if iiif_uri is not None:
                    iiif_obj.id = iiif_uri
                else:
                    raise ValueError(
                        "[ERROR] One of your IIIF objects is missing an 'id' field so we cannot identify it's URL. Please pass the `iiif_uris` argument."
                    )

        self.iiif = iiif_objs

    def save_georeferenced_maps(
        self,
        path_save: str | pathlib.Path = "maps",
    ):
        """Save georefereced maps from a IIIF presentation 3 json as tiffs.

        Parameters
        ----------
        path_save : str | pathlib.Path
            Path to save the images

        Notes
        -----
        Calls the `save_georeferenced_map` method for each IIIF object.
        """
        for iiif in self.iiif:
            self.save_georeferenced_map(iiif, path_save=path_save)

    def save_georeferenced_map(
        self,
        iiif: str | IIIFPresentation3,
        path_save: str | pathlib.Path = "maps",
        iiif_uri: str = None,
    ):
        """Save a single georefereced map from a IIIF presentation 3 json as a tiff.

        Parameters
        ----------
        iiif : str | dict | IIIFPresentation3
            the IIIF url or IIIF presentation API object containing the map(s)
        path_save : str | pathlib.Path
            Path to save the images
        iiif_uri : str
            The URI of the IIIF object, needed if the IIIF object is missing an 'id' field.

        Notes
        -----
        Only first order polynomial transformations are currently supported, see the [allmaps documentation](https://allmaps.org/) for more information on the transformation options.

        Raises
        ------
        ValueError
            Only first order polynomial transformations are currently supported.
        """
        if isinstance(iiif, str):
            iiif_obj = load_iiif_presentation(iiif, 3)
        elif isinstance(iiif, IIIFPresentation3):
            iiif_obj = iiif
        else:
            raise ValueError("[ERROR] `iiif` must be a string or IIIF3 object.")

        if not hasattr(iiif_obj, "id"):
            if iiif_uri is not None:
                iiif_obj.id = iiif_uri
            else:
                raise ValueError(
                    "[ERROR] IIIF object is missing an 'id' field so we cannot identify its URI. Please manually pass the `iiif_uri` argument."
                )
        iiif_uri = iiif_obj.id

        metadata = pd.DataFrame(columns=["name", "id", "iiif_uri"])
        for annot in tqdm(iiif_obj.collect_annotations()):
            if not isinstance(annot, GeoreferenceAnnotation3):
                print(
                    f"[WARNING] Skipping annotation '{annot.id}' as it is not a georeference annotation."
                )
                continue

            # Check we are working with first order polynomial transformation (i.e. affine)
            if (annot.body["transformation"]["type"] != "polynomial") or (
                annot.body["transformation"]["options"]["order"] != 1
            ):
                raise ValueError(
                    "[ERROR] Only first order polynomial transformations are currently supported"
                )

            # Get filename
            # host, prefix, identifier
            id = annot.resource["service"]["id"] if annot.id is None else annot.id
            fname = ".".join(id.removeprefix("https://").split("/")[2:])

            if not os.path.exists(path_save):
                os.makedirs(path_save, exist_ok=True)
            if os.path.exists(f"{path_save}/{fname}.tif"):
                print(f"[INFO] '{fname}' already exists. Skipping download.")
                continue

            # Download image
            image = self.download_image(3, annot)

            # Get GCPs
            features = gpd.GeoDataFrame.from_features(
                annot.body["features"], crs="WGS84"
            )
            resource_points = features[
                "resourceCoords"
            ].to_list()  # in image coordinates
            gcp_points = features["geometry"].to_list()  # in WGS84

            # resource points are x, y (i.e col, row)
            # gcp points are lon, lat (i.e x, y)

            # Create affine transform
            affine = transform.from_gcps(
                [
                    transform.GroundControlPoint(
                        resource_point[1], resource_point[0], gcp_point.x, gcp_point.y
                    )
                    for resource_point, gcp_point in zip(resource_points, gcp_points)
                ]
            )

            # Idenfity annotation bounds
            svg = BeautifulSoup(annot.target["selector"]["value"], "lxml").find("svg")
            points = svg.find("polygon")["points"]
            points = np.array(
                [
                    [int(x), int(y)]
                    for x, y in [point.split(",") for point in points.split()]
                ]
            )  # points are x, y (i.e col, row)

            # Apply affine transform to annotation bounds
            t = transform.AffineTransformer(affine)
            x1, y1 = t.xy(*points[1][::-1])  # input as row, col (i.e y, x)
            x0, y0 = t.xy(*points[0][::-1])
            x2, y2 = t.xy(*points[2][::-1])
            x3, y3 = t.xy(*points[3][::-1])

            # Save tiff
            height, width, channels = image.height, image.width, len(image.getbands())

            with rasterio.open(
                f"{path_save}/{fname}.tif",
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=channels,
                transform=affine,
                dtype="uint8",
                nodata=0,
                crs="WGS84",
            ) as dst:
                array = reshape_as_raster(
                    image
                )  # (row, col, bands) to (bands, row, col)
                dst.write(array)

            # Mask tiff to annotation bounds
            tiff_image = rasterio.open(f"{path_save}/{fname}.tif")

            polygon = Polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3)])
            masked_image, out_transform = rasterio.mask.mask(
                tiff_image, [polygon], crop=False
            )
            out_meta = tiff_image.meta.copy()

            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": masked_image.shape[1],
                    "width": masked_image.shape[2],
                    "transform": out_transform,
                }
            )

            with rasterio.open(
                f"{path_save}/{fname}_masked.tif", "w", **out_meta
            ) as dst:
                dst.write(masked_image)

            metadata.loc[len(metadata)] = [f"{fname}_masked.tif", fname, iiif_obj.id]
            metadata.to_csv(f"{path_save}/metadata.csv")

    def save_maps(
        self,
        path_save: str | pathlib.Path = "maps",
    ):
        """Save maps from a IIIF presentation json as pngs.

        Parameters
        ----------
        path_save : str | pathlib.Path
            Path to save the images

        Notes
        -----
        Calls the `save_map` method for each IIIF object.
        """
        for iiif in self.iiif:
            self.save_map(iiif, path_save=path_save)

    def save_map(
        self,
        iiif: str | IIIFPresentation3 | IIIFPresentation2,
        iiif_version: int | float | str = 3,
        path_save: str | pathlib.Path = "maps",
        iiif_uri: str = None,
    ):
        """Save a single map from a IIIF presentation json as a png.

        Parameters
        ----------
        iiif : str | dict | IIIFPresentation3 | IIIFPresentation2
            the IIIF url or IIIF presentation API object containing the map(s)
        iiif_version : int | float | str
            The IIIF version, ignored if iiif is already a IIIF object
        path_save : str | pathlib.Path
            Path to save the images
        iiif_uri : str
            The URI of the IIIF object, needed if the IIIF object is missing an 'id' field.
        """
        if isinstance(iiif, str):
            iiif_obj = load_iiif_presentation(iiif, iiif_version)
        elif isinstance(iiif, (IIIFPresentation3, IIIFPresentation2)):
            iiif_obj = iiif
        else:
            raise ValueError("`iiif` must be a string or IIIF object.")

        if not hasattr(iiif_obj, "id"):
            if iiif_uri is not None:
                iiif_obj.id = iiif_uri
            else:
                raise ValueError(
                    "[ERROR] IIIF object is missing an 'id' field so we cannot identify its URL. Please manually pass the `iiif_url` argument."
                )
        iiif_uri = iiif_obj.id

        metadata = pd.DataFrame(columns=["filename", "iiif_uri"])
        for annot in tqdm(iiif_obj.collect_annotations()):
            # Get filename
            # host, prefix, identifier
            id = annot.resource["service"]["id"] if annot.id is None else annot.id
            fname = ".".join(id.removeprefix("https://").split("/")[2:])

            if not os.path.exists(path_save):
                os.makedirs(path_save, exist_ok=True)
            if os.path.exists(f"{path_save}/{fname}.png"):
                print(f"[INFO] '{fname}' already exists. Skipping download.")
                continue

            metadata.loc[len(metadata)] = [fname, iiif_obj.id]

            # Download image
            image = self.download_image(iiif_version, annot)
            image.save(f"{path_save}/{fname}.png")

            if isinstance(iiif_obj, IIIFPresentation3):
                if "selector" in annot.target.keys():
                    # Idenfity annotation bounds
                    svg = BeautifulSoup(annot.target["selector"]["value"], "xml").find(
                        "svg"
                    )
                    points = svg.find("polygon")["points"]
                    points = np.array(
                        [
                            [int(x), int(y)]
                            for x, y in [point.split(",") for point in points.split()]
                        ]
                    )  # points are x, y (i.e col, row)

                    # -- Mask image to annotation bounds
                    # Convert to numpy array
                    image_array = np.array(image)

                    # Create mask
                    polygon = Polygon(points)
                    mask_image = Image.new(
                        "L", (image_array.shape[1], image_array.shape[0]), 0
                    )
                    ImageDraw.Draw(mask_image).polygon(
                        polygon.exterior.coords, outline=1, fill=1
                    )
                    mask = np.array(mask_image)

                    # Apply mask
                    image_array = image_array * mask[:, :, None]

                    masked_image = Image.fromarray(image_array)
                    masked_image.save(f"{path_save}/{fname}_masked.png")

            metadata.to_csv(f"{path_save}/metadata.csv")

    def download_image(iiif, iiif_version, annot):
        image_url = annot.get_image_url()
        image_manifest = load_iiif_image(
            image_url, iiif_version
        )  # TODO: method to get image version

        width = image_manifest.width
        height = image_manifest.height
        tile_width = image_manifest.tiles[0]["width"]
        tile_height = image_manifest.tiles[0].get("height", tile_width)

        image = Image.new("RGB", (width, height))
        for x in np.arange(0, ((width // tile_width) + 1) * tile_width, tile_width):
            for y in np.arange(
                0, ((height // tile_height) + 1) * tile_height, tile_height
            ):
                if x + tile_width > width:
                    x = width - tile_width
                if y + tile_height > height:
                    y = height - tile_height
                response = requests.get(
                    f"{image_url}/{x},{y},{tile_width},{tile_height}/full/0/default.jpg"
                )
                response.raise_for_status()

                tile = Image.open(io.BytesIO(response.content))
                image.paste(tile, (x, y))
        return image

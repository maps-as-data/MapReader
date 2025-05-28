from __future__ import annotations

import os
import pathlib
import re
from itertools import combinations

import geopandas as gpd
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xyzservices as xyz
from PIL import Image
from shapely import LineString, MultiPolygon, Polygon, from_wkt
from tqdm.auto import tqdm

from mapreader import MapImages
from mapreader.utils.load_frames import eval_dataframe, load_from_csv, load_from_geojson

from .dataclasses import GeoPrediction, ParentPrediction, PatchPrediction


class DetRunner:
    def __init__() -> None:
        """Initialise the DetRunner class."""
        # empty in the base class

    def _load_df(
        self,
        patch_df: pd.DataFrame | gpd.GeoDataFrame | str | pathlib.Path,
        parent_df: pd.DataFrame | gpd.GeoDataFrame | str | pathlib.Path,
        delimiter: str = ",",
    ):
        """Load the patch and parent dataframes.

        Parameters
        ----------
        patch_df : pd.DataFrame or gpd.GeoDataFrame or str or pathlib.Path
            The dataframe containing patch information. If a string, it should be the path to a csv file.
        parent_df : pd.DataFrame or gpd.GeoDataFrame or str or pathlib.Path
            The dataframe containing parent information. If a string, it should be the path to a csv file.
        delimiter : str, optional
            The delimiter used in the csv file, by default ",".
        """
        if isinstance(patch_df, pd.DataFrame):
            self.patch_df = patch_df

        elif isinstance(patch_df, (str, pathlib.Path)):
            print(f'[INFO] Reading "{patch_df}"')
            if re.search(r"\..?sv$", str(patch_df)):
                self.patch_df = load_from_csv(
                    patch_df,
                    delimiter=delimiter,
                )
            elif re.search(r"\..*?json$", str(patch_df)):
                self.patch_df = load_from_geojson(patch_df)
            else:
                raise ValueError(
                    "[ERROR] ``patch_df`` must be a path to a CSV/TSV/etc or geojson file, a pandas DataFrame or a geopandas GeoDataFrame."
                )

        else:
            raise ValueError(
                "[ERROR] ``patch_df`` must be a path to a CSV/TSV/etc or geojson file, a pandas DataFrame or a geopandas GeoDataFrame."
            )

        if parent_df is None:
            self.parent_df = pd.DataFrame()  # empty dataframe

        elif isinstance(parent_df, pd.DataFrame):
            self.parent_df = parent_df

        elif isinstance(parent_df, (str, pathlib.Path)):
            print(f'[INFO] Reading "{parent_df}"')
            if re.search(r"\..?sv$", str(parent_df)):
                self.parent_df = load_from_csv(
                    parent_df,
                    delimiter=delimiter,
                )
            elif re.search(r"\..*?json$", str(parent_df)):
                self.parent_df = load_from_geojson(parent_df)
            else:
                raise ValueError(
                    "[ERROR] ``parent_df`` must be a path to a CSV/TSV/etc or geojson file, a pandas DataFrame or a geopandas GeoDataFrame."
                )

        else:
            raise ValueError(
                "[ERROR] ``parent_df`` must be a path to a CSV/TSV/etc or geojson file, a pandas DataFrame or a geopandas GeoDataFrame."
            )

        self.check_georeferencing()

    def check_georeferencing(self):
        if "coordinates" in self.parent_df.columns:
            if "dlat" in self.parent_df.columns and "dlon" in self.parent_df.columns:
                self.georeferenced = True
            else:
                self._add_coord_increments()
                self.check_georeferencing()
        else:
            print(
                "[WARNING] Will not be able to georeference results, please ensure parent_df has 'coordinates' column."
            )
            self.georeferenced = False

    def _add_coord_increments(self):
        maps = MapImages()
        maps.load_df(self.parent_df)
        maps.add_coord_increments()
        parent_df, _ = maps.convert_images()
        self.parent_df = parent_df

    @staticmethod
    def _dict_to_dataframe(
        preds: dict,
    ) -> pd.DataFrame:
        """Convert the predictions dictionary to a pandas DataFrame.

        Parameters
        ----------
        preds : dict
            A dictionary of predictions.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the predictions.
        """

        if len(preds):
            preds_df = pd.concat(
                pd.DataFrame(
                    preds[k],
                    index=np.full(len(preds[k]), k),
                )
                for k in preds.keys()
            )
            # drop empty cols
            preds_df.dropna(inplace=True, axis=1)

            if "crs" in preds_df.columns:
                # get the crs (should be the same for all)
                if not preds_df["crs"].nunique() == 1:
                    raise ValueError("[ERROR] Multiple crs found in the predictions.")
                crs = preds_df["crs"].unique()[0]

                preds_df = gpd.GeoDataFrame(
                    preds_df,
                    geometry="geometry",
                    crs=crs,
                )
        else:
            preds_df = pd.DataFrame()  # empty dataframe

        preds_df.index.name = "image_id"
        preds_df.reset_index(inplace=True)  # reset index to get image_id as a column
        return preds_df

    def run_all(
        self,
        return_dataframe: bool = False,
        min_ioa: float = 0.7,
    ) -> dict | pd.DataFrame:
        """Run the model on all images in the patch dataframe.

        Parameters
        ----------
        return_dataframe : bool, optional
            Whether to return the predictions as a pandas DataFrame, by default False
        min_ioa : float, optional
            The minimum intersection over area to consider two polygons the same, by default 0.7

        Returns
        -------
        dict or pd.DataFrame or gpd.GeoDataFrame
            A dictionary of predictions for each patch image or a DataFrame if `return_dataframe` is True.
        """
        img_paths = self.patch_df["image_path"].to_list()

        patch_predictions = self.run_on_images(
            img_paths, return_dataframe=return_dataframe, min_ioa=min_ioa
        )
        return patch_predictions

    def run_on_images(
        self,
        img_paths: str | pathlib.Path | list,
        return_dataframe: bool = False,
        min_ioa: float = 0.7,
    ) -> dict | pd.DataFrame:
        """Run the model on a list of images.

        Parameters
        ----------
        img_paths : str, pathlib.Path or list
            A list of image paths to run the model on.
        return_dataframe : bool, optional
            Whether to return the predictions as a pandas DataFrame, by default False
        min_ioa : float, optional
            The minimum intersection over area to consider two polygons the same, by default 0.7

        Returns
        -------
        dict or pd.DataFrame
            A dictionary of predictions for each image or a DataFrame if `return_dataframe` is True.
        """

        if isinstance(img_paths, (str, pathlib.Path)):
            img_paths = [img_paths]

        for img_path in tqdm(img_paths):
            _ = self.run_on_image(img_path, return_outputs=False, min_ioa=min_ioa)

        if return_dataframe:
            return self._dict_to_dataframe(self.patch_predictions)
        return self.patch_predictions

    def run_on_image(
        self,
        img_path: str | pathlib.Path,
        return_outputs=False,
        return_dataframe: bool = False,
        min_ioa: float = 0.7,
    ) -> dict | pd.DataFrame:
        """Run the model on a single image.

        Parameters
        ----------
        img_path : str or pathlib.Path
            The path to the image to run the model on.
        return_outputs : bool, optional
            Whether to return the outputs direct from the model, by default False
        return_dataframe : bool, optional
            Whether to return the predictions as a pandas DataFrame, by default False
        min_ioa : float, optional
            The minimum intersection over area to consider two polygons the same, by default 0.7

        Returns
        -------
        dict or pd.DataFrame
            The predictions for the image or the outputs from the model if `return_outputs` is True.
        """
        # load image
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)

        # run inference
        outputs = self.predictor(img_array)
        outputs["image_id"] = os.path.basename(img_path)
        outputs["img_path"] = img_path

        if return_outputs:
            return outputs

        patch_predictions = self._get_patch_predictions(
            outputs, return_dataframe=return_dataframe, min_ioa=min_ioa
        )
        return patch_predictions

    def _deduplicate(self, image_id, min_ioa=0.7):
        polygons = [
            instance.pixel_geometry for instance in self.patch_predictions[image_id]
        ]

        def calc_ioa(polygons, i, j):
            return polygons[i].intersection(polygons[j]).area / polygons[i].area

        # create an intersection matrix
        i_matrix = np.zeros((len(polygons), len(polygons)))

        for i, j in combinations(range(len(polygons)), 2):
            if polygons[i].intersects(polygons[j]):
                ioa_i = calc_ioa(polygons, i, j)
                ioa_j = calc_ioa(polygons, j, i)
                # bigger ioa means more overlap, so keep the one with the bigger ioa (only if more than min_ioa)
                if ioa_i > ioa_j and ioa_i > min_ioa:
                    i_matrix[i, j] = ioa_i
                elif ioa_j > ioa_i and ioa_j > min_ioa:
                    i_matrix[j, i] = ioa_j

        # if there is an intersection with another polygon then remove this polygon
        for i, row in enumerate(i_matrix):
            if any([ioa != 0 for ioa in row]):
                self.patch_predictions[image_id][i] = None
                i_matrix[:, i] = 0

        # remove the None values
        self.patch_predictions[image_id] = [
            prediction
            for prediction in self.patch_predictions[image_id]
            if prediction is not None
        ]

    def convert_to_parent_pixel_bounds(
        self,
        return_dataframe: bool = False,
        deduplicate: bool = False,
        min_ioa: float = 0.7,
    ) -> dict | pd.DataFrame:
        """Convert the patch predictions to parent predictions by converting pixel bounds.

        Parameters
        ----------
        return_dataframe : bool, optional
            Whether to return the predictions as a pandas DataFrame, by default False
        deduplicate : bool, optional
            Whether to deduplicate the parent predictions, by default False.
            Depending on size of parent images, this can be slow.
        min_ioa : float, optional
            The minimum intersection over area to consider two polygons the same, by default 0.7
            This is only used if `deduplicate` is True.

        Returns
        -------
        dict or pd.DataFrame
            A dictionary of predictions for each parent image or a DataFrame if `return_dataframe` is True.
        """
        self.parent_predictions = {}  # reset parent predictions

        for image_id, prediction in self.patch_predictions.items():
            parent_id = self.patch_df.loc[image_id, "parent_id"]
            if parent_id not in self.parent_predictions.keys():
                self.parent_predictions[parent_id] = []

            for instance in prediction:
                # convert polygon
                polygon = instance.pixel_geometry

                xx, yy = (np.array(i) for i in polygon.exterior.xy)
                xx = xx + self.patch_df.loc[image_id, "pixel_bounds"][0]  # add min_x
                yy = yy + self.patch_df.loc[image_id, "pixel_bounds"][1]  # add min_y

                parent_polygon = Polygon(zip(xx, yy)).buffer(0)

                # convert line
                if instance.pixel_line is not None:
                    line = instance.pixel_line
                    xx, yy = (np.array(i) for i in line.xy)
                    xx = (
                        xx + self.patch_df.loc[image_id, "pixel_bounds"][0]
                    )  # add min_x
                    yy = (
                        yy + self.patch_df.loc[image_id, "pixel_bounds"][1]
                    )  # add min_y

                    parent_line = LineString(zip(xx, yy))
                else:
                    parent_line = None

                self.parent_predictions[parent_id].append(
                    ParentPrediction(
                        pixel_geometry=parent_polygon,
                        pixel_line=parent_line,
                        score=instance.score,
                        text=instance.text,
                        patch_id=image_id,
                    )
                )

        if deduplicate:
            for parent_id in self.parent_predictions.keys():
                self._deduplicate_parent_level(parent_id, min_ioa=min_ioa)

        if return_dataframe:
            return self._dict_to_dataframe(self.parent_predictions)
        return self.parent_predictions

    def _deduplicate_parent_level(self, image_id, min_ioa=0.7):
        # get parent predictions for selected parent image
        all_patches = [pred.patch_id for pred in self.parent_predictions[image_id]]
        patches = np.unique(all_patches).tolist()

        for patch_i, patch_j in combinations(patches, 2):
            # get patch bounds
            patch_bounds_i = Polygon.from_bounds(
                *self.patch_df.loc[patch_i, "pixel_bounds"]
            )
            patch_bounds_j = Polygon.from_bounds(
                *self.patch_df.loc[patch_j, "pixel_bounds"]
            )

            if patch_bounds_i.intersects(patch_bounds_j):
                # get patch intersection
                intersection = patch_bounds_i.intersection(patch_bounds_j)

                # get polygons that overlap with the patch intersection
                polygons = []
                for i, pred in enumerate(np.array(self.parent_predictions[image_id])):
                    if pred is None:
                        continue
                    elif pred.patch_id in [
                        patch_i,
                        patch_j,
                    ] and pred.pixel_geometry.intersects(intersection):
                        polygons.append([i, pred.pixel_geometry])

                def calc_ioa(polygons, i, j):
                    return (
                        polygons[i][1].intersection(polygons[j][1]).area
                        / polygons[i][1].area
                    )

                # create an intersection matrix
                i_matrix = np.zeros((len(polygons), len(polygons)))

                # calculate intersection over area for these polygons
                for i, j in combinations(range(len(polygons)), 2):
                    if polygons[i][1].intersects(polygons[j][1]):
                        ioa_i = calc_ioa(polygons, i, j)
                        ioa_j = calc_ioa(polygons, j, i)
                        # bigger ioa means more overlap, so keep the one with the bigger ioa (only if more than min_ioa)
                        if ioa_i > ioa_j and ioa_i > min_ioa:
                            i_matrix[i, j] = ioa_i
                        elif ioa_j > ioa_i and ioa_j > min_ioa:
                            i_matrix[j, i] = ioa_j

                # if there is an intersection with another polygon then remove this polygon
                for i, row in enumerate(i_matrix):
                    # index of the polygon in the parent_preds array
                    index = polygons[i][0]
                    if any([ioa != 0 for ioa in row]):
                        self.parent_predictions[image_id][index] = None
                        i_matrix[:, i] = 0

        # remove the None values
        self.parent_predictions[image_id] = [
            prediction
            for prediction in self.parent_predictions[image_id]
            if prediction is not None
        ]

    def convert_to_coords(
        self,
        return_dataframe: bool = False,
    ) -> dict | gpd.GeoDataFrame:
        """Convert the parent predictions to georeferenced predictions by converting pixel bounds to coordinates.

        Parameters
        ----------
        return_dataframe : bool, optional
            Whether to return the predictions as a geopandas GeoDataFrame, by default False

        Returns
        -------
        dict or gpd.GeoDataFrame
            A dictionary of predictions for each parent image or a DataFrame if `return_dataframe` is True.
        """
        self.check_georeferencing()
        if not self.georeferenced:
            raise ValueError(
                "[ERROR] Cannot convert to coordinates as parent_df does not have 'coordinates' column."
            )

        if self.parent_predictions == {}:
            print("[INFO] Converting patch pixel bounds to parent pixel bounds.")
            _ = self.convert_to_parent_pixel_bounds()

        self.geo_predictions = {}  # reset geo predictions

        for parent_id, prediction in self.parent_predictions.items():
            if parent_id not in self.geo_predictions.keys():
                self.geo_predictions[parent_id] = []

                for instance in prediction:
                    # convert polygon
                    polygon = instance.pixel_geometry

                    xx, yy = (np.array(i) for i in polygon.exterior.xy)
                    xx = (
                        xx * self.parent_df.loc[parent_id, "dlon"]
                        + self.parent_df.loc[parent_id, "coordinates"][0]
                    )
                    yy = (
                        self.parent_df.loc[parent_id, "coordinates"][3]
                        - yy * self.parent_df.loc[parent_id, "dlat"]
                    )

                    parent_polygon_geo = Polygon(zip(xx, yy)).buffer(0)
                    crs = self.parent_df.loc[parent_id, "crs"]

                    # convert line
                    if instance.pixel_line is not None:
                        line = instance.pixel_line

                        xx, yy = (np.array(i) for i in line.xy)
                        xx = (
                            xx * self.parent_df.loc[parent_id, "dlon"]
                            + self.parent_df.loc[parent_id, "coordinates"][0]
                        )
                        yy = (
                            self.parent_df.loc[parent_id, "coordinates"][3]
                            - yy * self.parent_df.loc[parent_id, "dlat"]
                        )
                        parent_line_geo = LineString(zip(xx, yy))
                    else:
                        parent_line_geo = None

                    self.geo_predictions[parent_id].append(
                        GeoPrediction(
                            pixel_geometry=instance.pixel_geometry,
                            pixel_line=instance.pixel_line,
                            score=instance.score,
                            text=instance.text,
                            patch_id=instance.patch_id,
                            geometry=parent_polygon_geo,
                            line=parent_line_geo,
                            crs=crs,
                        )
                    )

        if return_dataframe:
            return self._dict_to_dataframe(self.geo_predictions)
        return self.geo_predictions

    def save_to_geojson(
        self,
        path_save: str | pathlib.Path,
        centroid: bool = False,
    ) -> None:
        """
        Save the georeferenced predictions to a GeoJSON file.

        Parameters
        ----------
        path_save : str | pathlib.Path, optional
            Path to save the GeoJSON file
        centroid : bool, optional
            Whether to convert the polygons to centroids, by default False.
            NOTE: The original polygon will still be saved as a separate column
        """
        print(
            "[WARNING] This method is deprecated and will soon be removed. Use `to_geojson` instead."
        )
        self.to_geojson(path_save, centroid)

    def to_geojson(
        self,
        path_save: str | pathlib.Path,
        centroid: bool = False,
    ) -> None:
        """Save the georeferenced predictions to a GeoJSON file.

        Parameters
        ----------
        path_save : str | pathlib.Path, optional
            Path to save the GeoJSON file
        centroid : bool, optional
            Whether to convert the polygons to centroids, by default False.
            NOTE: The original polygon will still be saved as a separate column
        """
        if self.geo_predictions == {}:
            raise ValueError(
                "[ERROR] No georeferenced predictions found. Run `convert_to_coords` first."
            )

        geo_df = self._dict_to_dataframe(self.geo_predictions)

        if centroid:
            geo_df["polygon"] = geo_df["geometry"].to_wkt()
            geo_df["geometry"] = (
                geo_df["geometry"].to_crs("27700").centroid.to_crs(geo_df.crs)
            )

        geo_df.to_file(path_save, driver="GeoJSON", engine="pyogrio")

    def to_csv(
        self,
        path_save: str | pathlib.Path,
        centroid: bool = False,
    ) -> None:
        """Saves the patch, parent and georeferenced predictions to CSV files.

        Parameters
        ----------
        path_save : str | pathlib.Path
            The path to save the CSV files. Files will be saved as `patch_predictions.csv`, `parent_predictions.csv` and `geo_predictions.csv`.
        centroid : bool, optional
            Whether to convert polygons to centroids, by default False.
            NOTE: The original polygon will still be saved as a separate column.

        Note
        ----
        Use the `save_to_geojson` method to save georeferenced predictions to a GeoJSON file.
        """
        if self.patch_predictions == {}:  # implies no parent or geo predictions
            raise ValueError("[ERROR] No patch predictions found.")

        if not os.path.exists(path_save):
            os.makedirs(path_save)

        print("[INFO] Saving patch predictions.")
        patch_df = self._dict_to_dataframe(self.patch_predictions)
        if centroid:
            patch_df["polygon"] = patch_df["pixel_geometry"]
            patch_df["pixel_geometry"] = patch_df["pixel_geometry"].apply(
                lambda x: x.centroid
            )
        patch_df.to_csv(f"{path_save}/patch_predictions.csv")

        if self.parent_predictions != {}:
            print("[INFO] Saving parent predictions.")
            parent_df = self._dict_to_dataframe(self.parent_predictions)
            if centroid:
                parent_df["polygon"] = parent_df["pixel_geometry"]
                parent_df["pixel_geometry"] = parent_df["pixel_geometry"].apply(
                    lambda x: x.centroid
                )
            parent_df.to_csv(f"{path_save}/parent_predictions.csv")

        if self.geo_predictions != {}:
            print("[INFO] Saving geo predictions.")
            geo_df = self._dict_to_dataframe(self.geo_predictions)
            if centroid:
                geo_df["polygon"] = geo_df["geometry"]
                geo_df["geometry"] = (
                    geo_df["geometry"].to_crs("27700").centroid.to_crs(geo_df.crs)
                )
            geo_df.to_csv(f"{path_save}/geo_predictions.csv")

    def show_predictions(
        self,
        image_id: str,
        figsize: tuple = (10, 10),
        border_color: str | None = "r",
        text_color: str | None = "b",
    ) -> None:
        """Show the predictions on an image.

        Parameters
        ----------
        image_id : str
            The image ID to show the predictions on.
        figsize : tuple | None, optional
            The size of the figure, by default (10, 10)
        border_color : str | None, optional
            The color of the border of the polygons, by default "r"
        text_color : str | None, optional
            The color of the text, by default "b"

        Raises
        ------
        ValueError
            If the image ID is not found in the patch or parent predictions.
        """

        if image_id in self.patch_predictions.keys():
            preds = self.patch_predictions
            image_path = self.patch_df.loc[image_id, "image_path"]

        elif image_id in self.parent_predictions.keys():
            preds = self.parent_predictions
            image_path = self.parent_df.loc[image_id, "image_path"]

        else:
            raise ValueError(
                f"[ERROR] {image_id} not found in patch or parent predictions. Check your ID is valid or try running `convert_to_parent_pixel_bounds` first."
            )

        img = Image.open(image_path)

        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")

        # check if grayscale
        if len(img.getbands()) == 1:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, zorder=1)
        else:
            ax.imshow(img, zorder=1)
        ax.set_title(image_id)

        for instance in preds[image_id]:
            polygon = np.array(instance.pixel_geometry.exterior.coords.xy)
            center = instance.pixel_geometry.centroid.coords.xy
            patch = patches.Polygon(polygon.T, edgecolor=border_color, facecolor="none")
            ax.add_patch(patch)
            ax.text(
                x=center[0][0],
                y=center[1][0],
                s=instance.text if instance.text is not None else instance.score,
                fontsize=8,
                color=text_color,
            )
        fig.show()

    def explore_predictions(
        self,
        parent_id: str,
        xyz_url: str | None = None,
        style_kwargs: dict | None = None,
    ):
        self.check_georeferencing()
        if not self.georeferenced:
            raise ValueError(
                "[ERROR] This method only works for georeferenced results. Please ensure parent_df has 'coordinates' column and run `convert_to_coords` first."
            )

        if parent_id not in self.geo_predictions.keys():
            raise ValueError(
                f"[ERROR] {parent_id} not found in geo predictions. Check your ID is valid or try running `convert_to_coords` first."
            )

        if style_kwargs is None:
            style_kwargs = {"fillOpacity": 0.2}

        if xyz_url:
            tiles = xyz.TileProvider(name=xyz_url, url=xyz_url, attribution=xyz_url)
        else:
            tiles = xyz.providers.OpenStreetMap.Mapnik

        preds_df = self._dict_to_dataframe(self.geo_predictions)
        preds_df.drop(
            columns=["pixel_geometry", "pixel_line", "line"], inplace=True
        )  # drop other geometry since we can't have more than one geometry columns

        return preds_df[preds_df["image_id"] == parent_id].explore(
            tiles=tiles,
            style_kwds=style_kwargs,
        )

    def load_geo_predictions(
        self,
        load_path: str | pathlib.Path,
    ):
        """Load georeferenced text predictions from a GeoJSON file.

        Parameters
        ----------
        load_path : str | pathlib.Path
            The path to the GeoJSON file.

        Raises
        ------
        ValueError
            If the path does not point to a GeoJSON file.

        Note
        ----
        This will overwrite any existing predictions!
        """
        if re.search(r"\..*?json$", str(load_path)):
            preds_df = load_from_geojson(load_path, engine="pyogrio")
        else:
            raise ValueError("[ERROR] ``load_path`` must be a path to a geojson file.")

        # convert pixel_geometry to shapely geometry
        preds_df["pixel_geometry"] = preds_df["pixel_geometry"].apply(
            lambda x: from_wkt(x)
        )

        # convert lines to shapely geometry
        # if line and pixel_line are not in the columns, add them as None values
        if "line" in preds_df.columns:
            preds_df["line"] = preds_df["line"].apply(lambda x: from_wkt(x))
        else:
            preds_df["line"] = None
        if "pixel_line" in preds_df.columns:
            preds_df["pixel_line"] = preds_df["pixel_line"].apply(lambda x: from_wkt(x))
        else:
            preds_df["pixel_line"] = None

        self.geo_predictions = {}
        self.parent_predictions = {}

        for image_id in preds_df.index.unique():
            if image_id not in self.geo_predictions.keys():
                self.geo_predictions[image_id] = []
            if image_id not in self.parent_predictions.keys():
                self.parent_predictions[image_id] = []

            for _, v in preds_df[preds_df.index == image_id].iterrows():
                self.geo_predictions[image_id].append(
                    GeoPrediction(
                        pixel_geometry=v.pixel_geometry,
                        pixel_line=v.pixel_line,
                        score=v.score,
                        text=v.text if "text" in v.index else None,
                        patch_id=v.patch_id,
                        geometry=v.geometry,
                        line=v.line,
                        crs=v.crs,
                    )
                )
                self.parent_predictions[image_id].append(
                    ParentPrediction(
                        pixel_geometry=v.pixel_geometry,
                        pixel_line=v.pixel_line,
                        score=v.score,
                        text=v.text if "text" in v.index else None,
                        patch_id=v.patch_id,
                    )
                )

        self.patch_predictions = {}  # reset patch predictions

        for _, prediction in self.parent_predictions.items():
            for instance in prediction:
                if instance.patch_id not in self.patch_predictions.keys():
                    self.patch_predictions[instance.patch_id] = []

                # convert polygon
                polygon = instance.pixel_geometry

                xx, yy = (np.array(i) for i in polygon.exterior.xy)
                xx = (
                    xx - self.patch_df.loc[instance.patch_id, "pixel_bounds"][0]
                )  # add min_x
                yy = (
                    yy - self.patch_df.loc[instance.patch_id, "pixel_bounds"][1]
                )  # add min_y
                patch_polygon = Polygon(zip(xx, yy)).buffer(0)

                # convert line
                if instance.pixel_line is not None:
                    line = instance.pixel_line

                    xx, yy = (np.array(i) for i in line.xy)
                    xx = (
                        xx - self.patch_df.loc[instance.patch_id, "pixel_bounds"][0]
                    )  # add min_x
                    yy = (
                        yy - self.patch_df.loc[instance.patch_id, "pixel_bounds"][1]
                    )  # add min_y
                    patch_line = LineString(zip(xx, yy))
                else:
                    patch_line = None

                self.patch_predictions[instance.patch_id].append(
                    PatchPrediction(
                        pixel_geometry=patch_polygon,
                        pixel_line=patch_line,
                        score=instance.score,
                        text=instance.text,
                    )
                )

    def load_patch_predictions(
        self,
        patch_preds: str | pathlib.Path | pd.DataFrame,
    ) -> None:
        if not isinstance(patch_preds, pd.DataFrame):
            if re.search(r"\..*?csv$", str(patch_preds)):
                patch_preds = pd.read_csv(patch_preds, index_col=0)
                patch_preds = eval_dataframe(patch_preds)
            else:
                raise ValueError(
                    "[ERROR] ``patch_preds`` must be a pandas DataFrame or path to a CSV file."
                )

            # if we have a polygon column, this implies the pixel_geometry column is the centroid
            if "polygon" in patch_preds.columns:
                patch_preds["pixel_geometry"] = patch_preds["polygon"]
                patch_preds.drop(columns=["polygon"], inplace=True)

            # convert pixel_geometry to shapely geometry
            patch_preds["pixel_geometry"] = patch_preds["pixel_geometry"].apply(
                lambda x: from_wkt(x)
            )

            # convert lines to shapely geometry
            # if line and pixel_line are not in the columns, add them as None values
            if "pixel_line" in patch_preds.columns:
                patch_preds["pixel_line"] = patch_preds["pixel_line"].apply(
                    lambda x: from_wkt(x)
                )
            else:
                patch_preds["pixel_line"] = None

        self.patch_predictions = {}  # reset patch predictions

        for image_id in patch_preds["image_id"].unique():
            if image_id not in self.patch_predictions.keys():
                self.patch_predictions[image_id] = []

            for _, v in patch_preds[patch_preds["image_id"] == image_id].iterrows():
                if not hasattr(v, "pixel_line"):
                    v.pixel_line = None
                self.patch_predictions[image_id].append(
                    PatchPrediction(
                        pixel_geometry=v.pixel_geometry,
                        pixel_line=v.pixel_line,
                        score=v.score,
                        text=v.text if "text" in v.index else None,
                    )
                )

        self.geo_predictions = {}
        self.parent_predictions = {}

        self.convert_to_parent_pixel_bounds()


class DetRecRunner(DetRunner):
    def _get_patch_predictions(
        self,
        outputs: dict,
        return_dataframe: bool = False,
        min_ioa: float = 0.7,
    ) -> dict | pd.DataFrame:
        """Post process the model outputs to get patch predictions.

        Parameters
        ----------
        outputs : dict
            The outputs from the model.
        return_dataframe : bool, optional
            Whether to return the predictions as a pandas DataFrame, by default False
        min_ioa : float, optional
            The minimum intersection over area to consider two polygons the same, by default 0.7

        Returns
        -------
        dict or pd.DataFrame
            A dictionary containing the patch predictions or a DataFrame if `as_dataframe` is True.
        """
        # key for predictions
        image_id = outputs["image_id"]
        self.patch_predictions[image_id] = []

        # get instances
        instances = outputs["instances"].to("cpu")
        ctrl_pnts = instances.ctrl_points.numpy()
        scores = instances.scores.tolist()
        recs = instances.recs
        bd_pts = np.asarray(instances.bd)

        self._post_process(image_id, ctrl_pnts, scores, recs, bd_pts)
        self._deduplicate(image_id, min_ioa=min_ioa)

        if return_dataframe:
            return self._dict_to_dataframe(self.patch_predictions)
        return self.patch_predictions

    def _process_ctrl_pnt(self, pnt):
        points = pnt.reshape(-1, 2)
        return points

    def _post_process(self, image_id, ctrl_pnts, scores, recs, bd_pnts):
        for ctrl_pnt, score, rec, bd in zip(ctrl_pnts, scores, recs, bd_pnts):
            # draw polygons
            if bd is not None:
                bd = np.hsplit(bd, 2)
                bd = np.vstack([bd[0], bd[1][::-1]])
                polygon = Polygon(bd).buffer(0)

                if isinstance(polygon, MultiPolygon):
                    polygon = polygon.convex_hull

            # draw center lines
            line = self._process_ctrl_pnt(ctrl_pnt)
            line = LineString(line)

            # draw text
            text = self._ctc_decode_recognition(rec)
            if self.voc_size == 37:
                text = text.upper()
            # text = "{:.2f}: {}".format(score, text)
            text = f"{text}"
            score = f"{score:.2f}"

            self.patch_predictions[image_id].append(
                PatchPrediction(
                    pixel_geometry=polygon, pixel_line=line, score=score, text=text
                )
            )

    def search_predictions(
        self, search_text: str, ignore_case: bool = True, return_dataframe: bool = False
    ) -> dict | pd.DataFrame:
        """Search the predictions for specific text. Accepts regex.

        Parameters
        ----------
        search_text : str
            The text to search for. Can be a regex pattern.
        ignore_case : bool, optional
            Whether to ignore case when searching, by default True.
        return_dataframe : bool, optional
            Whether to return the results as a pandas DataFrame, by default False.

        Returns
        -------
        dict | pd.DataFrame
            A dictionary containing the search results or a DataFrame if `return_dataframe` is True.

        Raises
        ------
        ValueError
            If no parent predictions are found.
        """
        # reset the search results
        self.search_results = {}

        # whether to ignore case
        kwargs = {"flags": re.IGNORECASE} if ignore_case else {}

        if self.parent_predictions == {}:
            raise ValueError(
                "[ERROR] No parent predictions found. You may need to run `convert_to_parent_pixel_bounds()`."
            )

        for image_id, preds in self.parent_predictions.items():
            for instance in preds:
                if re.search(search_text, instance.text, **kwargs):
                    if image_id in self.search_results:
                        self.search_results[image_id].append(instance)
                    else:
                        self.search_results[image_id] = [instance]

        if return_dataframe:
            return self._dict_to_dataframe(self.search_results)
        return self.search_results

    def show_search_results(
        self,
        parent_id: str,
        figsize: tuple | None = (10, 10),
        border_color: str | None = "r",
        text_color: str | None = "b",
    ) -> None:
        """Show the search results on an image.

        Parameters
        ----------
        parent_id : str
            The image ID to show the predictions on (must be parent level).
        figsize : tuple | None, optional
            The size of the figure, by default (10, 10)
        border_color : str | None, optional
            The color of the border of the polygons, by default "r"
        text_color : str | None, optional
            The color of the text, by default "b".

        Raises
        ------
        ValueError
            If the image ID is not found in the patch or parent predictions.
        """
        if parent_id in self.parent_predictions.keys():
            image_path = self.parent_df.loc[parent_id, "image_path"]
        else:
            raise ValueError(f"[ERROR] {parent_id} not found in parent predictions.")

        img = Image.open(image_path)

        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")

        # check if grayscale
        if len(img.getbands()) == 1:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, zorder=1)
        else:
            ax.imshow(img, zorder=1)
        ax.set_title(parent_id)

        preds = self.search_results

        for instance in preds[parent_id]:
            polygon = np.array(instance.pixel_geometry.exterior.coords.xy)
            center = instance.pixel_geometry.centroid.coords.xy
            patch = patches.Polygon(polygon.T, edgecolor=border_color, facecolor="none")
            ax.add_patch(patch)
            ax.text(
                x=center[0][0],
                y=center[1][0],
                s=instance.text,
                fontsize=8,
                color=text_color,
            )

        fig.show()

    def _get_geo_search_results(self):
        """Convert search results to georeferenced search results.

        Returns
        -------
        dict
            Dictionary containing georeferenced search results.
        """
        self.check_georeferencing()
        if not self.georeferenced:
            raise ValueError(
                "[ERROR] Cannot convert to coordinates as parent_df does not have 'coordinates' column."
            )

        geo_search_results = {}

        for parent_id, prediction in self.search_results.items():
            if parent_id not in geo_search_results.keys():
                geo_search_results[parent_id] = []

                for instance in prediction:
                    # convert polygon
                    polygon = instance.pixel_geometry

                    xx, yy = (np.array(i) for i in polygon.exterior.xy)
                    xx = (
                        xx * self.parent_df.loc[parent_id, "dlon"]
                        + self.parent_df.loc[parent_id, "coordinates"][0]
                    )
                    yy = (
                        self.parent_df.loc[parent_id, "coordinates"][3]
                        - yy * self.parent_df.loc[parent_id, "dlat"]
                    )

                    parent_polygon_geo = Polygon(zip(xx, yy)).buffer(0)
                    crs = self.parent_df.loc[parent_id, "crs"]

                    # convert line
                    line = instance.pixel_line

                    xx, yy = (np.array(i) for i in line.xy)
                    xx = (
                        xx * self.parent_df.loc[parent_id, "dlon"]
                        + self.parent_df.loc[parent_id, "coordinates"][0]
                    )
                    yy = (
                        self.parent_df.loc[parent_id, "coordinates"][3]
                        - yy * self.parent_df.loc[parent_id, "dlat"]
                    )
                    parent_line_geo = LineString(zip(xx, yy))

                    geo_search_results[parent_id].append(
                        GeoPrediction(
                            pixel_geometry=instance.pixel_geometry,
                            pixel_line=instance.pixel_line,
                            score=instance.score,
                            text=instance.score,
                            patch_id=instance.patch_id,
                            geometry=parent_polygon_geo,
                            line=parent_line_geo,
                            crs=crs,
                        )
                    )

        return geo_search_results

    def explore_search_results(
        self,
        parent_id: str,
        xyz_url: str | None = None,
        style_kwargs: dict | None = None,
    ):
        self.check_georeferencing()
        if not self.georeferenced:
            raise ValueError(
                "[ERROR] This method only works for georeferenced results. Please ensure parent_df has 'coordinates' column and run `convert_to_coords` first."
            )

        if parent_id not in self.geo_predictions.keys():
            raise ValueError(f"[ERROR] {parent_id} not found in geo predictions.")

        if style_kwargs is None:
            style_kwargs = {"fillOpacity": 0.2}

        if xyz_url:
            tiles = xyz.TileProvider(name=xyz_url, url=xyz_url, attribution=xyz_url)
        else:
            tiles = xyz.providers.OpenStreetMap.Mapnik

        geo_search_results = self._get_geo_search_results()
        geo_df = self._dict_to_dataframe(geo_search_results)
        geo_df.drop(
            columns=["pixel_geometry", "pixel_line", "line"], inplace=True
        )  # drop other geometry since we can't have more than one geometry columns

        return geo_df[geo_df["image_id"] == parent_id].explore(
            tiles=tiles,
            style_kwds=style_kwargs,
        )

    def save_search_results_to_geojson(
        self,
        path_save: str | pathlib.Path,
        centroid: bool = False,
    ) -> None:
        """Convert the search results to georeferenced search results and save them to a GeoJSON file.

        Parameters
        ----------
        path_save : str | pathlib.Path
            The path to save the GeoJSON file.
        centroid : bool, optional
            Whether to save the centroid of the polygons as the geometry column, by default False.
            Note: The original polygon will stil be saved as a separate column.

        Raises
        ------
        ValueError
            If no search results are found.
        """
        print(
            "[WARNING] This method is deprecated and will soon be removed. Use `search_results_to_geojson` instead."
        )
        self.search_results_to_geojson(path_save, centroid)

    def search_results_to_geojson(
        self,
        path_save: str | pathlib.Path,
        centroid: bool = False,
    ) -> None:
        """Convert the search results to georeferenced search results and save them to a GeoJSON file.

        Parameters
        ----------
        path_save : str | pathlib.Path
            The path to save the GeoJSON file.
        centroid : bool, optional
            Whether to save the centroid of the polygons as the geometry column, by default False.
            Note: The original polygon will stil be saved as a separate column.

        Raises
        ------
        ValueError
            If no search results are found.
        """
        if self.search_results == {}:
            raise ValueError("[ERROR] No results to save!")

        geo_search_results = self._get_geo_search_results()
        geo_df = self._dict_to_dataframe(geo_search_results)

        if centroid:
            geo_df["polygon"] = geo_df["geometry"].to_wkt()
            geo_df["geometry"] = (
                geo_df["geometry"].to_crs("27700").centroid.to_crs(geo_df.crs)
            )

        geo_df.to_file(path_save, driver="GeoJSON", engine="pyogrio")

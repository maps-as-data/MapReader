#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TODO

import copy
from glob import glob
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pandas as pd
import shutil

from typing import Union, Optional, List, Tuple, Literal, Dict

from .tileserver_helpers import create_hf
from .tileserver_helpers import collect_coord_info
from .tileserver_helpers import check_par_jobs
from .tileserver_scraper import scraper
from .tileserver_stitcher import stitcher


class TileServer:
    """
    A class representing a tile server for a map.

    Parameters
    ----------
    metadata_path : str, dict, or list
        The path to the metadata file for the map. This can be a string
        representing the file path, a dictionary containing the metadata,
        or a list of metadata features. Usually, it is a string representing
        the file path to a metadata file downloaded from a tileserver.

        Some example metadata files can be found in
        `MapReader/worked_examples/persistent_data <https://github.com/Living-with-machines/MapReader/tree/main/worked_examples/persistent_data>`_.
    geometry : str, optional
        The type of geometry that defines the boundaries in the map. Defaults
        to ``"polygon"``.
    download_url : str, optional
        The base URL pattern used to download tiles from the server. This
        should contain placeholders for the x coordinate (``x``), the y
        coordinate (``y``) and the zoom level (``z``).

        Defaults to a URL for a
        specific tileset:
        ``https://mapseries-tilesets.s3.amazonaws.com/1inch_2nd_ed/{z}/{x}/{y}.png``

    Attributes
    ----------
    detected_rect_boundary : bool
        Whether or not the rectangular boundary of the map has been detected.
    found_queries : None
        Placeholder for a list of found queries.
    geometry : str
        The type of geometry used for the map.
    download_url : str
        The URL pattern used to download tiles from the server.
    metadata : list
        A list of metadata features for the map.  Each key in this dict should
        contain:

        - ``["geometry"]["coordinates"]``
        - ``["properties"]["IMAGEURL"]``
        - ``["properties"]["IMAGE"]``

    Raises
    ------
    ValueError
        If the metadata file could not be found or loaded.
    """

    def __init__(
        self,
        metadata_path: Union[str, dict, list],
        geometry: Optional[str] = "polygon",
        download_url: Optional[
            str
        ] = "https://mapseries-tilesets.s3.amazonaws.com/1inch_2nd_ed/{z}/{x}/{y}.png",  # noqa
    ):
        """
        Initializer for the class representing a tile server for a map.
        """
        # Initialize variables
        self.detected_rect_boundary = False
        self.found_queries = None
        self.geometry = geometry
        self.download_url = download_url

        if isinstance(metadata_path, str) and os.path.isfile(metadata_path):
            # Read a metadata file
            json_fio = open(metadata_path, "r")
            json_f = json_fio.readlines()[0]
            # change metadata to a dictionary:
            metadata_all = eval(json_f)
            metadata = metadata_all["features"]

        elif isinstance(metadata_path, dict):
            metadata = metadata_all["features"]

        elif isinstance(metadata_path, list):
            metadata = metadata_path[:]

        else:
            raise ValueError(f"Could not find or load: {metadata_path}")

        # metadata contains the extracted "features" of the ``metadata_path``
        self.metadata = metadata

        print(self.__str__())

    def __str__(self) -> str:
        info = f"Metadata file has {self.__len__()} items.\n"
        info += f"Download URL: {self.download_url}\n"
        info += f"Geometry: {self.geometry}"
        return info

    def __len__(self) -> int:
        return len(self.metadata)

    def create_info(self) -> None:
        """
        Collects metadata information and boundary coordinates for fast
        queries.

        Populates the ``metadata_info_list`` and ``metadata_coord_arr``
        attributes of the ``TileServer`` instance with information about the
        map's metadata and boundary coordinates, respectively. Sets the
        ``detected_rect_boundary`` attribute to ``True``.

        Returns
        -------
        None

        Notes
        -----
        This is a helper function for other methods in this class
        """
        metadata_info_list = []
        metadata_coord_arr = []
        for one_item in self.metadata:
            (
                min_lon,
                max_lon,
                min_lat,
                max_lat,
            ) = self.detect_rectangle_boundary(
                one_item["geometry"]["coordinates"][0][0]
            )
            # XXX hardcoded: the file path is hardcoded to map_<IMAGE>.png
            metadata_info_list.append(
                [
                    one_item["properties"]["IMAGEURL"],
                    "map_" + one_item["properties"]["IMAGE"] + ".png",
                ]
            )
            # Collect boundaries for fast queries
            metadata_coord_arr.append([min_lon, max_lon, min_lat, max_lat])
        self.metadata_info_list = metadata_info_list
        self.metadata_coord_arr = np.array(metadata_coord_arr).astype(float)
        self.detected_rect_boundary = True

    def modify_metadata(
        self,
        remove_image_ids: Optional[List[str]] = [],
        only_keep_image_ids: Optional[List[str]] = [],
    ) -> None:
        """
        Modifies the metadata by removing or keeping specified images.

        Parameters
        ----------
        remove_image_ids : list of str, optional
            List of image IDs to remove from the metadata (default is an empty
            list, ``[]``).
        only_keep_image_ids : list of str, optional
            List of image IDs to keep in the metadata (default is an empty
            list, ``[]``).

        Returns
        -------
        None

        Notes
        -----
        Removes image metadata whose IDs are in the ``remove_image_ids`` list,
        and keeps only image metadata whose IDs are in ``only_keep_image_ids``.
        Populates the ``metadata`` attribute of the ``TileServer`` instance
        with the modified metadata. If any metadata is removed, the
        :meth:`mapreader.download.tileserver_access.create_info` method is
        called to update the boundary coordinates for fast queries.
        """

        print(f"[INFO] #images (before modification): {len(self.metadata)}")

        one_edit = False
        if len(remove_image_ids) != 0:
            for i in range(len(self.metadata) - 1, -1, -1):
                id = self.metadata[i]["properties"]["IMAGE"]
                if id in remove_image_ids:
                    print(f"Removing {id}")
                    del self.metadata[i]
                    one_edit = True

        if len(only_keep_image_ids) != 0:
            for i in range(len(self.metadata) - 1, -1, -1):
                id = self.metadata[i]["properties"]["IMAGE"]
                if id not in only_keep_image_ids:
                    print(f"Removing {id}")
                    del self.metadata[i]
                    one_edit = True

        if one_edit:
            print("[INFO] run .create_info")
            self.create_info()

        print(f"[INFO] #images (after modification): {len(self.metadata)}")

    def query_point(
        self,
        latlon_list: Union[List[Tuple[float, float]], Tuple[float, float]],
        append: Optional[bool] = False,
    ) -> None:
        """
        Queries the point(s) specified by ``latlon_list`` and returns
        information about the map tile(s) that contain the point(s).

        Parameters
        ----------
        latlon_list : list of tuples or tuple
            The list of latitude-longitude pairs to query. Each tuple must
            have the form ``(latitude, longitude)``. If only one pair is
            provided, it can be passed as a tuple instead of a list of tuples.
        append : bool, optional
            Whether to append the query results to any previously found
            queries, or to overwrite them. Defaults to ``False``.

        Returns
        -------
        None
            The query results are stored in the attribute `found_queries` of
            the TileServer instance.

        Notes
        -----
        Before performing the query, the function checks if the boundaries of
        the map tiles have been detected. If not, it runs the method
        :meth:`mapreader.download.tileserver_access.create_info` to detect the
        boundaries.

        The query results are stored in the attribute `found_queries` of the
        TileServer instance as a list of lists, where each sublist corresponds
        to a map tile and has the form: ``[image_url, image_filename,
        [min_lon, max_lon, min_lat, max_lat], index_in_metadata]``.
        """
        # Detect the boundaries of maps
        if not self.detected_rect_boundary:
            self.create_info()

        if not isinstance(latlon_list[0], list):
            latlon_list = [latlon_list]

        if append and (self.found_queries is not None):
            found_queries = copy.deepcopy(self.found_queries)
        else:
            found_queries = []

        for one_q in latlon_list:
            lat_q, lon_q = one_q
            indx_q = np.where(
                (self.metadata_coord_arr[:, 2] <= lat_q)
                & (lat_q <= self.metadata_coord_arr[:, 3])
                & (self.metadata_coord_arr[:, 0] <= lon_q)
                & (lon_q <= self.metadata_coord_arr[:, 1])
            )[0]
            if len(indx_q) == 0:
                print(f"Could not find any candidates for {one_q}")
                continue

            indx_q = indx_q[0]

            # Check if the query is already in the list
            already_in_candidates = False
            if len(found_queries) > 0:
                for query in found_queries:
                    if self.metadata_info_list[indx_q][0] == query[0]:
                        already_in_candidates = True
                        print(f"Already in the query list: {query[0]}")
                        break

            if not already_in_candidates:
                found_queries.append(
                    copy.deepcopy(self.metadata_info_list[indx_q])
                )
                found_queries[-1].extend(
                    [copy.deepcopy(self.metadata_coord_arr[indx_q]), indx_q]
                )

        self.found_queries = found_queries

    def print_found_queries(self) -> None:
        """
        Print the found queries in a formatted way.

        Returns
        -------
        None

        Examples
        --------
        .. code-block:: python

            >>> obj = TileServer()
            >>> obj.query_point([(40.0, -105.0)])
            >>> obj.print_found_queries()
            ------------
            Found items:
            ------------
            URL:      https://example.com/image1.png
            filepath: map_image1.png
            coords:   [min_lon, max_lon, min_lat, max_lat]
            index:    0
            ====================
        """
        divider = 12 * "-"
        print(f"{divider}\nFound items:\n{divider}")
        if not self.found_queries:
            print("[]")
        else:
            for item in self.found_queries:
                print(f"URL:     \t{item[0]}")
                print(f"filepath:\t{item[1]}")
                print(f"coords:  \t{item[2]}")
                print(f"index:   \t{item[3]}")
                print(20 * "=")

    def detect_rectangle_boundary(
        self,
        coords: List[
            Tuple[
                float,
                float,
                float,
                float,
            ]
        ],
    ) -> Tuple[float, float, float, float]:
        """
        Detects the rectangular boundary of a polygon defined by a list of
        coordinates.

        Parameters
        ----------
        coords : list of tuples
            The list of coordinates defining the polygon.

        Returns
        -------
        float, float, float, float
            The minimum longitude, maximum longitude, minimum latitude, and
            maximum latitude of the rectangular boundary of the polygon.

        """
        if self.geometry == "polygon":
            coord_arr = np.array(coords)
            # if len(coord_arr) != 5:
            #    raise ValueError(f"[ERROR] expected length of coordinate list is 5. coords: {coords}") # noqa
            min_lon = np.min(coord_arr[:, 0])
            max_lon = np.max(coord_arr[:, 0])
            min_lat = np.min(coord_arr[:, 1])
            max_lat = np.max(coord_arr[:, 1])

            """
            # this method results in smaller rectangles (compared to the
            # original polygon) particularly if the map is strongly tilted
            min_lon = np.sort(coord_arr[:-1, 0])[1]
            max_lon = np.sort(coord_arr[:-1, 0])[2]
            min_lat = np.sort(coord_arr[:-1, 1])[1]
            max_lat = np.sort(coord_arr[:-1, 1])[2]
            """
        return min_lon, max_lon, min_lat, max_lat

    def create_metadata_query(self) -> None:
        """
        Create a list of metadata query based on the found queries.

        Returns
        -------
        None
            Nothing is returned but the TileServer instance's
            ``metadata_query`` property is set to a list of metadata query
            based on the found queries.

        Notes
        -----
        This is used in the method
        :meth:`mapreader.download.tileserver_access.download_tileserver`.
        """
        self.metadata_query = []
        for one_item in self.found_queries:
            self.metadata_query.append(
                copy.deepcopy(self.metadata[one_item[3]])
            )

    def minmax_latlon(self):
        """
        Print the minimum and maximum longitude and latitude values for the
        metadata.

        Returns
            None
                Will print a result like this:

                .. code-block:: python
                
                    Min/Max Lon: <min_longitude>, <max_longitude>
                    Min/Max Lat: <min_latitude>, <max_latitude>

        Notes
        -----
        If the rectangle boundary has not been detected yet, the method checks
        if the boundaries of the map tiles have been detected. If not, it runs
        the method :meth:`mapreader.download.tileserver_access.create_info` to
        detect the boundaries.
        """
        if not self.detected_rect_boundary:
            self.create_info()

        min_lon = np.min(self.metadata_coord_arr[:, 0])
        max_lon = np.max(self.metadata_coord_arr[:, 1])
        min_lat = np.min(self.metadata_coord_arr[:, 2])
        max_lat = np.max(self.metadata_coord_arr[:, 3])

        print(f"Min/Max Lon: {min_lon}, {max_lon}")
        print(f"Min/Max Lat: {min_lat}, {max_lat}")

    def download_tileserver(
        self,
        mode: Optional[str] = "queries",
        num_img2test: Optional[int] = -1,
        zoom_level: Optional[int] = 14,
        # adjust_mult: Optional[float] = 0.005, # <-- not used in the function
        retries: Optional[int] = 10,
        scraper_max_connections: Optional[int] = 4,
        failed_urls_path: Optional[str] = "failed_urls.txt",
        tile_tmp_dir: Optional[str] = "tiles",
        output_maps_dirname: Optional[str] = "maps",
        output_metadata_filename: Optional[str] = "metadata.csv",
        pixel_closest: Optional[int] = None,
        redownload: Optional[bool] = False,
        id1: Optional[int] = 0,
        id2: Optional[int] = -1,
        error_path: Optional[str] = "errors.txt",
        max_num_errors: Optional[int] = 20,
    ) -> None:
        """
        Downloads map tiles from a tileserver using a scraper and stitches
        them into a larger map image.

        Parameters
        ----------
        mode : str, optional
            Metadata query type, which can be ``"queries"`` (default) or
            ``"query"``, both of which will download the queried maps. It can
            also be set to ``"all"``, which means that all maps in the
            metadata file will be downloaded.
        num_img2test : int, optional
            Number of images to download for testing, by default ``-1``.
        zoom_level : int, optional
            Zoom level to retrieve map tiles from, by default ``14``.
        retries : int, optional
            Number of times to retry a failed download, by default ``10``.
        scraper_max_connections : int, optional
            Maximum number of simultaneous connections for the scraper, by
            default ``4``.
        failed_urls_path : str, optional
            Path to save failed URLs, by default ``"failed_urls.txt"``.
        tile_tmp_dir : str, optional
            Directory to temporarily save map tiles, by default ``"tiles"``.
        output_maps_dirname : str, optional
            Directory to save combined map images, by default ``"maps"``.
        output_metadata_filename : str, optional
            Name of the output metadatata file, by default ``"metadata.csv"``.

            *Note: This file will be saved in the path equivalent to
            output_maps_dirname/output_metadata_filename.*
        pixel_closest : int, optional
            Adjust the number of pixels in both directions (width and height)
            after downloading a map. For example, if ``pixel_closest = 100``,
            the number of pixels in both directions will be multiples of 100.

            `This helps to create only square tiles in the processing step.`
        redownload : bool, optional
            Whether to redownload previously downloaded maps that already
            exist in the local directory, by default ``False``.
        id1 : int, optional
            The starting index (in the ``metadata`` property) for downloading
            maps, by default ``0``.
        id2 : int, optional
            The ending index (in the ``metadata`` property) for downloading
            maps, by default ``-1`` (all images).
        error_path : str, optional
            The path to the file for logging errors, by default
            ``"errors.txt"``.
        max_num_errors : int, optional
            The maximum number of errors to allow before skipping a map, by
            default ``20``.

        Returns
        -------
        None
        """

        if not os.path.isdir(output_maps_dirname):
            os.makedirs(output_maps_dirname)

        if not os.path.isdir("geojson"):
            os.makedirs("geojson")

        if mode in ["query", "queries"]:
            self.create_metadata_query()
            metadata = self.metadata_query
        elif mode == "all":
            metadata = self.metadata
        else:
            raise NotImplementedError(
                "Mode must be 'query', 'queries', or 'all'."
            )

        # Header and Footer for GeoJSON
        header, footer = create_hf(geom="polygon")

        if id2 < 0:
            metadata = metadata[id1:]
        elif id2 < id1:
            raise ValueError("id2 should be > id1.")
        else:
            metadata = metadata[id1:id2]

        # --- metadata file
        metadata_output_path = os.path.join(
            output_maps_dirname, output_metadata_filename
        )

        saved_metadata = None
        try_cond1 = try_cond2 = False
        if not os.path.isfile(metadata_output_path):
            with open(metadata_output_path, "w") as f:
                f.writelines("|name|url|coord|pub_date|region|polygon\n")
                counter = 0
        else:
            with open(metadata_output_path, "r") as f:
                counter = len(f.readlines()) - 1
            saved_metadata = pd.read_csv(metadata_output_path, delimiter="|")

        for metadata_item in metadata:
            try:
                # only for testing
                if counter == num_img2test:
                    break

                properties = metadata_item["properties"]
                geometry = metadata_item["geometry"]

                print("-------------------")
                output_stitcher = os.path.join(
                    output_maps_dirname,
                    "map_" + properties["IMAGE"] + ".png",
                )

                if saved_metadata is not None:
                    try_cond1 = (
                        str(geometry["coordinates"][0][0])
                        in saved_metadata["polygon"].to_list()
                    )
                    try_cond2 = (
                        f'map_{properties["IMAGE"]}.png'
                        in saved_metadata["name"].to_list()
                    )

                    if (not try_cond1) and (try_cond2):
                        append_id = int(metadata_item["id"].split(".")[1])
                        output_stitcher = os.path.join(
                            output_maps_dirname,
                            f'map_{properties["IMAGE"]}@{append_id}.png',
                        )

                if os.path.isfile(output_stitcher) and (not redownload):
                    print(f"File already exists: {output_stitcher}")
                    continue

                print("Retrieving: ", properties["IMAGEURL"])

                # failed_urls_path is created by scraper.py
                # remove the file and check if it will be created by scraper
                if os.path.isfile(failed_urls_path):
                    os.remove(failed_urls_path)

                values = geometry["coordinates"][0][0]
                str2write = header + str(values) + footer

                if (not try_cond1) and (try_cond2):
                    poly_filename = (
                        f'{properties["IMAGE"]}@{append_id}_{counter}.geojson'
                    )
                else:
                    poly_filename = f'{properties["IMAGE"]}_{counter}.geojson'

                f = open(os.path.join("geojson", poly_filename), "w")
                f.writelines(str2write)
                f.close()

                # Remove previously retrieved tiles if they exist
                if os.path.isdir(tile_tmp_dir):
                    shutil.rmtree(tile_tmp_dir)

                # Run the scraper
                scraper(
                    os.path.join("geojson", poly_filename),
                    zoom_level,
                    self.download_url,
                    tile_tmp_dir,
                    max_connections=scraper_max_connections,
                    retries=retries,
                )

                # If there are any failed URLs, log and continue
                if os.path.isfile(failed_urls_path):
                    with open(failed_urls_path, "r") as f:
                        errors = f.readlines()
                        if len(errors) > max_num_errors:
                            print(
                                f"[WARNING] Could not retrieve {len(errors)} tiles."  # noqa
                            )
                            with open(error_path, "a+") as f:
                                line2write = f"{counter}|"
                                line2write += f"{poly_filename}|"
                                line2write += f"{len(errors)}|"
                                line2write += (
                                    f"{os.path.basename(output_stitcher)}|"
                                )
                                line2write += f"{properties['IMAGEURL']}|"
                                line2write += f"{values}\n"
                                f.writelines(line2write)
                            continue

                # Run stitcher
                stitcher(tile_tmp_dir, output_stitcher, pixel_closest)

                # Collect the new bounding boxes according to the retrieved
                # tiles
                list_files = glob(os.path.join(tile_tmp_dir, "*png"))
                coord_info = collect_coord_info(list_files)

                (
                    name_region,
                    _,
                    _,
                    published_date,
                ) = self.extract_region_dates_metadata(metadata_item)

                with open(metadata_output_path, "a+") as f:
                    line2write = f"{counter}|"
                    line2write += f"{os.path.basename(output_stitcher)}|"
                    line2write += f"{metadata_item['properties']['IMAGEURL']}|"
                    line2write += f"{coord_info}|"
                    line2write += f"{published_date}|"
                    line2write += f"{name_region}|"
                    line2write += f"{values}\n"
                    f.writelines(line2write)
            except Exception as err:
                print(f"[ERROR] {err}")
                with open(error_path, "a+") as f:
                    line2write = f"{counter}|"
                    line2write += f"{poly_filename}|"
                    line2write += "EXCEPTION|"
                    line2write += f"{os.path.basename(output_stitcher)}|"
                    line2write += f"{metadata_item['properties']['IMAGEURL']}|"
                    line2write += f"{values}\n"
                    f.writelines(line2write)

            counter += 1

        # clean-up
        if os.path.isdir(tile_tmp_dir):
            shutil.rmtree(tile_tmp_dir)

    def extract_region_dates_metadata(
        self, metadata_item: dict
    ) -> Tuple[str, int, int, int]:
        """
        Extracts region name, surveyed date, revised date, and published date
        from a given GeoJSON feature, provided as a ``metadata_item``.

        Parameters
        ----------
        metadata_item : dict
            A GeoJSON feature, i.e. a dictionary which contains at least a
            nested dictionary in in the ``"properties"`` key, which contains a
            ``"WFS_TITLE"`` value.

        Returns
        -------
        Tuple[str, int, int, int]
            A tuple containing the region name (str), surveyed date (int),
            revised date (int), and published date (int). If any of the dates
            cannot be found, its value will be ``-1``. If no region name can
            be found, it will be ``"None"``.
        """
        # Initialize variables
        name_region = "None"
        surveyed_date = -1
        revised_date = -1
        published_date = -1

        try:
            one_item_split = metadata_item["properties"]["WFS_TITLE"].split(
                ","
            )
            name_region = one_item_split[0].strip()
            for ois in one_item_split[1:]:
                if "surveyed" in ois.lower():
                    surveyed_date = self.find_and_clean_date(
                        ois, ois_key="surveyed"
                    )
                if "revised" in ois.lower():
                    revised_date = self.find_and_clean_date(
                        ois, ois_key="revised"
                    )
                if "published" in ois.lower():
                    published_date = self.find_and_clean_date(
                        ois, ois_key="published"
                    )
        except:
            pass

        return name_region, surveyed_date, revised_date, published_date

    @staticmethod
    def find_and_clean_date(
        ois: str, ois_key: Optional[str] = "surveyed"
    ) -> str:
        """
        Find and extract a date string from a given string (``ois``), typically
        representing a date metadata attribute. The date string is cleaned by
        removing unnecessary tokens like "to" and "ca.".

        Parameters
        ----------
        ois : str
            A string containing the date metadata attribute and its value.
        ois_key : str, optional
            The keyword used to identify the date metadata attribute, by
            default ``"surveyed"``.

        Returns
        -------
        str
            The extracted date string, cleaned of unnecessary tokens.
        """

        found_date = ois.lower().split(ois_key + ": ")[1].strip()

        for strip_token in ["to", "ca."]:
            if strip_token in found_date:
                found_date = found_date.split(strip_token)[1].strip()

        return found_date

    def plot_metadata_on_map(
        self,
        list2remove: Optional[List[str]] = [],
        map_extent: Optional[
            Union[
                Literal["uk"], List[float], Tuple[float, float, float, float]
            ]
        ] = None,
        add_text=False,
    ) -> None:
        """
        Plots metadata on a map (using ``cartopy`` library, if available).

        Parameters
        ----------
        list2remove : list, optional
            A list of IDs to remove from the plot. The default is ``[]``.
        map_extent : tuple or list, optional
            The extent of the map to be plotted. It should be a tuple or a
            list of the format ``[lon_min, lon_max, lat_min, lat_max]``. It
            can also be set to ``"uk"`` which will limit the map extent to the
            UK's boundaries. The default is ``None``.
        add_text : bool, optional
            If ``True``, adds ID texts next to each plotted metadata. The
            default is ``False``.

        Returns
        -------
        None
        """
        # Check for cartopy
        cartopy_installed = False
        try:
            import cartopy.crs as ccrs

            cartopy_installed = True
        except ImportError:
            print("[WARNING] cartopy is not installed!")

        # Setup the figure/map
        plt.figure(figsize=(15, 15))
        if cartopy_installed:
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.coastlines(resolution="10m", color="black", linewidth=1)
            if map_extent == "uk":
                extent = [-8.08999993, 1.81388127, 49.8338702, 60.95000002]
                ax.set_extent(extent)
            elif type(map_extent) in [list, tuple]:
                ax.set_extent(map_extent)
            else:
                pass
        else:
            ax = plt.axes()

        for i in range(len(self.metadata)):
            # Drop if ID in list2remove
            text_id = self.metadata[i]["id"].split(".")[1]
            if text_id in list2remove:
                continue

            # Get coordinates
            coords = np.array(
                self.metadata[i]["geometry"]["coordinates"][0][0]
            )

            # Plot coordinates
            if cartopy_installed:
                plt.plot(
                    coords[:, 0],
                    coords[:, 1],
                    c="r",
                    alpha=0.5,
                    transform=ccrs.Geodetic(),
                )
            else:
                plt.plot(coords[:, 0], coords[:, 1], c="r", alpha=0.5)

            # Add text to plot (if add_text is True)
            if add_text:
                plt.text(
                    np.mean(coords[:, 0]) - 0.15,
                    np.mean(coords[:, 1]) - 0.05,
                    f"{text_id}",
                    color="r",
                )

        # Add gridlines
        if cartopy_installed:
            ax.gridlines()
        else:
            plt.grid()

        plt.show()

    def hist_published_dates(
        self, min_date: Optional[int] = None, max_date: Optional[int] = None
    ) -> None:
        """
        Plot a histogram of the published dates for all metadata items.

        Parameters
        ----------
        min_date : int, optional
            Minimum published date to be included in the histogram. If not
            given, the minimum published date among all metadata items will be
            used.
        max_date : int, optional
            Maximum published date to be included in the histogram. If not
            given, the maximum published date among all metadata items will be
            used.

        Raises
        ------
        ValueError
            If any of the published dates cannot be converted to an integer.

        Returns
        -------
        None

        Notes
        -----
        The method extracts the published date from each metadata item using
        the method
        :meth:`mapreader.download.tileserver_access.extract_region_dates_metadata`
        and creates a histogram of the counts of published dates falling
        within the given range. The histogram is plotted using
        `matplotlib.pyplot.hist`.

        If `min_date` or `max_date` are given, only the published dates
        falling within that range will be included in the histogram. Otherwise,
        the histogram will include all published dates in the metadata.
        """
        # Get all published dates
        all_published_date = []
        for metadata_item in self.metadata:
            _, _, _, published_date = self.extract_region_dates_metadata(
                metadata_item
            )
            all_published_date.append(int(published_date))

        # Get min/max date for all published dates
        _min_date = min(all_published_date)
        _max_date = max(all_published_date)
        print(f"Min/Max published dates: {_min_date}, {_max_date}")

        # Set min_date and max_date accordingly
        if min_date is None:
            min_date = _min_date
        if max_date is None:
            max_date = _max_date

        # Plot
        plt.figure(figsize=(10, 5))
        plt.hist(
            all_published_date,
            align="mid",
            bins=np.arange(int(min_date) - 0.5, int(max_date) + 0.5, 1),
            color="k",
        )
        plt.grid()
        plt.xticks(size=14)
        plt.yticks(size=14)
        plt.xlabel("Published date", size=18)
        plt.ylabel("Counts", size=18)
        plt.show()

    def __str__(self):
        info = f"Metadata file has {self.__len__()} items."
        info += f"\nDownload URL: {self.download_url}"
        info += f"\nGeometry: {self.geometry}"
        return info

    def __len__(self):
        return len(self.metadata)

    def download_tileserver_rect(
        self,
        mode: Optional[str] = "queries",
        num_img2test: Optional[int] = -1,
        zoom_level: Optional[int] = 14,
        adjust_mult: Optional[float] = 0.005,
        retries: Optional[int] = 1,
        failed_urls_path: Optional[str] = "failed_urls.txt",
        tile_tmp_dir: Optional[str] = "tiles",
        output_maps_dirname: Optional[str] = "maps",
        output_metadata_filename: Optional[str] = "metadata.csv",
        pixel_closest: Optional[int] = None,
        redownload: Optional[bool] = False,
        id1: Optional[int] = 0,
        id2: Optional[int] = -1,
        min_lat_len: Optional[float] = 0.05,
        min_lon_len: Optional[float] = 0.05,
    ):
        """
        Downloads map tiles from a tileserver using a scraper and stitches
        them into a larger map image.

        Parameters
        ----------
        mode : str, optional
            Metadata query type, which can be ``"queries"`` (default) or
            ``"query"``, both of which will download the queried maps. It can
            also be set to ``"all"``, which means that all maps in the
            metadata file will be downloaded.
        num_img2test : int, optional
            Number of images to download for testing, by default ``-1``.
        zoom_level : int, optional
            Zoom level to retrieve map tiles from, by default ``14``.
        adjust_mult : float, optional
            If some tiles cannot be downloaded, shrink the requested bounding
            box by this factor. Defaults to ``0.005``.
        retries : int, optional
            Number of times to retry a failed download, by default ``10``.
        failed_urls_path : str, optional
            Path to save failed URLs, by default ``"failed_urls.txt"``.
        tile_tmp_dir : str, optional
            Directory to temporarily save map tiles, by default ``"tiles"``.
        output_maps_dirname : str, optional
            Directory to save combined map images, by default ``"maps"``.
        output_metadata_filename : str, optional
            Name of the output metadata file, by default ``"metadata.csv"``.

            *Note: This file will be saved in the path equivalent to
            output_maps_dirname/output_metadata_filename.*
        pixel_closest : int, optional
            Adjust the number of pixels in both directions (width and height)
            after downloading a map. For example, if ``pixel_closest = 100``,
            the number of pixels in both directions will be multiples of 100.

            `This helps to create only square tiles in the processing step.`
        redownload : bool, optional
            Whether to redownload previously downloaded maps that already
            exist in the local directory, by default ``False``.
        id1 : int, optional
            The starting index (in the ``metadata`` property) for downloading
            maps, by default ``0``.
        id2 : int, optional
            The ending index (in the ``metadata`` property) for downloading
            maps, by default ``-1`` (all images).
        min_lat_len : float, optional
            Minimum length of the latitude (in degrees) of the bounding box
            for each tileserver request. Default is ``0.05``.
        min_lon_len : float, optional
            Minimum length of the longitude (in degrees) of the bounding box
            for each tileserver request. Default is ``0.05``.

        Returns
        -------
        None

        Notes
        -----
        The ``min_lat_len`` and ``min_lon_len`` are optional float parameters
        that represent the minimum length of the latitude and longitude,
        respectively, of the bounding box for each tileserver request. These
        parameters are used in the method to adjust the boundary of the map
        tile to be requested from the server. If the difference between the
        maximum and minimum latitude or longitude is less than the
        corresponding min_lat_len or min_lon_len, respectively, then the
        ``adjust_mult`` parameter is used to shrink the boundary until the
        minimum length requirements are met.
        """

        if not os.path.isdir(output_maps_dirname):
            os.makedirs(output_maps_dirname)

        if not os.path.isdir("geojson"):
            os.makedirs("geojson")

        if mode in ["query", "queries"]:
            self.create_metadata_query()
            metadata = self.metadata_query
        elif mode == "all":
            metadata = self.metadata
        else:
            raise NotImplementedError(
                "Mode must be 'query', 'queries', or 'all'."
            )

        # Header and Footer for GeoJSON
        header, footer = create_hf(geom="polygon")

        if id2 < 0:
            metadata = metadata[id1:]
        elif id2 < id1:
            raise ValueError("id2 should be > id1.")
        else:
            metadata = metadata[id1:id2]

        # Dataframe to collect some information about results
        result_df = pd.DataFrame(
            columns=["name", "url", "coord", "pub_date", "region"]
        )
        counter = 0
        for metadata_item in metadata:
            # only for testing
            if counter == num_img2test:
                break

            properties = metadata_item["properties"]
            geometry = metadata_item["geometry"]

            print("-------------------")
            output_stitcher = os.path.join(
                output_maps_dirname,
                "map_" + properties["IMAGE"] + ".png",
            )
            if os.path.isfile(output_stitcher) and (not redownload):
                print(f"File already exists: {output_stitcher}")
                continue

            print("Retrieving: ", properties["IMAGEURL"])

            # Change the request to a rectangular one by finding min/max of
            # lats/lons
            (
                min_lon,
                max_lon,
                min_lat,
                max_lat,
            ) = self.detect_rectangle_boundary(geometry["coordinates"][0][0])

            # only save an image if no error was raised
            not_saved = True
            while not_saved:
                # failed_urls_path is created by scraper.py
                # remove the file and check if it will be created by scraper
                if os.path.isfile(failed_urls_path):
                    os.remove(failed_urls_path)

                # --- Create a geojson file
                values = """[
                        [{0},  {1}],
                        [{2},  {3}],
                        [{4},  {5}],
                        [{6},  {7}],
                        [{8},  {9}]]""".format(
                    min_lon,
                    min_lat,
                    max_lon,
                    min_lat,
                    max_lon,
                    max_lat,
                    min_lon,
                    max_lat,
                    min_lon,
                    min_lat,
                )
                str2write = header + values + footer

                poly_filename = properties["IMAGE"] + "_{}.geojson".format(
                    counter
                )
                with open(os.path.join("geojson", poly_filename), "w") as f:
                    f.writelines(str2write)

                if os.path.isdir(tile_tmp_dir):
                    # remove previously retrieved tiles
                    shutil.rmtree(tile_tmp_dir)

                # Run the scraper
                scraper(
                    os.path.join("geojson", poly_filename),
                    zoom_level,
                    self.download_url,
                    tile_tmp_dir,
                    retries=retries,
                )

                # If any failed URLs, adjust the request. The request is
                # adjusted by decreasing the requested bounding box
                if os.path.isfile(failed_urls_path):
                    with open(failed_urls_path, "r") as f:
                        errors = f.readlines()
                        if len(errors) > 0:
                            print(
                                f"[WARNING] Could not retrieve {len(errors)} tiles, changing the coordinates"  # noqa
                            )
                            lat_len = abs(max_lat - min_lat)
                            lon_len = abs(max_lon - min_lon)
                            if (lat_len <= min_lat_len) or (
                                lon_len <= min_lon_len
                            ):
                                not_saved = True
                                break
                            min_lat += adjust_mult * lat_len
                            min_lon += adjust_mult * lon_len
                            max_lat -= adjust_mult * lat_len
                            max_lon -= adjust_mult * lon_len
                            print(lat_len, lon_len)
                        else:
                            not_saved = False
                else:
                    not_saved = False

            if not_saved:
                continue

            # Run stitcher
            stitcher(tile_tmp_dir, output_stitcher, pixel_closest)

            # Collect the new bounding boxes according to the retrieved tiles
            list_files = glob(os.path.join(tile_tmp_dir, "*png"))
            coord_info = collect_coord_info(list_files)

            (
                name_region,
                _,
                _,
                published_date,
            ) = self.extract_region_dates_metadata(metadata_item)

            result_df.loc[counter] = [
                os.path.basename(output_stitcher),
                properties["IMAGEURL"],
                coord_info,
                published_date,
                name_region,
            ]
            counter += 1

        # clean-up
        if os.path.isdir(tile_tmp_dir):
            shutil.rmtree(tile_tmp_dir)

        result_file = os.path.join(
            output_maps_dirname, output_metadata_filename
        )
        if (not redownload) and (os.path.isfile(result_file)):
            result_df.to_csv(result_file, mode="a+", header=False)
        else:
            result_df.to_csv(result_file, mode="w", header=True)


def download_tileserver_parallel(
    metadata: List[Dict], start: int, end: int, process_np: int = 8, **kwds
) -> None:
    """
    Downloads map tiles from a tileserver in parallel using multiprocessing.

    .. warning::
        This function does currently not work.

    Parameters
    ----------
    metadata : list of dictionaries
        A list of metadata dictionaries (in GeoJSON format) determining what
        to download from the tile server. Each dictionary that contains info
        about the maps to be downloaded needs to have three keys:
        ``"features"``, ``"geometry"`` (with a nested list of
        ``"coordinates"``), and ``"properties"`` (with two values for the keys
        ``"IMAGEURL"`` AND ``"IMAGE"``).
    start : int
        The index of the first element in the ``metadata`` property to
        download.
    end : int
        The index of the last element in ``metadata`` property to download.
    process_np : int, optional
        The number of processes to use for downloading. Defaults to ``8``.
    **kwds : dict, optional
        Keyword arguments passed to the ``download_tileserver`` function.

    Returns
    -------
    None

    ..
        Note/TODO: This function will not work, as download_tileserver is not
        defined in this scope. It belongs to the TileServer class...
    """

    if end < 0:
        end = len(metadata)

    req_proc = min(end, process_np)

    step = int((end - start) / req_proc + 1)

    jobs = []
    for index in range(req_proc):
        starti = start + index * step
        endi = min(start + (index + 1) * step, end)
        if starti >= endi:
            break

        metadata_one_node = metadata[starti:endi]
        p = multiprocessing.Process(
            target=download_tileserver,  # TODO: Bug
            args=(
                metadata_one_node,
                kwds["geometry"],
                kwds["num_img2test"],
                kwds["download_url"],
                kwds["zoom_level"],
                kwds["adjust_mult"],
                kwds["retries"],
                f'{index}_{kwds["failed_urls_path"]}',
                f'{index}_{kwds["tile_tmp_dir"]}',
                f'{kwds["output_maps_dirname"]}',
                f'{index}_{kwds["output_metadata_filename"]}',
            ),
        )
        jobs.append(p)
    for i in range(len(jobs)):
        jobs[i].start()

    check_par_jobs(jobs)

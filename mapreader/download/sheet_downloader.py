from __future__ import annotations

import os
import re
import shutil
import time
import urllib
import urllib.request

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import LineString, Point, Polygon
from tqdm.auto import tqdm

from mapreader.utils.load_frames import load_from_geojson

from .downloader_utils import get_grid_bb_from_polygon, get_polygon_from_grid_bb
from .tile_loading import DEFAULT_TEMP_FOLDER, TileDownloader
from .tile_merging import TileMerger


class SheetDownloader:
    """
    A class to download map sheets using metadata.
    """

    def __init__(
        self,
        metadata_path: str,
        download_url: str | list,
    ) -> None:
        """
        Initialize SheetDownloader class

        Parameters
        ----------
        metadata_path : str
            path to metadata.json
        download_url : Union[str, list]
            The base URL pattern used to download tiles from the server. This
            should contain placeholders for the x coordinate (``x``), the y
            coordinate (``y``) and the zoom level (``z``).
        """
        self.found_queries = gpd.GeoDataFrame()
        self.merged_polygon = None

        assert isinstance(
            metadata_path, str
        ), "[ERROR] Please pass metadata_path as string."

        if os.path.isfile(metadata_path):
            self.metadata = load_from_geojson(metadata_path)
            print(self.__str__())
        else:
            raise ValueError("[ERROR] Metadata file not found.")

        if isinstance(download_url, str):
            my_ts = [download_url]
        elif isinstance(download_url, list):
            my_ts = download_url
        else:
            raise ValueError(
                "[ERROR] Please pass ``download_url`` as string or list of strings."
            )

        self.tile_server = my_ts

        self.crs = self.metadata.crs.to_string()

    def __str__(self) -> str:
        info = f"[INFO] Metadata file has {self.__len__()} item(s)."
        return info

    def __len__(self) -> int:
        return len(self.metadata)

    def get_grid_bb(self, zoom_level: int | None = 14) -> None:
        """
        Creates a grid bounding box for each map in metadata.

        Parameters
        ----------
        zoom_level : int, optional
            The zoom level to use when creating the grid bounding box.
            Later used when downloading maps, by default 14.
        """
        if self.crs != "EPSG:4326":
            raise NotImplementedError(
                "[ERROR] At the moment, MapReader can only create grid bounding boxes and download map sheets using coordinates in WGS1984 (aka EPSG:4326)."
            )

        self.metadata["grid_bb"] = self.metadata["geometry"].apply(
            lambda x: get_grid_bb_from_polygon(x, zoom_level)
        )

    def extract_wfs_id_nos(self) -> None:
        """
        Extracts WFS ID numbers from metadata.
        """
        self.metadata["wfs_id_no"] = self.metadata["id"].apply(
            lambda x: int(x.split(sep=".")[-1])
        )  # extract WFS ID number

    def extract_published_dates(
        self,
        date_col: str | None = None,
    ) -> None:
        """
        Extracts publication dates from metadata.

        Parameters
        ----------
            date_col : str or None, optional
                A string indicating the metadata column containing the publication date.
                If  None, "WFS_TITLE" will be used. Date will then be extracted by regex searching for "Published: XXX".
                By default None.
        """
        if date_col:
            if not isinstance(date_col, str):
                raise ValueError("[ERROR] Please pass ``date_col`` as a string.")

            if date_col in self.metadata.columns:
                self.metadata["published_date"] = self.metadata[date_col].apply(
                    self._convert_date_to_int
                )
            else:
                raise ValueError(f"[ERROR] {date_col} not found in metadata columns.")

        else:
            self.metadata["published_date"] = self.metadata["WFS_TITLE"].apply(
                self._extract_date_from_wfs_title
            )

        self.metadata["published_date"] = self.metadata["published_date"].astype(
            "Int64"
        )  # convert to Int64

        n_missing = self.metadata["published_date"].isna().sum()
        if n_missing > 0:
            if n_missing == len(self.metadata) and date_col:
                raise ValueError(
                    "[ERROR] No publication dates found. Please check your `date_col`"
                )
            print(
                f'[INFO] {self.metadata["published_date"].isna().sum()} maps are missing publication dates.'
            )

    @staticmethod
    def _extract_date_from_wfs_title(x):
        date = re.findall(r"Published.*[\D]([\d]+)", x, flags=re.IGNORECASE)
        if len(date) == 0:  # no date found
            return None
        elif len(date) == 1:  # single date found
            return int(date[0])
        elif len(date) > 1:  # multiple dates found
            print("[WARNING] Multiple published dates detected. Using last date.")
            return int(date[-1])

    @staticmethod
    def _convert_date_to_int(x):
        try:
            return int(x)
        except ValueError:
            return None

    def get_merged_polygon(self) -> None:
        """
        Creates a multipolygon representing all maps in metadata.
        """
        self.merged_polygon = self.metadata["geometry"].unary_union

    def get_minmax_latlon(self) -> None:
        """
        Prints minimum and maximum latitudes and longitudes of all maps in metadata.
        """
        if self.merged_polygon is None:
            self.get_merged_polygon()

        min_x, min_y, max_x, max_y = self.merged_polygon.bounds
        print(
            f"[INFO] Min lat: {min_y}, max lat: {max_y} \n\
[INFO] Min lon: {min_x}, max lon: {max_x}"
        )

    ## queries
    def query_map_sheets_by_wfs_ids(
        self,
        wfs_ids: list | int,
        append: bool | None = False,
        print: bool | None = False,
    ) -> None:
        """
        Find map sheets by WFS ID numbers.

        Parameters
        ----------
        wfs_ids : Union[list, int]
            The WFS ID numbers of the maps to download.
        append : bool, optional
            Whether to append to current query results or start over.
            By default False
        print: bool, optional
            Whether to print query results or not.
            By default False
        """
        if "wfs_id_no" not in self.metadata.columns:
            self.extract_wfs_id_nos()

        if isinstance(wfs_ids, list):
            requested_maps = wfs_ids
        elif isinstance(wfs_ids, int):
            requested_maps = [wfs_ids]
        else:
            raise ValueError("[ERROR] Please pass ``wfs_ids`` as int or list of ints.")

        if not append:
            self.found_queries = gpd.GeoDataFrame()  # reset each time

        self.found_queries = pd.concat(
            [
                self.found_queries,
                self.metadata[self.metadata["wfs_id_no"].isin(requested_maps)],
            ]
        ).drop_duplicates(subset="id")

        if print:
            self.print_found_queries()

    def query_map_sheets_by_polygon(
        self,
        polygon: Polygon,
        mode: str | None = "within",
        append: bool | None = False,
        print: bool | None = False,
    ) -> None:
        """
        Find map sheets which are found within or intersecting with a defined polygon.

        Parameters
        ----------
        polygon : Polygon
            shapely Polygon
        mode : str, optional
            The mode to use when finding maps.
            Options of ``"within"``, which returns all map sheets which are completely within the defined polygon,
            and ``"intersects""``, which returns all map sheets which intersect/overlap with the defined polygon.
            By default "within".
        append : bool, optional
            Whether to append to current query results or start over.
            By default False
        print: bool, optional
            Whether to print query results or not.
            By default False

        Notes
        -----
        Use ``create_polygon_from_latlons()`` to create polygon.
        """
        if not isinstance(polygon, Polygon):
            raise ValueError(
                "[ERROR] Please pass polygon as shapely.geometry.Polygon object.\n\
[HINT] Use ``create_polygon_from_latlons()`` to create polygon."
            )

        if self.merged_polygon is None:
            self.get_merged_polygon()

        if self.merged_polygon.disjoint(polygon):
            raise ValueError("[ERROR] Polygon is out of map metadata bounds.")

        if mode not in ["within", "intersects"]:
            raise NotImplementedError(
                '[ERROR] Please use ``mode="within"`` or ``mode="intersects"``.'
            )

        if not append:
            self.found_queries = gpd.GeoDataFrame()  # reset each time

        if mode == "within":
            self.found_queries = pd.concat(
                [
                    self.found_queries,
                    self.metadata[self.metadata["geometry"].within(polygon)],
                ]
            ).drop_duplicates()
        elif mode == "intersects":
            self.found_queries = pd.concat(
                [
                    self.found_queries,
                    self.metadata[self.metadata["geometry"].intersects(polygon)],
                ]
            ).drop_duplicates(subset="id")

        if print:
            self.print_found_queries()

    def query_map_sheets_by_coordinates(
        self,
        coords: tuple,
        append: bool | None = False,
        print: bool | None = False,
    ) -> None:
        """
        Find maps sheets which contain a defined set of coordinates.
        Coordinates are (x,y).

        Parameters
        ----------
        coords : tuple
            Coordinates in ``(x,y)`` format.
        append : bool, optional
            Whether to append to current query results or start over.
            By default False
        print: bool, optional
            Whether to print query results or not.
            By default False
        """
        if not isinstance(coords, tuple):
            raise ValueError("[ERROR] Please pass coords as a tuple in the form (x,y).")

        coords = Point(coords)

        if self.merged_polygon is None:
            self.get_merged_polygon()

        if self.merged_polygon.disjoint(coords):
            raise ValueError("[ERROR] Coordinates are out of map metadata bounds.")

        if not append:
            self.found_queries = gpd.GeoDataFrame()  # reset each time

        self.found_queries = pd.concat(
            [
                self.found_queries,
                self.metadata[self.metadata["geometry"].contains(coords)],
            ]
        ).drop_duplicates(subset="id")

        if print:
            self.print_found_queries()

    def query_map_sheets_by_line(
        self,
        line: LineString,
        append: bool | None = False,
        print: bool | None = False,
    ) -> None:
        """
        Find maps sheets which intersect with a line.

        Parameters
        ----------
        line : LineString
            shapely LineString
        append : bool, optional
            Whether to append to current query results or start over.
            By default False
        print: bool, optional
            Whether to print query results or not.
            By default False

        Notes
        -----
        Use ``create_line_from_latlons()`` to create line.
        """

        if not isinstance(line, LineString):
            raise ValueError(
                "[ERROR] Please pass line as shapely.geometry.LineString object.\n\
[HINT] Use ``create_line_from_latlons()`` to create line."
            )

        if self.merged_polygon is None:
            self.get_merged_polygon()

        if self.merged_polygon.disjoint(line):
            raise ValueError("[ERROR] Line is out of map metadata bounds.")

        if not append:
            self.found_queries = gpd.GeoDataFrame()  # reset each time

        self.found_queries = pd.concat(
            [
                self.found_queries,
                self.metadata[self.metadata["geometry"].intersects(line)],
            ]
        ).drop_duplicates(subset="id")

        if print:
            self.print_found_queries()

    def query_map_sheets_by_string(
        self,
        string: str,
        columns: str | list | None = None,
        append: bool | None = False,
        print: bool | None = False,
    ) -> None:
        """
        Find map sheets by searching for a string in the metadata.

        Parameters
        ----------
        string : str
            The string to search for.
            Can be raw string and use regular expressions.
        columns : str or list, optional
            A column or list of columns to search in.
            If ``None``, will search in all metadata fields.
        append : bool, optional
            Whether to append to current query results or start over.
            By default False
        print: bool, optional
            Whether to print query results or not.
            By default False

        Notes
        -----
        ``string`` is case insensitive.
        """
        if not isinstance(string, str):
            raise ValueError("[ERROR] Please pass ``string`` as a string.")

        if columns is None:
            columns = [*self.metadata.columns]
        if isinstance(columns, str):
            columns = [columns]
        if not isinstance(columns, list):
            raise ValueError("[ERROR] Please pass key(s) as string or list of strings.")

        if not append:
            self.found_queries = gpd.GeoDataFrame()  # reset each time

        self.found_queries = pd.concat(
            [
                self.found_queries,
                self.metadata[
                    self.metadata[columns].apply(
                        lambda x: bool(re.search(string, str(x.values), re.IGNORECASE)),
                        axis=1,
                    )
                ],
            ]
        ).drop_duplicates(subset="id")

        if print:
            self.print_found_queries()

    def print_found_queries(self) -> None:
        """
        Prints query results.
        """

        if len(self.found_queries) == 0:
            print("[INFO] No query results found/saved.")
        else:
            divider = 14 * "="
            print(f"{divider}\nQuery results:\n{divider}")
            for i in self.found_queries.index:
                map_url = self.found_queries.loc[i, "IMAGEURL"]
                map_bounds = self.found_queries.loc[i, "geometry"].bounds
                print(f"URL:     \t{map_url}")
                print(f"coordinates (bounds):  \t{map_bounds}")
                print(20 * "-")

    ## download
    def _initialise_downloader(self):
        """
        Initialise TileDownloader object
        """
        self.downloader = TileDownloader(self.tile_server)

    def _initialise_merger(self, path_save: str):
        """
        Initialise TileMerger object.

        Parameters
        ----------
        path_save : str
            Path to save merged items (i.e. whole map sheets)
        """
        self.merger = TileMerger(output_folder=f"{path_save}/")

    def _check_map_sheet_exists(self, feature: gpd.GeoSeries, metadata_fname) -> bool:
        """
        Checks if a map sheet is already saved.

        Parameters
        ----------
        feature : gpd.GeoSeries

        Returns
        -------
        bool
            img_path if file exists, False if not.
        """
        path_save = self.merger.output_folder

        try:
            # get image id with same coords in metadata
            existing_metadata_df = pd.read_csv(
                f"{path_save}{metadata_fname}", sep=",", index_col=0
            )
        except FileNotFoundError:
            return False

        polygon = get_polygon_from_grid_bb(feature["grid_bb"])
        if str(polygon.bounds) in existing_metadata_df["coordinates"].values:
            image_id = existing_metadata_df[
                existing_metadata_df["coordinates"] == str(polygon.bounds)
            ].iloc[0]["name"]
        else:
            return False  # coordinates not in metadata means image doesn't exist

        if os.path.exists(f"{path_save}{image_id}"):
            try:
                # check image is valid
                mpimg.imread(f"{path_save}{image_id}")
                return image_id
            except OSError:
                return False
        return False

    def _download_map(
        self,
        feature: gpd.GeoSeries,
        existing_id: str | bool,
        download_in_parallel: bool = True,
        overwrite: bool = False,
        error_on_missing_map=True,
    ) -> str | bool:
        """
        Downloads a single map sheet and saves as png file.

        Parameters
        ----------
        feature : gpd.GeoSeries
            The feature for which to download the map sheet.
        existing_id : str | bool
            The existing image id if the map sheet already exists.
        download_in_parallel : bool, optional
            Whether to download tiles in parallel, by default ``True``.
        overwrite : bool, optional
            Whether to overwrite existing maps, by default ``False``.
        error_on_missing_map : bool, optional
            Whether to raise an error if a map sheet is missing, by default True.

        Returns
        -------
        str or bool
            image path if map was downloaded successfully, False if not.
        """
        self.downloader.download_tiles(
            feature["grid_bb"], download_in_parallel=download_in_parallel
        )

        if existing_id is False:
            map_name = f"map_{feature['IMAGE']}"
        else:
            map_name = existing_id[:-4]  # remove file extension (assuming .png)

        img_path, success = self.merger.merge(
            feature["grid_bb"],
            file_name=map_name,
            overwrite=overwrite,
            error_on_missing_map=error_on_missing_map,
        )

        if success:
            print(f'[INFO] Downloaded "{img_path}"')
        else:
            print("[WARNING] Download unsuccessful.")

        # Try to remove the temporary folder
        try:
            shutil.rmtree(DEFAULT_TEMP_FOLDER)
        except PermissionError:
            # try again
            time.sleep(5)
            shutil.rmtree(DEFAULT_TEMP_FOLDER)
        except OSError:
            # try again
            time.sleep(5)
            shutil.rmtree(DEFAULT_TEMP_FOLDER)

        return img_path

    def _save_metadata(
        self,
        feature: gpd.GeoSeries,
        out_filepath: str,
        img_path: str,
        metadata_to_save: dict | None = None,
        **kwargs: dict | None,
    ) -> None:
        """
        Saves selected metadata to a csv file.
        If file exists, metadata list is appended.

        Parameters
        ----------
        feature : gpd.GeoSeries
            The feature for which to extract the metadata from
        out_filepath : str
            The path to save metadata csv.
        img_path : str
            The path to the downloaded map sheet.
        metadata_to_save : dict, optional
            A dictionary containing a mapping between desired column names (str) and metadata columns (str) to save to metadata csv.
            e.g. ``{"county": "COUNTY", "id": "id"}``
        **kwargs: dict, optional
            Keyword arguments to pass to the
            :meth:`~.download.sheet_downloader.SheetDownloader.extract_published_dates`
            method.

        Notes
        -----
        Default metadata items are: ``name``, ``url``, ``coordinates``,
        ``crs``, ``published_date``, ``grid_bb``.
        Additional items can be added using the ``metadata_to_save`` argument.
        """
        metadata_cols = [
            "name",
            "url",
            "coordinates",
            "crs",
            "published_date",
            "grid_bb",
        ]
        metadata_dict = {col: None for col in metadata_cols}

        # get default metadata
        metadata_dict["name"] = os.path.basename(img_path)
        metadata_dict["url"] = str(feature["IMAGEURL"])

        if "published_date" not in feature.keys():
            date_col = kwargs.get("date_col", None)
            if date_col:
                date = self._convert_date_to_int(feature[date_col])
            else:
                date = self._extract_date_from_wfs_title(feature["WFS_TITLE"])
            metadata_dict["published_date"] = date
        else:
            metadata_dict["published_date"] = feature["published_date"]

        metadata_dict["grid_bb"] = feature["grid_bb"]

        polygon = get_polygon_from_grid_bb(
            metadata_dict["grid_bb"]
        )  # use grid_bb to get coords of actually downloaded tiles
        metadata_dict["coordinates"] = polygon.bounds
        metadata_dict["crs"] = self.crs

        if metadata_to_save:
            for col, metadata_col in metadata_to_save.items():
                if not isinstance(metadata_col, str):
                    raise ValueError(
                        "[ERROR] Please pass ``metadata_to_save`` metadata columns as strings."
                    )

                if metadata_col not in feature.keys():
                    raise KeyError(f"[ERROR] {metadata_col} not found in feature.")

                metadata_dict[col] = feature[metadata_col]

        new_metadata_df = pd.DataFrame.from_dict(metadata_dict, orient="index").T

        if os.path.exists(out_filepath):
            existing_metadata_df = pd.read_csv(out_filepath, sep=",", index_col=0)
            metadata_df = pd.concat(
                [existing_metadata_df, new_metadata_df], ignore_index=True
            )
            metadata_df = metadata_df.loc[
                metadata_df.astype(str).drop_duplicates(subset=metadata_cols).index
            ]  # https://stackoverflow.com/questions/43855462/pandas-drop-duplicates-method-not-working-on-dataframe-containing-lists
        else:
            metadata_df = new_metadata_df

        metadata_df.to_csv(out_filepath, sep=",")

    def _download_map_sheets(
        self,
        features: gpd.GeoDataFrame,
        path_save: str | None = "maps",
        metadata_fname: str | None = "metadata.csv",
        overwrite: bool | None = False,
        download_in_parallel: bool = True,
        force: bool = False,
        error_on_missing_map: bool = True,
        **kwargs: dict | None,
    ):
        """Download map sheets from features.

        Parameters
        ----------
        features : gpd.GeoDataFrame
            The features to download map sheets for.
        path_save : str, optional
            Path to save map sheets, by default "maps"
        metadata_fname : str, optional
            Name to use for metadata file, by default "metadata.csv"
        overwrite : bool, optional
            Whether to overwrite existing maps, by default ``False``.
        download_in_parallel : bool, optional
            Whether to download tiles in parallel, by default ``True``.
        force : bool, optional
            Whether to force the download or ask for confirmation, by default ``False``.
        error_on_missing_map : bool, optional
            Whether to raise an error if a map sheet is missing, by default True.
        **kwargs : dict, optional
            Keyword arguments to pass to the
            :meth:`~.download.sheet_downloader.SheetDownloader._save_metadata`
            method.
        """
        if len(features) == 0:
            raise ValueError("[ERROR] No maps to download.")

        try:
            # get url for single tile to estimate size
            tile_url = self.downloader.generate_tile_url(
                features.iloc[0]["grid_bb"].lower_corner, 0
            )

            user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36"
            headers = {"User-Agent": user_agent}
            request = urllib.request.Request(tile_url, None, headers)
            response = urllib.request.urlopen(request)
            # get size of single tile
            size_bytes = len(response.read())
            size_mb = size_bytes * 1e-6

            # get total number of tiles
            no_tiles = sum(
                [feature["grid_bb"].covered_cells for _, feature in features.iterrows()]
            )
            total_size_mb = no_tiles * size_mb

            if total_size_mb > 100:
                print(f"[WARNING] Estimated total size: {total_size_mb * 1e-3:.2f}GB.")
                if not force:
                    raise Warning(
                        f"[WARNING] This will download approximately {total_size_mb * 1e-3:.2f}GB of data. Please confirm download by setting ``force=True``."
                    )
            else:
                print(f"[INFO] Estimated total size: {total_size_mb:.2f}MB.")
        except urllib.error.URLError as e:
            print(f"[WARNING] Unable to estimate download size. {e}")
            if not force:
                raise Warning(
                    "[WARNING] This could download a lot of data. Please confirm download by setting ``force=True``."
                )

        for _, feature in tqdm(features.iterrows()):
            existing_id = self._check_map_sheet_exists(feature, metadata_fname)
            if (
                not overwrite and existing_id is not False
            ):  # if map already exists and overwrite is False then skip
                print(f'[INFO] "{existing_id}" already exists. Skipping download.')
                continue
            img_path = self._download_map(
                feature,
                existing_id,
                download_in_parallel=download_in_parallel,
                overwrite=overwrite,
                error_on_missing_map=error_on_missing_map,
            )
            if img_path is not False:
                metadata_path = f"{path_save}/{metadata_fname}"
                self._save_metadata(
                    feature=feature,
                    out_filepath=metadata_path,
                    img_path=img_path,
                    **kwargs,
                )

    def download_all_map_sheets(
        self,
        path_save: str | None = "maps",
        metadata_fname: str | None = "metadata.csv",
        overwrite: bool | None = False,
        download_in_parallel: bool = True,
        **kwargs: dict | None,
    ) -> None:
        """
        Downloads all map sheets in metadata.

        Parameters
        ----------
        path_save : str, optional
            Path to save map sheets, by default "maps"
        metadata_fname : str, optional
            Name to use for metadata file, by default "metadata.csv"
        overwrite : bool, optional
            Whether to overwrite existing maps, by default ``False``.
        download_in_parallel : bool, optional
            Whether to download tiles in parallel, by default ``True``.
        **kwargs : dict, optional
            Keyword arguments to pass to the
            :meth:`~.download.sheet_downloader.SheetDownloader._download_map_sheets`
            method.
        """
        if "grid_bb" not in self.metadata.columns:
            raise ValueError("[ERROR] Please first run ``get_grid_bb()``")

        self._initialise_downloader()
        self._initialise_merger(path_save)

        self._download_map_sheets(
            self.metadata,
            path_save,
            metadata_fname,
            overwrite,
            download_in_parallel=download_in_parallel,
            **kwargs,
        )

    def download_map_sheets_by_wfs_ids(
        self,
        wfs_ids: list | int,
        path_save: str | None = "maps",
        metadata_fname: str | None = "metadata.csv",
        overwrite: bool | None = False,
        download_in_parallel: bool = True,
        **kwargs: dict | None,
    ) -> None:
        """
        Downloads map sheets by WFS ID numbers.

        Parameters
        ----------
        wfs_ids : Union[list, int]
            The WFS ID numbers of the maps to download.
        path_save : str, optional
            Path to save map sheets, by default "maps"
        metadata_fname : str, optional
            Name to use for metadata file, by default "metadata.csv"
        overwrite : bool, optional
            Whether to overwrite existing maps, by default ``False``.
        download_in_parallel : bool, optional
            Whether to download tiles in parallel, by default ``True``.
        **kwargs : dict, optional
            Keyword arguments to pass to the
            :meth:`~.download.sheet_downloader.SheetDownloader._download_map_sheets`
            method.
        """

        self.query_map_sheets_by_wfs_ids(wfs_ids, append=False, print=False)

        if "grid_bb" not in self.metadata.columns:
            raise ValueError("[ERROR] Please first run ``get_grid_bb()``")

        self._initialise_downloader()
        self._initialise_merger(path_save)

        self._download_map_sheets(
            self.found_queries,
            path_save,
            metadata_fname,
            overwrite,
            download_in_parallel=download_in_parallel,
            **kwargs,
        )

        # reset found_queries
        self.found_queries = gpd.GeoDataFrame()

    def download_map_sheets_by_polygon(
        self,
        polygon: Polygon,
        path_save: str | None = "maps",
        metadata_fname: str | None = "metadata.csv",
        mode: str | None = "within",
        overwrite: bool | None = False,
        download_in_parallel: bool = True,
        **kwargs: dict | None,
    ) -> None:
        """
        Downloads any map sheets which are found within or intersecting with a defined polygon.

        Parameters
        ----------
        polygon : Polygon
            shapely Polygon
        path_save : str, optional
            Path to save map sheets, by default "maps"
        metadata_fname : str, optional
            Name to use for metadata file, by default "metadata.csv"
        mode : str, optional
            The mode to use when finding maps.
            Options of ``"within"``, which returns all map sheets which are completely within the defined polygon,
            and ``"intersects""``, which returns all map sheets which intersect/overlap with the defined polygon.
            By default "within".
        overwrite : bool, optional
            Whether to overwrite existing maps, by default ``False``.
        download_in_parallel : bool, optional
            Whether to download tiles in parallel, by default ``True``.
        **kwargs : dict, optional
            Keyword arguments to pass to the
            :meth:`~.download.sheet_downloader.SheetDownloader._download_map_sheets`
            method.

        Notes
        -----
        Use ``create_polygon_from_latlons()`` to create polygon.
        """
        self.query_map_sheets_by_polygon(polygon, mode=mode, append=False, print=False)

        if "grid_bb" not in self.metadata.columns:
            raise ValueError("[ERROR] Please first run ``get_grid_bb()``")

        self._initialise_downloader()
        self._initialise_merger(path_save)

        self._download_map_sheets(
            self.found_queries,
            path_save,
            metadata_fname,
            overwrite,
            download_in_parallel=download_in_parallel,
            **kwargs,
        )

        # reset found_queries
        self.found_queries = gpd.GeoDataFrame()

    def download_map_sheets_by_coordinates(
        self,
        coords: tuple,
        path_save: str | None = "maps",
        metadata_fname: str | None = "metadata.csv",
        overwrite: bool | None = False,
        download_in_parallel: bool = True,
        **kwargs: dict | None,
    ) -> None:
        """
        Downloads any maps sheets which contain a defined set of coordinates.
        Coordinates are (x,y).

        Parameters
        ----------
        coords : tuple
            Coordinates in ``(x,y)`` format.
        path_save : str, optional
            Path to save map sheets, by default "maps"
        metadata_fname : str, optional
            Name to use for metadata file, by default "metadata.csv"
        overwrite : bool, optional
            Whether to overwrite existing maps, by default ``False``.
        download_in_parallel : bool, optional
            Whether to download tiles in parallel, by default ``True``.
        **kwargs : dict, optional
            Keyword arguments to pass to the
            :meth:`~.download.sheet_downloader.SheetDownloader._download_map_sheets`
            method.
        """
        self.query_map_sheets_by_coordinates(coords, append=False, print=False)

        if "grid_bb" not in self.metadata.columns:
            raise ValueError("[ERROR] Please first run ``get_grid_bb()``")

        self._initialise_downloader()
        self._initialise_merger(path_save)

        self._download_map_sheets(
            self.found_queries,
            path_save,
            metadata_fname,
            overwrite,
            download_in_parallel=download_in_parallel,
            **kwargs,
        )

        # reset found_queries
        self.found_queries = gpd.GeoDataFrame()

    def download_map_sheets_by_line(
        self,
        line: LineString,
        path_save: str | None = "maps",
        metadata_fname: str | None = "metadata.csv",
        overwrite: bool | None = False,
        download_in_parallel: bool = True,
        **kwargs: dict | None,
    ) -> None:
        """
        Downloads any maps sheets which intersect with a line.

        Parameters
        ----------
        line : LineString
            shapely LineString
        path_save : str, optional
            Path to save map sheets, by default "maps"
        metadata_fname : str, optional
            Name to use for metadata file, by default "metadata.csv"
        overwrite : bool, optional
            Whether to overwrite existing maps, by default ``False``
        download_in_parallel : bool, optional
            Whether to download tiles in parallel, by default ``True``.
        **kwargs : dict, optional
            Keyword arguments to pass to the
            :meth:`~.download.sheet_downloader.SheetDownloader._download_map_sheets`
            method.

        Notes
        -----
        Use ``create_line_from_latlons()`` to create line.
        """
        self.query_map_sheets_by_line(line, append=False, print=False)

        if "grid_bb" not in self.metadata.columns:
            raise ValueError("[ERROR] Please first run ``get_grid_bb()``")

        self._initialise_downloader()
        self._initialise_merger(path_save)

        self._download_map_sheets(
            self.found_queries,
            path_save,
            metadata_fname,
            overwrite,
            download_in_parallel=download_in_parallel,
            **kwargs,
        )

        # reset found_queries
        self.found_queries = gpd.GeoDataFrame()

    def download_map_sheets_by_string(
        self,
        string: str,
        columns: str | list | None = None,
        path_save: str | None = "maps",
        metadata_fname: str | None = "metadata.csv",
        overwrite: bool | None = False,
        download_in_parallel: bool = True,
        **kwargs: dict | None,
    ) -> None:
        """
        Download map sheets by searching for a string in the metadata.

        Parameters
        ----------
        string : str
            The string to search for.
            Can be raw string and use regular expressions.
        columns : str or list or None, optional
            A column or list of columns to search in.
            If ``None``, will search in all metadata fields. By default ``None``.
        path_save : str, optional
            Path to save map sheets, by default "maps"
        metadata_fname : str, optional
            Name to use for metadata file, by default "metadata.csv"
        overwrite : bool, optional
            Whether to overwrite existing maps, by default ``False``.
        download_in_parallel : bool, optional
            Whether to download tiles in parallel, by default ``True``.
        **kwargs : dict, optional
            Keyword arguments to pass to the
            :meth:`~.download.sheet_downloader.SheetDownloader._download_map_sheets`
            method.

        Notes
        -----
        ``string`` is case insensitive.
        """
        self.query_map_sheets_by_string(
            string, columns=columns, append=False, print=False
        )

        if "grid_bb" not in self.metadata.columns:
            raise ValueError("[ERROR] Please first run ``get_grid_bb()``")

        self._initialise_downloader()
        self._initialise_merger(path_save)

        self._download_map_sheets(
            self.found_queries,
            path_save,
            metadata_fname,
            overwrite,
            download_in_parallel=download_in_parallel,
            **kwargs,
        )

        # reset found_queries
        self.found_queries = gpd.GeoDataFrame()

    def download_map_sheets_by_queries(
        self,
        path_save: str | None = "maps",
        metadata_fname: str | None = "metadata.csv",
        overwrite: bool | None = False,
        download_in_parallel: bool = True,
        **kwargs: dict | None,
    ) -> None:
        """
        Downloads map sheets saved as query results.

        Parameters
        ----------
        path_save : str, optional
            Path to save map sheets, by default "maps"
        metadata_fname : str, optional
            Name to use for metadata file, by default "metadata.csv"
        overwrite : bool, optional
            Whether to overwrite existing maps, by default ``False``.
        download_in_parallel : bool, optional
            Whether to download tiles in parallel, by default ``True``.
        **kwargs : dict, optional
            Keyword arguments to pass to the
            :meth:`~.download.sheet_downloader.SheetDownloader._download_map_sheets`
            method.
        """
        if "grid_bb" not in self.metadata.columns:
            raise ValueError("[ERROR] Please first run ``get_grid_bb()``")

        self._initialise_downloader()
        self._initialise_merger(path_save)

        features = self.metadata[
            self.metadata.index.isin(self.found_queries.index)
        ]  # to update query data e.g. if we have added grid_bb

        self._download_map_sheets(
            features,
            path_save,
            metadata_fname,
            overwrite,
            download_in_parallel=download_in_parallel,
            **kwargs,
        )

    def plot_features_on_map(
        self,
        features: gpd.GeoDataFrame,
        map_extent: str | (list | tuple) | None = None,
        add_id: bool = False,
    ) -> None:
        """
        Plot boundaries of map sheets on a map using cartopy.

        Parameters
        ----------
        map_extent : Union[str, list, tuple, None], optional
            The extent of the underlying map to be plotted.

            If a tuple or list, must be of the format ``[lon_min, lon_max, lat_min, lat_max]``.
            If a string, only ``"uk"``, ``"UK"`` or ``"United Kingdom"`` are accepted and will limit the map extent to the UK's boundaries.
            If None, the map extent will be set automatically.
            By default None.
        add_id : bool, optional
            Whether to add an ID (WFS ID number) to each map sheet, by default False.
        """
        plt.figure(figsize=[15, 15])

        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines(resolution="10m", color="black", linewidth=1)

        if isinstance(map_extent, str):
            if map_extent in ["uk", "UK", "United Kingdom"]:
                extent = [-8.08999993, 1.81388127, 49.8338702, 60.95000002]
                ax.set_extent(extent)
            else:
                raise NotImplementedError(
                    "[ERROR] Currently only UK is implemented. \
Try passing coordinates (min_x, max_x, min_y, max_y) instead or leave blank to auto-set map extent."
                )
        elif isinstance(map_extent, (list, tuple)):
            ax.set_extent(map_extent)
        else:
            pass

        features.plot(ax=ax, facecolor="none", edgecolor="r")
        if add_id:
            if "wfs_id_no" not in self.metadata.columns:
                self.extract_wfs_id_nos()
            features.apply(
                lambda x: ax.text(
                    x.geometry.centroid.x,
                    x.geometry.centroid.y,
                    x["wfs_id_no"],
                    color="r",
                    horizontalalignment="center",  # center text
                    verticalalignment="center",  # center text
                ),
                axis=1,
            )

        plt.show()

    def plot_all_metadata_on_map(
        self,
        map_extent: str | (list | tuple) | None = None,
        add_id: bool = False,
    ) -> None:
        """
        Plots boundaries of all map sheets in metadata on a map using cartopy.

        Parameters
        ----------
        map_extent : Union[str, list, tuple, None], optional
            The extent of the underlying map to be plotted.

            If a tuple or list, must be of the format ``[lon_min, lon_max, lat_min, lat_max]``.
            If a string, only ``"uk"``, ``"UK"`` or ``"United Kingdom"`` are accepted and will limit the map extent to the UK's boundaries.
            If None, the map extent will be set automatically.
            By default None.
        add_id : bool, optional
            Whether to add an ID (WFS ID number) to each map sheet, by default False.
        """

        self.plot_features_on_map(self.metadata, map_extent, add_id)

    def plot_queries_on_map(
        self,
        map_extent: str | (list | tuple) | None = None,
        add_id: bool = False,
    ) -> None:
        """
        Plots boundaries of query results on a map using cartopy.

        Parameters
        ----------
        map_extent : Union[str, list, tuple, None], optional
            The extent of the underlying map to be plotted.

            If a tuple or list, must be of the format ``[lon_min, lon_max, lat_min, lat_max]``.
            If a string, only ``"uk"``, ``"UK"`` or ``"United Kingdom"`` are accepted and will limit the map extent to the UK's boundaries.
            If None, the map extent will be set automatically.
            By default None.
        add_id : bool, optional
            Whether to add an ID (WFS ID number) to each map sheet, by default False.
        """

        self.plot_features_on_map(self.found_queries, map_extent, add_id)

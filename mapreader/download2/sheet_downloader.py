import json
import os
from typing import Union
from shapely.geometry import Polygon, LineString, Point, shape
from shapely.ops import unary_union
from .data_structures import Coordinate, GridBoundingBox
from .tile_loading import TileDownloader
from .tile_merging import TileMerger
from .downloader_utils import get_index_from_coordinate
import re
import matplotlib.pyplot as plt
# import cartopy.crs as ccrs - would be good to get this fixed (i think by conda package)
import numpy as np
import pandas as pd
import shutil


class SheetDownloader:
    """
    A class to download map sheets using metadata.
    """

    def __init__(
        self,
        metadata_path: str,
        download_url: Union[str, list],
    ) -> None:
        """
        Iniitalise SheetDownloader class

        Parameters
        ----------
        metadata_path : str
            path to metadata.json
        download_url : Union[str, list]
            The base URL pattern used to download tiles from the server. This
            should contain placeholders for the x coordinate (``x``), the y
            coordinate (``y``) and the zoom level (``z``).
        """
        self.polygons = False
        self.grid_bbs = False
        self.wfs_id_nos = False
        self.published_dates = False
        self.found_queries = []
        self.merged_polygon = None

        assert isinstance(
            metadata_path, str
        ), "[ERROR] Please pass metadata_path as string."

        if os.path.isfile(metadata_path):
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
                self.features = self.metadata["features"]
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

    def __str__(self) -> str:
        info = f"[INFO] Metadata file has {self.__len__()} item(s)."
        return info

    def __len__(self) -> int:
        return len(self.features)

    def get_polygons(self) -> None:
        """
        For each map in metadata, creates a polygon from map geometry and saves to ``features`` dictionary.
        """
        for i, feature in enumerate(self.features):
            polygon = shape(feature["geometry"])
            map_name = feature["properties"]["IMAGE"]
            if len(polygon.geoms) != 1:
                f"[WARNING] Multiple geometries found in map {map_name}. Using first instance."
            feature["polygon"] = polygon.geoms[0]

        self.polygons = True

    def get_grid_bb(self, zoom_level: int = 14) -> None:
        """
        For each map in metadata, creates a grid bounding box from map polygons and saves to ``features`` dictionary.

        Parameters
        ----------
        zoom_level : int, optional
            The zoom level to use when creating the grid bounding box.
            Later used when downloading maps, by default 14.
        """
        if not self.polygons:
            self.get_polygons()

        for feature in self.features:
            polygon = feature["polygon"]
            min_x, min_y, max_x, max_y = polygon.bounds

            start = Coordinate(min_y, max_x)  # (lat, lon)
            end = Coordinate(max_y, min_x)  # (lat, lon)

            start_idx = get_index_from_coordinate(start, zoom_level)
            end_idx = get_index_from_coordinate(end, zoom_level)
            grid_bb = GridBoundingBox(start_idx, end_idx)

            feature["grid_bb"] = grid_bb

        self.grid_bbs = True

    def extract_wfs_id_nos(self) -> None:
        """
        For each map in metadata, extracts WFS ID numbers from WFS information and saves to ``features`` dictionary.
        """
        for feature in self.features:
            wfs_id = feature["id"]
            wfs_id_no = wfs_id.split(sep=".")[-1]

            feature["wfs_id_no"] = eval(wfs_id_no)

        self.wfs_id_nos = True

    def extract_published_dates(self) -> None:
        """
        For each map in metadata, extracts publication date from WFS title and saves to ``features`` dictionary.
        """
        for feature in self.features:
            wfs_title = feature["properties"]["WFS_TITLE"]
            published_date = re.findall(
                r"Published.*[\D]([\d]+)", wfs_title, flags=re.IGNORECASE
            )
            if len(published_date) > 0:
                feature["properties"]["published_date"] = eval(published_date[0])
                if len(published_date) != 1:
                    map_name = feature["properties"]["IMAGE"]
                    print(
                        f"[WARNING] Multiple published dates detected in {map_name}. Using first date."
                    )
            else:
                feature["properties"]["published_date"] = []
                map_name = feature["properties"]["IMAGE"]
                print(f"[WARNING] No published date detected in {map_name}.")

        self.published_dates = True

    def get_merged_polygon(self) -> None:
        """
        Creates a multipolygon representing all maps in metadata.
        """

        if not self.polygons:
            self.get_polygons()

        polygon_list = [feature["polygon"] for feature in self.features]

        merged_polygon = unary_union(polygon_list)
        self.merged_polygon = merged_polygon

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
        self.merger = TileMerger(output_folder=path_save, show_progress=False)
    
    def _check_map_sheet_exists(self, feature: dict) -> bool:
        """
        Checks if a map sheet is already saved.

        Parameters
        ----------
        feature : dict

        Returns
        -------
        bool
            True if file exists, False if not.
        """
        map_name = str("map_" + feature["properties"]["IMAGE"])        
        path_save = self.merger.output_folder
        if os.path.exists(f"{path_save}{map_name}.png"):
            print(f'[INFO] "{path_save}{map_name}.png" already exists. Skipping download.')
            return True
        return False

    def _download_map(self, feature: dict) -> bool:
        """
        Downloads a single map sheet and saves as png file.

        Parameters
        ----------
        feature : dict

        Returns
        -------
        bool
            True if map was downloaded sucessfully, False if not.
        """
        map_name = str("map_" + feature["properties"]["IMAGE"])
        self.downloader.download_tiles(feature["grid_bb"])
        success = self.merger.merge(feature["grid_bb"], map_name)
        if success:
            print(f'[INFO] Downloaded "{map_name}.png"')
        else:
            print(f'[WARNING] Download of "{map_name}.png" was unsuccessfull.')
        
        shutil.rmtree("_tile_cache/")
        return success

    def _save_metadata(
        self, 
        feature: dict
    ) -> list:
        """
        Creates list of selected metadata items.

        Parameters
        ----------
        feature : dict

        Returns
        -------
        list
            List of selected metadata (to be saved)
        """

        map_name = str("map_" + feature["properties"]["IMAGE"] + ".png")
        map_url = str(feature["properties"]["IMAGEURL"])
        coords = feature["geometry"]["coordinates"][0][0]

        if not self.published_dates:
            self.extract_published_dates()

        published_date = feature["properties"]["published_date"]
        grid_bb = feature["grid_bb"]

        return [map_name, map_url, coords, published_date, grid_bb]

    def _create_metadata_df(self, metadata_to_save: list, out_filepath) -> None:
        """
        Creates metadata datframe from list of ``metadata_to_save`` and saves as csv.

        Parameters
        ----------
        metadata_to_save : list
            List of metadata to save
        out_filepath : _type_
            File path where metadata csv will be saved.
        """
        metadata_df = pd.DataFrame(
            metadata_to_save,
            columns=["name", "url", "coordinates", "published_date", "grid_bb"],
        )
        exists = True if os.path.exists(out_filepath) else False
        metadata_df.to_csv(out_filepath, sep="|", mode="a", header=not exists)

    def query_map_sheets_by_wfs_ids(
        self,
        wfs_ids: Union[list, int],
        append: bool = False,
        print: bool = False,
    ) -> None:
        """
        Find map sheets by WFS ID numbers.

        Parameters
        ----------
        wfs_ids : Union[list, int]
            The WFS ID numbers of the maps to download.
        append : bool, optional
            Whether to append to current query results list or, if False, start a new list. 
            By default False
        print: bool, optional
            Whether to print query results or not.
            By default False
        """
        if not self.wfs_id_nos:
            self.extract_wfs_id_nos()

        if isinstance(wfs_ids, list):
            requested_maps = wfs_ids
        elif isinstance(wfs_ids, int):
            requested_maps = [wfs_ids]
        else:
            raise ValueError(
                "[ERROR] Please pass ``wfs_ids`` as int or list of ints."
            )

        if not append:
            self.found_queries = []  # reset each time

        for feature in self.features:
            wfs_id_no = feature["wfs_id_no"]

            if wfs_id_no in requested_maps:
                if feature not in self.found_queries: #only append if new item
                    self.found_queries.append(feature)

        if print:
            self.print_found_queries()

    def query_map_sheets_by_polygon(
        self, polygon: Polygon, mode: str = "within", append: bool = False, print: bool = False,
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
            Whether to append to current query results list or, if False, start a new list. 
            By default False
        print: bool, optional
            Whether to print query results or not.
            By default False

        Notes
        -----
        Use ``create_polygon_from_latlons()`` to create polygon.
        """
        assert isinstance(
            polygon, Polygon
        ), "[ERROR] Please pass polygon as shapely.geometry.Polygon object.\n\
[HINT] Use ``create_polygon_from_latlons()`` to create polygon."

        assert mode in [
            "within",
            "intersects",
        ], '[ERROR] Please use ``mode="within"`` or ``mode="intersects"``.'

        if not self.polygons:
            self.get_polygons()

        if not append:
            self.found_queries = []  # reset each time

        for feature in self.features:
            map_polygon = feature["polygon"]

            if mode == "within":
                if map_polygon.within(polygon):
                    if map_polygon not in self.found_queries: #only append if new item
                        self.found_queries.append(feature)
            elif mode == "intersects":
                if map_polygon.intersects(polygon):
                    if feature not in self.found_queries: #only append if new item
                        self.found_queries.append(feature)

        if print:
            self.print_found_queries()

    def query_map_sheets_by_coordinates(
        self, coords: tuple, append: bool = False, print: bool = False
    ) -> None:
        """
        Find maps sheets which contain a defined set of coordinates.
        Coordinates are (x,y).

        Parameters
        ----------
        coords : tuple
            Coordinates in ``(x,y)`` format.
        append : bool, optional
            Whether to append to current query results list or, if False, start a new list. 
            By default False
        print: bool, optional
            Whether to print query results or not.
            By default False
        """
        assert isinstance(
            coords, tuple
        ), "[ERROR] Please pass coords as a tuple in the form (x,y)."

        coords = Point(coords)

        if not self.polygons:
            self.get_polygons()

        if not append:
            self.found_queries = []  # reset each time

        for feature in self.features:
            map_polygon = feature["polygon"]

            if map_polygon.contains(coords):
                if feature not in self.found_queries: #only append if new item
                    self.found_queries.append(feature)

        if print:
            self.print_found_queries()

    def query_map_sheets_by_line(
        self, line: LineString, append: bool = False, print: bool = False
    ) -> None:
        """
        Find maps sheets which intersect with a line.

        Parameters
        ----------
        line : LineString
            shapely LineString
        append : bool, optional
            Whether to append to current query results list or, if False, start a new list. 
            By default False
        print: bool, optional
            Whether to print query results or not.
            By default False
        
        Notes
        -----
        Use ``create_line_from_latlons()`` to create line.
        """

        assert isinstance(
            line, LineString
        ), "[ERROR] Please pass line as shapely.geometry.LineString object.\n\
[HINT] Use ``create_line_from_latlons()`` to create line."

        if not self.polygons:
            self.get_polygons()

        if not append:
            self.found_queries = []  # reset each time

        for feature in self.features:
            map_polygon = feature["polygon"]

            if map_polygon.intersects(line):
                if feature not in self.found_queries: #only append if new item
                    self.found_queries.append(feature)

        if print:
            self.print_found_queries()

    def print_found_queries(self) -> None:
        """
        Prints query results.
        """
        if not self.polygons:
            self.get_polygons()

        if len(self.found_queries) == 0:
            print("[INFO] No query results found/saved.")
        else:
            divider = 14 * "="
            print(f"{divider}\nQuery results:\n{divider}")
            for feature in self.found_queries:
                map_url = feature["properties"]["IMAGEURL"]
                map_bounds = feature["polygon"].bounds
                print(f"URL:     \t{map_url}")
                print(f"coordinates:  \t{map_bounds}")
                print(20 * "-")

    def download_all_map_sheets(
        self, path_save: str = "./maps/", metadata_fname="metadata.csv"
    ) -> None:
        """
        Downloads all map sheets in metadata.

        Parameters
        ----------
        path_save : str, optional
            Path to save map sheets, by default "./maps/"
        metadata_fname : str, optional
            Name to use for metadata file, by default "metadata.csv"
        """
        if not self.grid_bbs:
            raise ValueError("[ERROR] Please first run ``get_grid_bb()``")

        self._initialise_downloader()
        self._initialise_merger(path_save)

        metadata_to_save = []
        for feature in self.features:
            exists = self._check_map_sheet_exists(feature)
            if not exists:
                success = self._download_map(feature)
                if success:
                    metadata_to_save.append(self._save_metadata(feature))

        metadata_path = "{}{}".format(path_save, metadata_fname)
        self._create_metadata_df(metadata_to_save, metadata_path)

    def download_map_sheets_by_wfs_ids(
        self,
        wfs_ids: Union[list, int],
        path_save: str = "./maps/",
        metadata_fname="metadata.csv",
    ) -> None:
        """
        Downloads map sheets by WFS ID numbers.

        Parameters
        ----------
        wfs_ids : Union[list, int]
            The WFS ID numbers of the maps to download.
        path_save : str, optional
            Path to save map sheets, by default "./maps/"
        metadata_fname : str, optional
            Name to use for metadata file, by default "metadata.csv"
        """

        if not self.wfs_id_nos:
            self.extract_wfs_id_nos()

        if isinstance(wfs_ids, list):
            requested_maps = wfs_ids
        elif isinstance(wfs_ids, int):
            requested_maps = [wfs_ids]
        else:
            raise ValueError(
                "[ERROR] Please pass ``wfs_ids`` as int or list of ints."
            )

        if not self.grid_bbs:
            raise ValueError("[ERROR] Please first run ``get_grid_bb()``")

        self._initialise_downloader()
        self._initialise_merger(path_save)

        wfs_id_list = [feature["wfs_id_no"] for feature in self.features]
        if set(wfs_id_list).isdisjoint(set(requested_maps)):
            raise ValueError("[ERROR] No map sheets with given WFS ID numbers found.")

        metadata_to_save = []
        for feature in self.features:
            wfs_id_no = feature["wfs_id_no"]
            if wfs_id_no in requested_maps:
                exists = self._check_map_sheet_exists(feature)
                if not exists:
                    success = self._download_map(feature)
                    if success:
                        metadata_to_save.append(self._save_metadata(feature))

        metadata_path = "{}{}".format(path_save, metadata_fname)
        self._create_metadata_df(metadata_to_save, metadata_path)

    def download_map_sheets_by_polygon(
        self,
        polygon: Polygon,
        path_save: str = "./maps/",
        metadata_fname="metadata.csv",
        mode: str = "within",
    ) -> None:
        """
        Donwloads any map sheets which are found within or intersecting with a defined polygon.

        Parameters
        ----------
        polygon : Polygon
            shapely Polygon
        path_save : str, optional
            Path to save map sheets, by default "./maps/"
        metadata_fname : str, optional
            Name to use for metadata file, by default "metadata.csv"
        mode : str, optional
            The mode to use when finding maps.
            Options of ``"within"``, which returns all map sheets which are completely within the defined polygon, 
            and ``"intersects""``, which returns all map sheets which intersect/overlap with the defined polygon.
            By default "within".

        Notes
        -----
        Use ``create_polygon_from_latlons()`` to create polygon.
        """
        assert isinstance(
            polygon, Polygon
        ), "[ERROR] Please pass polygon as shapely.geometry.Polygon object.\n\
[HINT] Use ``create_polygon_from_latlons()`` to create polygon."

        assert mode in [
            "within",
            "intersects",
        ], '[ERROR] Please use ``mode="within"`` or ``mode="intersects"``.'

        if not self.grid_bbs:
            raise ValueError("[ERROR] Please first run ``get_grid_bb()``")

        self._initialise_downloader()
        self._initialise_merger(path_save)

        if self.merged_polygon is None:
            self.get_merged_polygon()

        if self.merged_polygon.disjoint(polygon):
            raise ValueError(f"[ERROR] Polygon is out of map metadata bounds.")

        metadata_to_save = []
        for feature in self.features:
            requested = False
            map_polygon = feature["polygon"]

            if mode == "within":
                if map_polygon.within(polygon):
                    requested = True
            elif mode == "intersects":
                if map_polygon.intersects(polygon):
                    requested = True

            if requested == True:
                exists = self._check_map_sheet_exists(feature)
                if not exists:
                    success = self._download_map(feature)
                    if success:
                        metadata_to_save.append(self._save_metadata(feature))

        metadata_path = "{}{}".format(path_save, metadata_fname)
        self._create_metadata_df(metadata_to_save, metadata_path)

    def download_map_sheets_by_coordinates(
        self, coords: tuple, path_save: str = "./maps/", metadata_fname="metadata.csv"
    ) -> None:
        """
        Downloads any maps sheets which contain a defined set of coordinates.
        Coordinates are (x,y).

        Parameters
        ----------
        coords : tuple
            Coordinates in ``(x,y)`` format.
        path_save : str, optional
            Path to save map sheets, by default "./maps/"
        metadata_fname : str, optional
            Name to use for metadata file, by default "metadata.csv"
        """

        assert isinstance(
            coords, tuple
        ), "[ERROR] Please pass coords as a tuple in the form (x,y)."

        coords = Point(coords)

        if not self.grid_bbs:
            raise ValueError("[ERROR] Please first run ``get_grid_bb()``")

        self._initialise_downloader()
        self._initialise_merger(path_save)

        if self.merged_polygon is None:
            self.get_merged_polygon()

        if self.merged_polygon.disjoint(coords):
            raise ValueError(f"[ERROR] Coordinates are out of map metadata bounds.")

        metadata_to_save = []
        for feature in self.features:
            map_polygon = feature["polygon"]

            if map_polygon.contains(coords):
                exists = self._check_map_sheet_exists(feature)
                if not exists:
                    success = self._download_map(feature)
                    if success:
                        metadata_to_save.append(self._save_metadata(feature))

        metadata_path = "{}{}".format(path_save, metadata_fname)
        self._create_metadata_df(metadata_to_save, metadata_path)

    def download_map_sheets_by_line(
        self, line: LineString, path_save: str = "./maps/", metadata_fname="metadata.csv"
    ) -> None:
        """
        Downloads any maps sheets which intersect with a line.

        Parameters
        ----------
        line : LineString
            shapely LineString
        path_save : str, optional
            Path to save map sheets, by default "./maps/"
        metadata_fname : str, optional
            Name to use for metadata file, by default "metadata.csv"

        Notes
        -----
        Use ``create_line_from_latlons()`` to create line.
        """

        assert isinstance(
            line, LineString
        ), "[ERROR] Please pass line as shapely.geometry.LineString object.\n\
[HINT] Use ``create_line_from_latlons()`` to create line."

        if not self.grid_bbs:
            raise ValueError("[ERROR] Please first run ``get_grid_bb()``")

        self._initialise_downloader()
        self._initialise_merger(path_save)

        if self.merged_polygon is None:
            self.get_merged_polygon()

        if self.merged_polygon.disjoint(line):
            raise ValueError(f"[ERROR] Line is out of map metadata bounds.")

        metadata_to_save = []
        for feature in self.features:
            map_polygon = feature["polygon"]

            if map_polygon.intersects(line):
                exists = self._check_map_sheet_exists(feature)
                if not exists:
                    success = self._download_map(feature)
                    if success:
                        metadata_to_save.append(self._save_metadata(feature))

        metadata_path = "{}{}".format(path_save, metadata_fname)
        self._create_metadata_df(metadata_to_save, metadata_path)        

    def download_map_sheets_by_queries(
        self,
        path_save: str = "./maps/",
        metadata_fname="metadata.csv",
    ) -> None:
        """
        Downloads map sheets saved as query results.

        Parameters
        ----------
        path_save : str, optional
            Path to save map sheets, by default "./maps/"
        metadata_fname : str, optional
            Name to use for metadata file, by default "metadata.csv"
        """
        if not self.grid_bbs:
            raise ValueError("[ERROR] Please first run ``get_grid_bb()``")

        self._initialise_downloader()
        self._initialise_merger(path_save)

        assert len(self.found_queries) > 0, "[ERROR] No query results found/saved."

        metadata_to_save = []
        for feature in self.found_queries:
            exists = self._check_map_sheet_exists(feature)
            if not exists:
                success = self._download_map(feature)
                if success:
                    metadata_to_save.append(self._save_metadata(feature))

        metadata_path = "{}{}".format(path_save, metadata_fname)
        self._create_metadata_df(metadata_to_save, metadata_path)

    def hist_published_dates(self, **kwargs) -> None:
        """
        Plots a histogram of the publication dates of maps in metadata.

        Parameters
        ----------
        kwargs : A dictionary containing keyword arguments to pass to plotting function.
            See matplotlib.pyplot.hist() for acceptable values.

            e.g. ``**dict(fc='c', ec='k')``

        Notes
        -----
        bins and range already set when plotting so are invalid kwargs.
        """
        if not self.published_dates:
            self.extract_published_dates()

        published_dates = [
            feature["properties"]["published_date"] for feature in self.features
        ]
        min_date = min(published_dates)
        max_date = max(published_dates)
        date_range = max_date - min_date
        print(min_date, max_date, date_range)

        plt.hist(published_dates, bins=date_range, range=(min_date, max_date), **kwargs)
        plt.locator_params(integer=True)
        plt.xticks(size=14)
        plt.yticks(size=14)
        plt.xlabel("Published date", size=18)
        plt.ylabel("Counts", size=18)
        plt.show()

    def plot_features_on_map(
        self,
        features: list,
        map_extent: Union[str, list, tuple, None] = None,
        add_id: bool = True,
    ) -> None:
        """
        Plots boundaries of map sheets on a map using ``cartopy`` library, (if available).

        Parameters
        ----------
        map_extent : Union[str, list, tuple, None], optional
            The extent of the underlying map to be plotted.
            
            If a tuple or list, must be of the format ``[lon_min, lon_max, lat_min, lat_max]``. 
            If a string, only ``"uk"``, ``"UK"`` or ``"United Kingdom"`` are accepted and will limit the map extent to the UK's boundaries.
            If None, the map extent will be set automatically. 
            By default None.
        add_id : bool, optional
            Whether to add an ID (WFS ID number) to each map sheet, by default True.
        """
        
        if add_id:
            if not self.wfs_id_nos:
                self.extract_wfs_id_nos()

        plt.figure(figsize=[15, 15])

        try:
            import cartopy.crs as ccrs
            
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.coastlines(resolution="10m", color="black", linewidth=1)

            if isinstance(map_extent, str):
                if map_extent in ["uk", "UK", "United Kingdom"]:
                    extent = [-8.08999993, 1.81388127, 49.8338702, 60.95000002]
                    ax.set_extent(extent)
                else:
                    raise NotImplementedError(
                        '[ERROR] Currently only UK is implemented. \
Try passing coordinates (min_x, max_x, min_y, max_y) instead or leave blank to auto-set map extent.'
                    )
            elif isinstance(map_extent, (list, tuple)):
                ax.set_extent(map_extent)
            else:
                pass

            for feature in features:
                text_id = feature["wfs_id_no"]
                coords = np.array(feature["geometry"]["coordinates"][0][0])

                # Plot coordinates
                plt.plot(
                    coords[:, 0],
                    coords[:, 1],
                    c="r",
                    linewidth=0.5,
                    transform=ccrs.Geodetic(),
                )

                if add_id:
                    plt.text(
                        np.mean(coords[:, 0]) - 0.15,
                        np.mean(coords[:, 1]) - 0.05,
                        f"{text_id}",
                        color="r",
                    )
                
        except:
            print("[WARNING] Cartopy is not installed. \
If you would like to install it, please follow instructions at https://scitools.org.uk/cartopy/docs/latest/installing.html")

            ax = plt.axes()

            for feature in features:
                text_id = feature["wfs_id_no"]
                coords = np.array(feature["geometry"]["coordinates"][0][0])

                plt.plot(coords[:, 0], coords[:, 1], c="r", alpha=0.5)

                if add_id:
                    plt.text(
                        np.mean(coords[:, 0]) - 0.15,
                        np.mean(coords[:, 1]) - 0.05,
                        f"{text_id}",
                        color="r",
                    )

        plt.show()

    def plot_all_metadata_on_map(
        self,
        map_extent: Union[str, list, tuple, None] = None,
        add_id: bool = True,
    ) -> None:
        """
        Plots boundaries of all map sheets in metadata on a map using ``cartopy`` library (if available).

        Parameters
        ----------
        map_extent : Union[str, list, tuple, None], optional
            The extent of the underlying map to be plotted.
            
            If a tuple or list, must be of the format ``[lon_min, lon_max, lat_min, lat_max]``. 
            If a string, only ``"uk"``, ``"UK"`` or ``"United Kingdom"`` are accepted and will limit the map extent to the UK's boundaries.
            If None, the map extent will be set automatically. 
            By default None.
        add_id : bool, optional
            Whether to add an ID (WFS ID number) to each map sheet, by default True.
        """

        features_to_plot=self.features
        self.plot_features_on_map(features_to_plot, map_extent, add_id)

    def plot_queries_on_map(
        self,
        map_extent: Union[str, list, tuple, None] = None,
        add_id: bool = True,
    ) -> None:
        """
        Plots boundaries of query results on a map using ``cartopy`` library (if available).

        Parameters
        ----------
        map_extent : Union[str, list, tuple, None], optional
            The extent of the underlying map to be plotted.
            
            If a tuple or list, must be of the format ``[lon_min, lon_max, lat_min, lat_max]``. 
            If a string, only ``"uk"``, ``"UK"`` or ``"United Kingdom"`` are accepted and will limit the map extent to the UK's boundaries.
            If None, the map extent will be set automatically. 
            By default None.
        add_id : bool, optional
            Whether to add an ID (WFS ID number) to each map sheet, by default True.
        """

        features_to_plot=self.found_queries
        self.plot_features_on_map(features_to_plot, map_extent, add_id)


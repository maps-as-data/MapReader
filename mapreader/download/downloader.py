from __future__ import annotations

import os
import shutil

from shapely.geometry import Polygon

from .data_structures import Coordinate, GridBoundingBox
from .downloader_utils import get_index_from_coordinate
from .tile_loading import DEFAULT_TEMP_FOLDER, TileDownloader
from .tile_merging import TileMerger


class Downloader:
    """
    A class to download maps (without using metadata)
    """

    def __init__(
        self,
        download_url: str | list,
    ):
        """Initialise Downloader object.

        Parameters
        ----------
        download_url : Union[str, list]
            The base URL pattern used to download tiles from the server. This
            should contain placeholders for the x coordinate (``x``), the y
            coordinate (``y``) and the zoom level (``z``).
        """
        if isinstance(download_url, str):
            my_ts = [download_url]
        elif isinstance(download_url, list):
            my_ts = download_url
        else:
            raise ValueError(
                "[ERROR] Please pass ``download_url`` as string or list of strings."
            )

        self.download_url = my_ts

    def __str__(self) -> str:
        info = f"[INFO] Downloading from {self.download_url}."
        return info

    def _initialise_downloader(self):
        """
        Initialise TileDownloader object.
        """
        self.downloader = TileDownloader(self.download_url)

    def _initialise_merger(self, path_save: str):
        """
        Initialise TileMerger object.

        Parameters
        ----------
        path_save : str
            Path to save merged items (i.e. whole map sheets)
        """
        self.merger = TileMerger(output_folder=path_save, show_progress=False)

    def _check_map_exists(self, grid_bb: GridBoundingBox, map_name: str | None) -> bool:
        """
        Checks if a map is already saved.

        Parameters
        ----------
        grid_bb : GridBoundingBox
            The grid bounding box of the map
        map_name : str, optional
            Name to use when saving the map, by default None

        Returns
        -------
        bool
            True if file exists, False if not.
        """
        if map_name is None:
            map_name = self.merger._get_output_name(grid_bb)
        path_save = self.merger.output_folder
        if os.path.exists(f"{path_save}{map_name}.png"):
            print(
                f'[INFO] "{path_save}{map_name}.png" already exists. Skipping download.'
            )
            return True
        return False

    def _download_map(self, grid_bb: GridBoundingBox, map_name: str | None) -> bool:
        """
        Downloads a map contained within a grid bounding box.

        Parameters
        ----------
        grid_bb : GridBoundingBox
            The grid bounding box of the map
        map_name : str, optional
            Name to use when saving the map, by default None

        Returns
        -------
        bool
            True if map was successfully downloaded, False if not.
        """
        if map_name is None:
            map_name = self.merger._get_output_name(grid_bb)
        self.downloader.download_tiles(grid_bb, download_in_parallel=False)
        success = self.merger.merge(grid_bb, map_name)
        if success:
            print(f'[INFO] Downloaded "{map_name}.png"')
        else:
            print(f'[WARNING] Download of "{map_name}.png" was unsuccessful.')

        shutil.rmtree(DEFAULT_TEMP_FOLDER)
        return success

    def download_map_by_polygon(
        self,
        polygon: Polygon,
        zoom_level: int | None = 14,
        path_save: str | None = "maps",
        overwrite: bool | None = False,
        map_name: str | None = None,
    ) -> None:
        """
        Downloads a map contained within a polygon.

        Parameters
        ----------
        polygon : Polygon
            A polygon defining the boundaries of the map
        zoom_level : int, optional
            The zoom level to use, by default 14
        path_save : str, optional
            Path to save map sheets, by default "maps"
        overwrite : bool, optional
            Whether to overwrite existing maps, by default ``False``.
        map_name : str, optional
            Name to use when saving the map, by default None
        """

        assert isinstance(
            polygon, Polygon
        ), "[ERROR] \
Please pass polygon as shapely.geometry.Polygon object.\n\
[HINT] Use ``create_polygon_from_latlons()`` to create polygon."

        min_x, min_y, max_x, max_y = polygon.bounds

        start = Coordinate(min_y, max_x)  # (lat, lon)
        end = Coordinate(max_y, min_x)  # (lat, lon)

        start_idx = get_index_from_coordinate(start, zoom_level)
        end_idx = get_index_from_coordinate(end, zoom_level)
        grid_bb = GridBoundingBox(start_idx, end_idx)

        self._initialise_downloader()
        self._initialise_merger(f"{path_save}/")
        if not overwrite:
            if self._check_map_exists(grid_bb, map_name):
                return
        self._download_map(grid_bb, map_name)

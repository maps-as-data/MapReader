from typing import Union, Tuple
from shapely.geometry import Polygon
from .data_structures import Coordinate, GridBoundingBox
from .tile_loading import TileDownloader
from .tile_merging import TileMerger
from .downloader_utils import get_index_from_coordinate


class Downloader:
    """
    A class to download maps (no metadata)
    """

    def __init__(
        self,
        download_url: Union[str, list],
    ):
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
        self.downloader = TileDownloader(tile_servers=self.download_url)

    def _initialise_merger(self, path_save: str):
        self.merger = TileMerger(output_folder=path_save, show_progress=False)

    def _download_map(self, grid_bb: GridBoundingBox):
        map_name = str(grid_bb)
        self.downloader.download_tiles(grid_bb, download_in_parallel=False)
        self.merger.merge(grid_bb)
        print(f"[INFO] Downloaded {map_name}")

    def download_map_by_polygon(
        self,
        polygon: Polygon,
        download_url: Union[str, list],
        zoom_level: int = 14,
        path_save: str = "./maps/",
    ) -> None:
        """
        Note
        -----
        Use ``create_polygon_from_latlons()`` to create polygon.
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
        self._initialise_merger(path_save)
        self._download_map(grid_bb)

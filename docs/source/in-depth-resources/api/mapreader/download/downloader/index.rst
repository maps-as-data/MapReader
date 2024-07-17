mapreader.download.downloader
=============================

.. py:module:: mapreader.download.downloader


Classes
-------

.. autoapisummary::

   mapreader.download.downloader.Downloader


Module Contents
---------------

.. py:class:: Downloader(download_url)

   A class to download maps (without using metadata)


   .. py:method:: download_map_by_polygon(polygon, zoom_level = 14, path_save = 'maps', overwrite = False, map_name = None)

      Downloads a map contained within a polygon.

      :param polygon: A polygon defining the boundaries of the map
      :type polygon: Polygon
      :param zoom_level: The zoom level to use, by default 14
      :type zoom_level: int, optional
      :param path_save: Path to save map sheets, by default "maps"
      :type path_save: str, optional
      :param overwrite: Whether to overwrite existing maps, by default ``False``.
      :type overwrite: bool, optional
      :param map_name: Name to use when saving the map, by default None
      :type map_name: str, optional

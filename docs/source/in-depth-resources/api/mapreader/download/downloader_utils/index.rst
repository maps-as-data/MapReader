mapreader.download.downloader_utils
===================================

.. py:module:: mapreader.download.downloader_utils


Functions
---------

.. autoapisummary::

   mapreader.download.downloader_utils.create_polygon_from_latlons
   mapreader.download.downloader_utils.create_line_from_latlons
   mapreader.download.downloader_utils.get_grid_bb_from_polygon
   mapreader.download.downloader_utils.get_polygon_from_grid_bb
   mapreader.download.downloader_utils.get_index_from_coordinate
   mapreader.download.downloader_utils.get_coordinate_from_index


Module Contents
---------------

.. py:function:: create_polygon_from_latlons(min_lat, min_lon, max_lat, max_lon)

   Creates a polygon from latitudes and longitudes.

   :param min_lat: minimum latitude
   :type min_lat: float
   :param min_lon: minimum longitude
   :type min_lon: float
   :param max_lat: maximum latitude
   :type max_lat: float
   :param max_lon: maximum longitude
   :type max_lon: float

   :returns: shapely Polgyon
   :rtype: Polygon


.. py:function:: create_line_from_latlons(lat1_lon1, lat2_lon2)

   Creates a line between two points.

   :param lat1_lon1: Tuple defining first point
   :type lat1_lon1: tuple
   :param lat2: Tuple defining second point
   :type lat2: tuple

   :returns: shapely LineString
   :rtype: LineString


.. py:function:: get_grid_bb_from_polygon(polygon, zoom_level)

   Create GridBoundingBox object from shapely.Polygon

   :param polygon: shapely.Polygon to convert.
   :type polygon: shapely.Polygon
   :param zoom_level: Zoom level to use when creating GridBoundingBox
   :type zoom_level: int

   :rtype: GridBoundingBox


.. py:function:: get_polygon_from_grid_bb(grid_bb)

   Create shapely.Polygon object from GridBoundingBox

   :param grid_bb: GridBoundingBox to convert.
   :type grid_bb: GridBoundingBox

   :rtype: shapely.Polygon


.. py:function:: get_index_from_coordinate(coordinate, zoom)

   Create GridIndex object from Coordinate.

   :param coordinate: Coordinate to convert
   :type coordinate: Coordinate
   :param zoom: Zoom level to use when creating GridIndex
   :type zoom: int

   :rtype: GridIndex


.. py:function:: get_coordinate_from_index(grid_index)

   Create Coordinate object from GridIndex.

   :param grid_index: GridIndex to convert
   :type grid_index: GridIndex

   :returns: The upper left corner of the tile.
   :rtype: Coordinate

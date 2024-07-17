mapreader.load.geo_utils
========================

.. py:module:: mapreader.load.geo_utils


Functions
---------

.. autoapisummary::

   mapreader.load.geo_utils.extractGeoInfo
   mapreader.load.geo_utils.reproject_geo_info


Module Contents
---------------

.. py:function:: extractGeoInfo(image_path)

   Extract geographic information (shape, CRS and coordinates) from GeoTiff files

   :param image_path: Path to image
   :type image_path: str

   :returns: shape, CRS, coord
   :rtype: list


.. py:function:: reproject_geo_info(image_path, target_crs='EPSG:4326', calc_size_in_m=False)

   Extract geographic information from GeoTiff files and reproject to specified CRS (`target_crs`).

   :param image_path: Path to image
   :type image_path: str
   :param target_crs: Projection to convert coordinates into, by default "EPSG:4326"
   :type target_crs: str, optional
   :param calc_size_in_m: Method to compute pixel widths and heights, choices between "geodesic" and "great-circle" or "gc", by default "great-circle", by default False
   :type calc_size_in_m: str or bool, optional

   :returns: shape, old CRS, new CRS, reprojected coord, size in meters
   :rtype: list

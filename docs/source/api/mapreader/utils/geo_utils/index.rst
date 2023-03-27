:py:mod:`mapreader.utils.geo_utils`
===================================

.. py:module:: mapreader.utils.geo_utils


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   mapreader.utils.geo_utils.extractGeoInfo
   mapreader.utils.geo_utils.reprojectGeoInfo



.. py:function:: extractGeoInfo(image_path)

   Extract geographic information (shape, CRS and coordinates) from GeoTiff files

   Parameters
   ----------
   image_path : str
       Path to image

   Returns
   -------
   list
       shape, CRS, coord


.. py:function:: reprojectGeoInfo(image_path, proj2convert='EPSG:4326', calc_size_in_m=False)

   Extract geographic information from GeoTiff files and reproject to specified CRS (`proj2convert`).

   Parameters
   ----------
   image_path : str
       Path to image
   proj2convert : str, optional
       Projection to convert coordinates into, by default "EPSG:4326"
   calc_size_in_m : str or bool, optional
       Method to compute pixel widths and heights, choices between "geodesic" and "great-circle" or "gc", by default "great-circle", by default False

   Returns
   -------
   list
       shape, old CRS, new CRS, reprojected coord, size in meters



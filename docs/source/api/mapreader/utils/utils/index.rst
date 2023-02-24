:py:mod:`mapreader.utils.utils`
===============================

.. py:module:: mapreader.utils.utils


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   mapreader.utils.utils.extractGeoInfo



.. py:function:: extractGeoInfo(image_path, proj2convert='EPSG:4326', calc_size_in_m=False)

   Extract geographic information (coordinates, size in meters) from GeoTiff files

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
       coords, tiff_shape, size_in_m



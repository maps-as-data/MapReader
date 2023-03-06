:py:mod:`mapreader.utils.utils`
===============================

.. py:module:: mapreader.utils.utils


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   mapreader.utils.utils.extractGeoInfo



.. py:function:: extractGeoInfo(image_path, proj1='epsg:3857', proj2='epsg:4326', calc_size_in_m=False)

   Extract geographic information (coordinates, size in meters) from GeoTiff files

   Args:
       image_path (str) -- Path to image (GeoTiff format)
       proj1 (str) -- Projection from proj1 ---> proj2, here, specify proj1. Defaults to 'epsg:3857'.
       proj2 (str) -- Projection from proj1 ---> proj2, here, specify proj2. Defaults to 'epsg:4326'.
       calc_size_in_m (bool, optional) -- Calculate size of the image (in meters).
           Options: 'geodesic'; 'gc' or 'great-circle'; False ; Defaults to False.

   Returns:
       xmin, xmax, ymin, ymax, tiff_shape, size_in_m



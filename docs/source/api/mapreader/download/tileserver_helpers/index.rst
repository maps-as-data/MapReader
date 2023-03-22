:py:mod:`mapreader.download.tileserver_helpers`
===============================================

.. py:module:: mapreader.download.tileserver_helpers


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   mapreader.download.tileserver_helpers.input_class



Functions
~~~~~~~~~

.. autoapisummary::

   mapreader.download.tileserver_helpers.create_hf
   mapreader.download.tileserver_helpers.latlon2tile
   mapreader.download.tileserver_helpers.tile2latlon
   mapreader.download.tileserver_helpers.collect_coord_info
   mapreader.download.tileserver_helpers.check_par_jobs



.. py:function:: create_hf(geom)

   Create header and footer for different types of geometries

   Args:
       geom (str): geometry type, e.g., polygon


.. py:class:: input_class(name)

   initialize input class


.. py:function:: latlon2tile(lat, lon, zoom)

   Convert lat/lon/zoom to tiles

   from OSM Slippy Tile definitions & https://github.com/Caged/tile-stitch
   Reference: https://github.com/stamen/the-ultimate-tile-stitcher


.. py:function:: tile2latlon(x, y, zoom)

   Convert x/y/zoom to lat/lon

   Reference: https://github.com/stamen/the-ultimate-tile-stitcher


.. py:function:: collect_coord_info(list_files)

   Collect min/max lat/lon from a list of tiles

   Args:
       list_files (list): list of files to be read


.. py:function:: check_par_jobs(jobs, sleep_time=1)

   check if all the parallel jobs are finished
   :param jobs:
   :param sleep_time:
   :return:



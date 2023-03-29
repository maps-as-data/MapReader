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

   Creates a header and footer for a GeoJSON file of a specified geometry
   type.

   Parameters
   ----------
   geom : str
       The geometry type for the GeoJSON file. Currently only ``"polygon"`` is
       implemented.

   Returns
   -------
   Tuple[str, str]
       A tuple of strings representing the header and footer of the GeoJSON
       file.

   Raises
   ------
   NotImplementedError
       If the specified geometry type is not implemented.


.. py:class:: input_class(name)

   A simple class with one property, ``name``, used by MapReader's TileServer
   scraper's and stitcher's ``runner`` and ``scraper`` functions respectively.

   Parameters
   ----------
   name : str
       The name of the input.

   Attributes
   ----------
   name : str
       The name of the input.


.. py:function:: latlon2tile(lat, lon, zoom)

   Convert latitude and longitude coordinates to tile indices at a given zoom
   level.

   Parameters
   ----------
   lat : float
       Latitude in decimal degrees.
   lon : float
       Longitude in decimal degrees.
   zoom : int
       Zoom level, which determines the resolution of the tile.

   Returns
   -------
   Tuple[int, int]
       The x and y tile indices corresponding to the input ``latitude`` and
       ``longitude`` coordinates at the provided ``zoom`` (zoom level).

   Notes
   -----
   From OSM Slippy Tile definitions & https://github.com/Caged/tile-stitch.

   Reference: https://github.com/stamen/the-ultimate-tile-stitcher.


.. py:function:: tile2latlon(x, y, zoom)

   Convert tile coordinates (``x``, ``y``) and ``zoom`` (zoom level) to
   latitude and longitude coordinates.

   Parameters
   ----------
   x : int
       Tile X coordinate.
   y : int
       Tile Y coordinate.
   zoom : int
       Zoom level.

   Returns
   -------
   tuple
       A tuple containing latitude and longitude coordinates in degrees.

   Notes
   -----
   Reference: https://github.com/stamen/the-ultimate-tile-stitcher.


.. py:function:: collect_coord_info(list_files)

   Collects the minimum and maximum latitude and longitude from a list of
   tiles.

   Parameters
   ----------
   list_files : list of str
       List of file paths to be read.

   Returns
   -------
   tuple
       A tuple containing the minimum longitude, maximum longitude,
       minimum latitude, and maximum latitude of the tiles.


.. py:function:: check_par_jobs(jobs, sleep_time = 1)

   Wait for all processes in a list of parallel jobs to finish.

   Parameters
   ----------
   jobs : list
       A list of processes.
   sleep_time : float, optional
       Time to wait before checking the status of processes. Defaults to
       ``1``.

   Returns
   -------
   None

   ..
       TODO: This function's documentation needs a type for the List[...]
       type provided for the jobs parameter above. What is it?



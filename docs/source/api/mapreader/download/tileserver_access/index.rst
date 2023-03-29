:py:mod:`mapreader.download.tileserver_access`
==============================================

.. py:module:: mapreader.download.tileserver_access


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   mapreader.download.tileserver_access.TileServer



Functions
~~~~~~~~~

.. autoapisummary::

   mapreader.download.tileserver_access.download_tileserver_parallel



.. py:class:: TileServer(metadata_path, geometry = 'polygon', download_url = 'https://mapseries-tilesets.s3.amazonaws.com/1inch_2nd_ed/{z}/{x}/{y}.png')

   A class representing a tile server for a map.

   Parameters
   ----------
   metadata_path : str, dict, or list
       The path to the metadata file for the map. This can be a string
       representing the file path, a dictionary containing the metadata,
       or a list of metadata features. Usually, it is a string representing
       the file path to a metadata file downloaded from a tileserver.

       Some example metadata files can be found in
       `MapReader/worked_examples/persistent_data <https://github.com/Living-with-machines/MapReader/tree/main/worked_examples/persistent_data>`_.
   geometry : str, optional
       The type of geometry that defines the boundaries in the map. Defaults
       to ``"polygone"``.
   download_url : str, optional
       The base URL pattern used to download tiles from the server. This
       should contain placeholders for the x coordinate (``x``), the y
       coordinate (``y``) and the zoom level (``z``).

       Defaults to a URL for a
       specific tileset:
       ``https://mapseries-tilesets.s3.amazonaws.com/1inch_2nd_ed/{z}/{x}/{y}.png``

   Attributes
   ----------
   detected_rect_boundary : bool
       Whether or not the rectangular boundary of the map has been detected.
   found_queries : None
       Placeholder for a list of found queries.
   geometry : str
       The type of geometry used for the map.
   download_url : str
       The URL pattern used to download tiles from the server.
   metadata : list
       A list of metadata features for the map.  Each key in this dict should
       contain:

       - ``["geometry"]["coordinates"]``
       - ``["properties"]["IMAGEURL"]``
       - ``["properties"]["IMAGE"]``

   Raises
   ------
   ValueError
       If the metadata file could not be found or loaded.

   .. py:method:: create_info()

      Collects metadata information and boundary coordinates for fast
      queries.

      Populates the ``metadata_info_list`` and ``metadata_coord_arr``
      attributes of the ``TileServer`` instance with information about the
      map's metadata and boundary coordinates, respectively. Sets the
      ``detected_rect_boundary`` attribute to ``True``.

      Returns
      -------
      None

      Notes
      -----
      This is a helper function for other methods in this class


   .. py:method:: modify_metadata(remove_image_ids = [], only_keep_image_ids = [])

      Modifies the metadata by removing or keeping specified images.

      Parameters
      ----------
      remove_image_ids : list of str, optional
          List of image IDs to remove from the metadata (default is an empty
          list, ``[]``).
      only_keep_image_ids : list of str, optional
          List of image IDs to keep in the metadata (default is an empty
          list, ``[]``).

      Returns
      -------
      None

      Notes
      -----
      Removes image metadata whose IDs are in the ``remove_image_ids`` list,
      and keeps only image metadata whose IDs are in ``only_keep_image_ids``.
      Populates the ``metadata`` attribute of the ``TileServer`` instance
      with the modified metadata. If any metadata is removed, the
      :meth:`mapreader.download.tileserver_access.create_info` method is
      called to update the boundary coordinates for fast queries.


   .. py:method:: query_point(latlon_list, append = False)

      Queries the point(s) specified by ``latlon_list`` and returns
      information about the map tile(s) that contain the point(s).

      Parameters
      ----------
      latlon_list : list of tuples or tuple
          The list of latitude-longitude pairs to query. Each tuple must
          have the form ``(latitude, longitude)``. If only one pair is
          provided, it can be passed as a tuple instead of a list of tuples.
      append : bool, optional
          Whether to append the query results to any previously found
          queries, or to overwrite them. Defaults to ``False``.

      Returns
      -------
      None
          The query results are stored in the attribute `found_queries` of
          the TileServer instance.

      Notes
      -----
      Before performing the query, the function checks if the boundaries of
      the map tiles have been detected. If not, it runs the method
      :meth:`mapreader.download.tileserver_access.create_info` to detect the
      boundaries.

      The query results are stored in the attribute `found_queries` of the
      TileServer instance as a list of lists, where each sublist corresponds
      to a map tile and has the form: ``[image_url, image_filename,
      [min_lon, max_lon, min_lat, max_lat], index_in_metadata]``.


   .. py:method:: print_found_queries()

      Print the found queries in a formatted way.

      Returns
      -------
      None

      Examples
      --------
      .. code-block:: python

          >>> obj = TileServer()
          >>> obj.query_point([(40.0, -105.0)])
          >>> obj.print_found_queries()
          ------------
          Found items:
          ------------
          URL:      https://example.com/image1.png
          filepath: map_image1.png
          coords:   [min_lon, max_lon, min_lat, max_lat]
          index:    0
          ====================


   .. py:method:: detect_rectangle_boundary(coords)

      Detects the rectangular boundary of a polygon defined by a list of
      coordinates.

      Parameters
      ----------
      coords : list of tuples
          The list of coordinates defining the polygon.

      Returns
      -------
      float, float, float, float
          The minimum longitude, maximum longitude, minimum latitude, and
          maximum latitude of the rectangular boundary of the polygon.



   .. py:method:: create_metadata_query()

      Create a list of metadata query based on the found queries.

      Returns
      -------
      None
          Nothing is returned but the TileServer instance's
          ``metadata_query`` property is set to a list of metadata query
          based on the found queries.

      Notes
      -----
      This is used in the method
      :meth:`mapreader.download.tileserver_access.download_tileserver`.


   .. py:method:: minmax_latlon()

      Print the minimum and maximum longitude and latitude values for the
      metadata.

      Returns
          None
              Will print a result like this:

              .. code-block:: python
              
                  Min/Max Lon: <min_longitude>, <max_longitude>
                  Min/Max Lat: <min_latitude>, <max_latitude>

      Notes
      -----
      If the rectangle boundary has not been detected yet, the method checks
      if the boundaries of the map tiles have been detected. If not, it runs
      the method :meth:`mapreader.download.tileserver_access.create_info` to
      detect the boundaries.


   .. py:method:: download_tileserver(mode = 'queries', num_img2test = -1, zoom_level = 14, retries = 10, scraper_max_connections = 4, failed_urls_path = 'failed_urls.txt', tile_tmp_dir = 'tiles', output_maps_dirname = 'maps', output_metadata_filename = 'metadata.csv', pixel_closest = None, redownload = False, id1 = 0, id2 = -1, error_path = 'errors.txt', max_num_errors = 20)

      Downloads map tiles from a tileserver using a scraper and stitches
      them into a larger map image.

      Parameters
      ----------
      mode : str, optional
          Metadata query type, which can be ``"queries"`` (default) or
          ``"query"``, both of which will download the queried maps. It can
          also be set to ``"all"``, which means that all maps in the
          metadata file will be downloaded.
      num_img2test : int, optional
          Number of images to download for testing, by default ``-1``.
      zoom_level : int, optional
          Zoom level to retrieve map tiles from, by default ``14``.
      retries : int, optional
          Number of times to retry a failed download, by default ``10``.
      scraper_max_connections : int, optional
          Maximum number of simultaneous connections for the scraper, by
          default ``4``.
      failed_urls_path : str, optional
          Path to save failed URLs, by default ``"failed_urls.txt"``.
      tile_tmp_dir : str, optional
          Directory to temporarily save map tiles, by default ``"tiles"``.
      output_maps_dirname : str, optional
          Directory to save combined map images, by default ``"maps"``.
      output_metadata_filename : str, optional
          Name of the output metadatata file, by default ``"metadata.csv"``.

          *Note: This file will be saved in the path equivalent to
          output_maps_dirname/output_metadata_filename.*
      pixel_closest : int, optional
          Adjust the number of pixels in both directions (width and height)
          after downloading a map. For example, if ``pixel_closest = 100``,
          the number of pixels in both directions will be multiples of 100.

          `This helps to create only square tiles in the processing step.`
      redownload : bool, optional
          Whether to redownload previously downloaded maps that already
          exist in the local directory, by default ``False``.
      id1 : int, optional
          The starting index (in the ``metadata`` property) for downloading
          maps, by default ``0``.
      id2 : int, optional
          The ending index (in the ``metadata`` property) for downloading
          maps, by default ``-1`` (all images).
      error_path : str, optional
          The path to the file for logging errors, by default
          ``"errors.txt"``.
      max_num_errors : int, optional
          The maximum number of errors to allow before skipping a map, by
          default ``20``.

      Returns
      -------
      None


   .. py:method:: extract_region_dates_metadata(metadata_item)

      Extracts region name, surveyed date, revised date, and published date
      from a given GeoJSON feature, provided as a ``metadata_item``.

      Parameters
      ----------
      metadata_item : dict
          A GeoJSON feature, i.e. a dictionary which contains at least a
          nested dictionary in in the ``"properties"`` key, which contains a
          ``"WFS_TITLE"`` value.

      Returns
      -------
      Tuple[str, int, int, int]
          A tuple containing the region name (str), surveyed date (int),
          revised date (int), and published date (int). If any of the dates
          cannot be found, its value will be ``-1``. If no region name can
          be found, it will be ``"None"``.


   .. py:method:: find_and_clean_date(ois, ois_key = 'surveyed')
      :staticmethod:

      Find and extract a date string from a given string (``ois``), typically
      representing a date metadata attribute. The date string is cleaned by
      removing unnecessary tokens like "to" and "ca.".

      Parameters
      ----------
      ois : str
          A string containing the date metadata attribute and its value.
      ois_key : str, optional
          The keyword used to identify the date metadata attribute, by
          default ``"surveyed"``.

      Returns
      -------
      str
          The extracted date string, cleaned of unnecessary tokens.


   .. py:method:: plot_metadata_on_map(list2remove = [], map_extent = None, add_text=False)

      Plots metadata on a map (using ``cartopy`` library, if available).

      Parameters
      ----------
      list2remove : list, optional
          A list of IDs to remove from the plot. The default is ``[]``.
      map_extent : tuple or list, optional
          The extent of the map to be plotted. It should be a tuple or a
          list of the format ``[lon_min, lon_max, lat_min, lat_max]``. It
          can also be set to ``"uk"`` which will limit the map extent to the
          UK's boundaries. The default is ``None``.
      add_text : bool, optional
          If ``True``, adds ID texts next to each plotted metadata. The
          default is ``False``.

      Returns
      -------
      None


   .. py:method:: hist_published_dates(min_date = None, max_date = None)

      Plot a histogram of the published dates for all metadata items.

      Parameters
      ----------
      min_date : int, optional
          Minimum published date to be included in the histogram. If not
          given, the minimum published date among all metadata items will be
          used.
      max_date : int, optional
          Maximum published date to be included in the histogram. If not
          given, the maximum published date among all metadata items will be
          used.

      Raises
      ------
      ValueError
          If any of the published dates cannot be converted to an integer.

      Returns
      -------
      None

      Notes
      -----
      The method extracts the published date from each metadata item using
      the method
      :meth:`mapreader.download.tileserver_access.extract_region_dates_metadata`
      and creates a histogram of the counts of published dates falling
      within the given range. The histogram is plotted using
      `matplotlib.pyplot.hist`.

      If `min_date` or `max_date` are given, only the published dates
      falling within that range will be included in the histogram. Otherwise,
      the histogram will include all published dates in the metadata.


   .. py:method:: download_tileserver_rect(mode = 'queries', num_img2test = -1, zoom_level = 14, adjust_mult = 0.005, retries = 1, failed_urls_path = 'failed_urls.txt', tile_tmp_dir = 'tiles', output_maps_dirname = 'maps', output_metadata_filename = 'metadata.csv', pixel_closest = None, redownload = False, id1 = 0, id2 = -1, min_lat_len = 0.05, min_lon_len = 0.05)

      Downloads map tiles from a tileserver using a scraper and stitches
      them into a larger map image.

      Parameters
      ----------
      mode : str, optional
          Metadata query type, which can be ``"queries"`` (default) or
          ``"query"``, both of which will download the queried maps. It can
          also be set to ``"all"``, which means that all maps in the
          metadata file will be downloaded.
      num_img2test : int, optional
          Number of images to download for testing, by default ``-1``.
      zoom_level : int, optional
          Zoom level to retrieve map tiles from, by default ``14``.
      adjust_mult : float, optional
          If some tiles cannot be downloaded, shrink the requested bounding
          box by this factor. Defaults to ``0.005``.
      retries : int, optional
          Number of times to retry a failed download, by default ``10``.
      failed_urls_path : str, optional
          Path to save failed URLs, by default ``"failed_urls.txt"``.
      tile_tmp_dir : str, optional
          Directory to temporarily save map tiles, by default ``"tiles"``.
      output_maps_dirname : str, optional
          Directory to save combined map images, by default ``"maps"``.
      output_metadata_filename : str, optional
          Name of the output metadata file, by default ``"metadata.csv"``.

          *Note: This file will be saved in the path equivalent to
          output_maps_dirname/output_metadata_filename.*
      pixel_closest : int, optional
          Adjust the number of pixels in both directions (width and height)
          after downloading a map. For example, if ``pixel_closest = 100``,
          the number of pixels in both directions will be multiples of 100.

          `This helps to create only square tiles in the processing step.`
      redownload : bool, optional
          Whether to redownload previously downloaded maps that already
          exist in the local directory, by default ``False``.
      id1 : int, optional
          The starting index (in the ``metadata`` property) for downloading
          maps, by default ``0``.
      id2 : int, optional
          The ending index (in the ``metadata`` property) for downloading
          maps, by default ``-1`` (all images).
      min_lat_len : float, optional
          Minimum length of the latitude (in degrees) of the bounding box
          for each tileserver request. Default is ``0.05``.
      min_lon_len : float, optional
          Minimum length of the longitude (in degrees) of the bounding box
          for each tileserver request. Default is ``0.05``.

      Returns
      -------
      None

      Notes
      -----
      The ``min_lat_len`` and ``min_lon_len`` are optional float parameters
      that represent the minimum length of the latitude and longitude,
      respectively, of the bounding box for each tileserver request. These
      parameters are used in the method to adjust the boundary of the map
      tile to be requested from the server. If the difference between the
      maximum and minimum latitude or longitude is less than the
      corresponding min_lat_len or min_lon_len, respectively, then the
      ``adjust_mult`` parameter is used to shrink the boundary until the
      minimum length requirements are met.



.. py:function:: download_tileserver_parallel(metadata, start, end, process_np = 8, **kwds)

   Downloads map tiles from a tileserver in parallel using multiprocessing.

   .. warning::
       This function does currently not work.

   Parameters
   ----------
   metadata : list of dictionaries
       A list of metadata dictionaries (in GeoJSON format) determining what
       to download from the tile server. Each dictionary that contains info
       about the maps to be downloaded needs to have three keys:
       ``"features"``, ``"geometry"`` (with a nested list of
       ``"coordinates"``), and ``"properties"`` (with two values for the keys
       ``"IMAGEURL"`` AND ``"IMAGE"``).
   start : int
       The index of the first element in the ``metadata`` property to
       download.
   end : int
       The index of the last element in ``metadata`` property to download.
   process_np : int, optional
       The number of processes to use for downloading. Defaults to ``8``.
   **kwds : dict, optional
       Keyword arguments passed to the ``download_tileserver`` function.

   Returns
   -------
   None

   ..
       Note/TODO: This function will not work, as download_tileserver is not
       defined in this scope. It belongs to the TileServer class...



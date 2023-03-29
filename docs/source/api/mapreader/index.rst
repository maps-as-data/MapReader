:py:mod:`mapreader`
===================

.. py:module:: mapreader


Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   annotate/index.rst
   download/index.rst
   learn/index.rst
   load/index.rst
   process/index.rst
   utils/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   mapreader.mapImages
   mapreader.TileServer
   mapreader.loadAnnotations
   mapreader.patchTorchDataset
   mapreader.patchContextDataset
   mapreader.classifier
   mapreader.classifierContext



Functions
~~~~~~~~~

.. autoapisummary::

   mapreader.loader
   mapreader.load_patches



.. py:class:: mapImages(path_images=False, tree_level='parent', parent_path=None, **kwds)

   mapImages class

   .. py:method:: imagesConstructor(image_path, parent_path=None, tree_level='patch', **kwds)

      Construct images instance variable,

      Arguments:
          image_path {str or None} -- Path to the image

      Keyword Arguments:
          parent_path {str or None} -- Path to the parent of image (default: {None})
          tree_level {str} -- Tree level, choices between parent and patch (default: {"patch"})


   .. py:method:: splitImagePath(inp_path)
      :staticmethod:

      split image path into basename and dirname,


   .. py:method:: add_metadata(metadata, columns=None, tree_level='parent', index_col=0, delimiter='|')

      Add metadata to images at tree_level,

      Args:
          metadata_path (path): path to a csv file, normally created from a pandas dataframe
          columns (list, optional): list of columns to be used. If None (default), all columns are used.
          tree_level (str, optional): parent/patch tree level. Defaults to "parent".
          index_col (int, optional): index column


   .. py:method:: show_sample(num_samples, tree_level='parent', random_seed=65, **kwds)

      Show sample images,

      Arguments:
          num_samples {int} -- Number of samples to be plotted

      Keyword Arguments:
          tree_level {str} -- XXX (default: {"patch"})
          random_seed {int} -- Random seed for reproducibility (default: {65})


   .. py:method:: list_parents()

      Return list of all parents


   .. py:method:: list_patches()

      Return list of all patches


   .. py:method:: add_shape(tree_level='parent')

      Run add_shape_id for all tree_level items


   .. py:method:: add_coord_increments(tree_level='parent')

      Run add_coord_increments_id for all tree_level items


   .. py:method:: add_center_coord(tree_level='patch')

      Run add_center_coord_id for all tree_level items


   .. py:method:: add_shape_id(image_id, tree_level='parent')

      Add an image/array shape to self.images[tree_level][image_id]

      Parameters
      ----------
      image_id : str
          image ID
      tree_level : str, optional
          Tree level, choices between parent and patch (default: {"patch"})


   .. py:method:: add_coord_increments_id(image_id, tree_level='parent')

      Add pixel-wise dlon and dlat to self.images[tree_level][image_id]

      Parameters
      ----------
      image_id : str
          image ID
      tree_level : str, optional
          Tree level, choices between parent and patch (default: {"patch"})


   .. py:method:: add_center_coord_id(image_id, tree_level='patch')

      Add center_lon and center_lat to self.images[tree_level][image_id]

      Parameters
      ----------
      image_id : str
          image ID
      tree_level : str, optional
          Tree level, choices between parent and patch (default: {"patch"})


   .. py:method:: calc_pixel_width_height(parent_id, calc_size_in_m='great-circle')

      Calculate width and height of pixels

      Args:
          parent_id (str): ID of the parent image
          calc_size_in_m (str, optional): How to compute the width/heigh, options: geodesic and great-circle (default).


   .. py:method:: patchifyAll(method='pixel', patch_size=100, path_save='patches', square_cuts=False, resize_factor=False, output_format='png', rewrite=False, verbose=False, tree_level='parent', add2par=True, id1=0, id2=-1)

      Patchify all images in the object (the list can be accessed via .images variable)

      Keyword Arguments:
          method {str} -- method to slice an image (default: {"pixel"})
          patch_size {int} -- Number of pixels in both x and y directions (default: {100})
          path_save {str} -- Directory to save the patches (default: {"patches"})
          square_cuts {bool} -- All patches will have the same number of pixels in x and y (default: {True})
          resize_factor {bool} -- Resize image before slicing (default: {False})
          output_format {str} -- Output format (default: {"png"})
          tree_level {str} -- image group to be sliced (default: {"parent"})
          verbose {bool} -- Print the progress (default: {False})


   .. py:method:: addPatches()

      Add patches to parent


   .. py:method:: calc_pixel_stats(parent_id=None, calc_mean=True, calc_std=True)

      Calculate stats of each patch in a parent_id and
         store the results

      Arguments:
          parent_id {str, None} -- ID of the parent image. If None, all parents will be used.


   .. py:method:: convertImages(fmt='dataframe')

      Convert images to a specified format (fmt)

      Keyword Arguments:
          fmt {str} -- convert images variable to this format (default: {"dataframe"})


   .. py:method:: show_par(parent_id, value=False, **kwds)

      A wrapper function for show,

      Arguments:
          parent_id {str} -- ID of the parent image to be plotted

      Keyword Arguments:
          value {bool, const, random, ...} -- Values to be plotted on the parent image (default: {False})


   .. py:method:: show(image_ids, value=False, plot_parent=True, border=True, border_color='r', vmin=0.5, vmax=2.5, colorbar='viridis', alpha=1.0, discrete_colorbar=256, tree_level='patch', grid_plot=(20000, 20000), plot_histogram=True, save_kml_dir=False, image_width_resolution=None, kml_dpi_image=None, **kwds)

      Plot a list of image ids,

      Arguments:
          image_ids {list} -- List of image ids to be plotted

      Keyword Arguments:
          value {False or list} -- Value to be plotted on patch images
          plot_parent {bool} -- Plot parent image in the background (default: {True})
          border {bool} -- Plot a border for each image id (default: {True})
          border_color {str} -- color of patch borders (default: {r})
          vmin {float or list} -- min. value for the colorbar (default: {0.5})
          vmax {float or list} -- max. value for the colorbar (default: {2.5})
          colorbar {str or list} -- colorbar to visualize "value" on maps (default: {viridis})
          alpha {float or list} -- set transparency level for plotting "value" on maps (default: {1.})
          discrete_colorbar {int or list} -- number of discrete colors to be used (default: {256})
          tree_level {str} -- Tree level for the plot XXX (default: {"patch"})
          grid_plot {list or tuple} -- Number of rows and columns in the image.
                                       This will later adjusted to the true min/max of all subplots.
                                       (default: (10000, 10000))
          plot_histogram {bool} -- Plot a histogram of 'value' (default: {True})
          save_kml_dir {False or str} -- Directory to save a KML files out of images or False
                                         (default: {False})
          image_width_resolution {None, int} -- pixel width to be used for plotting, only when tree_level="parent"
                                                pixel height will be adjusted according to the width/height ratio
          kml_dpi_image {None, int} -- The resolution in dots per inch for images created when save_kml_dir is specified


   .. py:method:: loadPatches(patch_paths, parent_paths=False, add_geo_par=False, clear_images=False)

      load patches from files (patch_paths) and add parents if parent_paths is provided

      Arguments:
          patch_paths {str, wildcard accepted} -- path to patches
          parent_paths {False or str, wildcard accepted} -- path to parents

      Keyword Arguments:
          clear_images {bool} -- clear images variable before loading patches (default: {False})


   .. py:method:: detectParIDfromPath(image_id, parent_delimiter='#')
      :staticmethod:

      Detect parent ID from path using parent_delimiter
      NOTE: Currently, only one parent can be detected.


   .. py:method:: detectBorderFromPath(image_id, border_delimiter='-')
      :staticmethod:

      Detect borders from the path using border_delimiter.
      Here, the assumption is that the patch image is named:
      NOTE: STRING-min_x-min_y-max_x-max_y-STRING


   .. py:method:: loadParents(parent_paths=False, parent_ids=False, update=False, add_geo=False)

      load parent images from files (parent_paths)
         if only parent_ids is specified, self.images["parent"] will be filled with no image_path.
         NOTE: if parent_paths is given, parent_ids will be omitted as ids will be
               detected from the basename

      Keyword Arguments:
          parent_paths {False or str, wildcard accepted} -- path to parents (default: {False})
          parent_ids {False or list/tuple} -- list of parent ids (default: {False})


   .. py:method:: loadDataframe(parents=None, patch_df=None, clear_images=True)

      Read dataframes and form images variable

      Keyword Arguments:
          parents_df {dataframe or path} -- Parents dataframe or path to parents (default: {None})
          patch_df {dataframe} -- patches/slices dataframe (default: {None})
          clear_images {bool} -- clear images before reading dataframes (default: {True})


   .. py:method:: load_csv_file(parent_path=None, patch_path=None, clear_images=False, index_col_patch=0, index_col_parent=0)

      Read parent and patch from CSV files



.. py:function:: loader(path_images=False, tree_level='parent', parent_path=None, **kwds)

   Construct mapImages object by passing the image path,

   Keyword Arguments:
       path_images {str or False} -- path to one or many images

   Returns:
       [mapImages object] -- mapImages object contains various methods to work with images


.. py:function:: load_patches(patch_paths, parent_paths=False, add_geo_par=False, clear_images=False)


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



.. py:class:: loadAnnotations

   .. py:method:: load_all(csv_paths, **kwds)

      Load multiple CSV files into the class instance using the ``load``
      method.

      Parameters
      ----------
      csv_paths : str
          The file path pattern to match CSV files to load.
      **kwds : dict
          Additional keyword arguments to pass to the ``load`` method.

      Returns
      -------
      None


   .. py:method:: load(csv_path, path2dir = None, col_path = 'image_id', keep_these_cols = False, append = True, col_label = 'label', shuffle_rows = True, reset_index = True, random_state = 1234)

      Read and append an annotation file to the class instance's annotations
      DataFrame.

      Parameters
      ----------
      csv_path : str
          Path to an annotation file in CSV format.
      path2dir : str, optional
          Update the ``col_path`` column by adding ``path2dir/col_path``, by
          default ``None``.
      col_path : str, optional
          Name of the column that contains image paths, by default
          ``"image_id"``.
      keep_these_cols : bool, optional
          Only keep these columns. If ``False`` (default), all columns will
          be kept.
      append : bool, optional
          Append a newly read CSV file to the ``annotations`` property if
          set to ``True``. By default, ``True``.
      col_label : str, optional
          Name of the column that contains labels. Default is ``"label"``.
      shuffle_rows : bool, optional
          Shuffle rows after reading annotations. Default is ``True``.
      reset_index : bool, optional
          Reset the index of the annotation DataFrame at the end of the
          method. Default is ``True``.
      random_state : int, optional
          Random seed for row shuffling. Default is ``1234``.

      Returns
      -------
      None


   .. py:method:: set_col_label(new_label = 'label')

      Set a new label for the column that contains the labels.

      Parameters
      ----------
      new_label : str, optional
          Name of the new label column, by default ``"label"``.

      Returns
      -------
      None


   .. py:method:: show_image(indx, cmap = 'viridis')

      Display an image specified by its index along with its label.

      Parameters
      ----------
      indx : int
          Index of the image in the annotations DataFrame to display.
      cmap : str, optional
          The colormap to use, by default ``"viridis"``.

          To see available colormaps, see
          https://matplotlib.org/stable/gallery/color/colormap_reference.html.

      Returns
      -------
      None


   .. py:method:: adjust_labels(shiftby = -1)

      Shift labels in the self.annotations DataFrame by the specified value
      (``shiftby``).

      Parameters
      ----------
      shiftby : int, optional
          The value to shift labels by. Default is ``-1``.

      Returns
      -------
      None

      Notes
      -----
      This function updates the ``self.annotations`` DataFrame by adding the
      value of ``shiftby`` to the values of the ``self.col_label`` column. It
      also prints the value counts of the ``self.col_label`` column before
      and after the shift.


   .. py:method:: review_labels(tar_label = None, start_indx = 1, chunks = 8 * 6, num_cols = 8, figsize = (8 * 3, 8 * 2), exclude_df = None, include_df = None, deduplicate_col = 'image_id')

      Perform image review on annotations and update labels for a given
      label or all labels.

      Parameters
      ----------
      tar_label : int, optional
          The target label to review. If not provided, all labels will be
          reviewed, by default ``None``.
      start_indx : int, optional
          The index of the first image to review, by default ``1``.
      chunks : int, optional
          The number of images to display at a time, by default ``8 * 6``.
      num_cols : int, optional
          The number of columns in the display, by default ``8``.
      figsize : list or tuple, optional
          The size of the display window, by default ``(8 * 3, 8 * 2)``.
      exclude_df : pandas.DataFrame, optional
          A DataFrame of images to exclude from review, by default ``None``.
      include_df : pandas.DataFrame, optional
          A DataFrame of images to include for review, by default ``None``.
      deduplicate_col : str, optional
          The column to use for deduplicating reviewed images, by default
          ``"image_id"``.

      Returns
      -------
      None

      Notes
      ------
      This method reviews images with their corresponding labels and allows
      the user to change the label for each image. The updated labels are
      saved in both the annotations and reviewed DataFrames. If
      ``exclude_df`` is provided, images with ``image_path`` in
      ``exclude_df["image_path"]`` are skipped in the review process. If
      ``include_df`` is provided, only images with ``image_path`` in
      ``include_df["image_path"]`` are reviewed. The reviewed DataFrame is
      deduplicated based on the ``deduplicate_col``.


   .. py:method:: show_image_labels(tar_label = 1, num_sample = 10)

      Show a random sample of images with the specified label (tar_label).

      Parameters
      ----------
      tar_label : int, optional
          The label to filter the images by. Default is ``1``.
      num_sample : int, optional
          The number of images to show. If ``None``, all images with the
          specified label will be shown. Default is ``10``.

      Returns
      -------
      None


   .. py:method:: split_annotations(stratify_colname = 'label', frac_train = 0.7, frac_val = 0.15, frac_test = 0.15, random_state = 1364)

      Splits the dataset into three subsets: training, validation, and test
      sets (DataFrames).

      Parameters
      ----------
      stratify_colname : str, optional
          Name of the column on which to stratify the split. The default is
          ``"label"``.
      frac_train : float, optional
          Fraction of the dataset to be used for training. The default is
          ``0.70``.
      frac_val : float, optional
          Fraction of the dataset to be used for validation. The default is
          ``0.15``.
      frac_test : float, optional
          Fraction of the dataset to be used for testing. The default is
          ``0.15``.
      random_state : int, optional
          Random seed to ensure reproducibility. The default is ``1364``.

      Raises
      ------
      ValueError
          If the sum of fractions of training, validation and test sets does
          not add up to 1.

      ValueError
          If ``stratify_colname`` is not a column in the dataframe.

      Returns
      -------
      None
          Sets properties ``df_train``, ``df_val``, ``df_test`` -- three
          Dataframes containing the three splits on the ``loadAnnotations``
          instance.

      Notes
      -----
      Following fractional ratios provided by the user, where each subset is
      stratified by the values in a specific column (that is, each subset has
      the same relative frequency of the values in the column). It performs
      this splitting by running ``train_test_split()`` twice.


   .. py:method:: sample_labels(tar_label, num_samples, random_state = 12345)

      Randomly sample a given number of annotations with a given target
      label and remove all other annotations from the dataframe.

      Parameters
      ----------
      tar_label : int or str
          The target label for which the annotations will be sampled.
      num_samples : int
          The number of annotations to be sampled.
      random_state : int, optional
          Seed to ensure reproducibility of the random number generator.
          Default is ``12345``.

      Raises
      ------
      ValueError
          If ``tar_label`` is not a column in the dataframe.

      Returns
      -------
      None
          The dataframe with remaining annotations is stored in
          ``self.annotations``.



.. py:class:: patchTorchDataset(patchframe, transform = None, label_col = 'label', convert2 = 'RGB', input_col = 0)

   Bases: :py:obj:`torch.utils.data.Dataset`

   An abstract class representing a :class:`Dataset`.

   All datasets that represent a map from keys to data samples should subclass
   it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
   data sample for a given key. Subclasses could also optionally overwrite
   :meth:`__len__`, which is expected to return the size of the dataset by many
   :class:`~torch.utils.data.Sampler` implementations and the default options
   of :class:`~torch.utils.data.DataLoader`.

   .. note::
     :class:`~torch.utils.data.DataLoader` by default constructs a index
     sampler that yields integral indices.  To make it work with a map-style
     dataset with non-integral indices/keys, a custom sampler must be provided.

   .. py:method:: return_orig_image(idx)

      Return the original image associated with the given index.

      Parameters
      ----------
      idx : int or Tensor
          The index of the desired image, or a Tensor containing the index.

      Returns
      -------
      PIL.Image.Image
          The original image associated with the given index.

      Notes
      -----
      This method returns the original image associated with the given index
      by loading the image file using the file path stored in the
      ``input_col`` column of the ``patchframe`` DataFrame at the given
      index. The loaded image is then converted to the format specified by
      the ``convert2`` attribute of the object. The resulting
      ``PIL.Image.Image`` object is returned.



.. py:class:: patchContextDataset(patchframe, transform1 = None, transform2 = None, label_col = 'label', convert2 = 'RGB', input_col = 0, context_save_path = './maps/maps_context', create_context = False, par_path = './maps', x_offset = 1.0, y_offset = 1.0, slice_method = 'scale')

   Bases: :py:obj:`torch.utils.data.Dataset`

   An abstract class representing a :class:`Dataset`.

   All datasets that represent a map from keys to data samples should subclass
   it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
   data sample for a given key. Subclasses could also optionally overwrite
   :meth:`__len__`, which is expected to return the size of the dataset by many
   :class:`~torch.utils.data.Sampler` implementations and the default options
   of :class:`~torch.utils.data.DataLoader`.

   .. note::
     :class:`~torch.utils.data.DataLoader` by default constructs a index
     sampler that yields integral indices.  To make it work with a map-style
     dataset with non-integral indices/keys, a custom sampler must be provided.

   .. py:method:: save_parents(num_req_p = 10, sleep_time = 0.001, use_parhugin = True, par_split = '#', loc_split = '-', overwrite = False)

      Save parent patches for all patches in the patchframe.

      Parameters
      ----------
      num_req_p : int, optional
          The number of required processors for the job, by default 10.
      sleep_time : float, optional
          The time to wait between jobs, by default 0.001.
      use_parhugin : bool, optional
          Flag indicating whether to use Parhugin to parallelize the job, by
          default True.
      par_split : str, optional
          The string used to separate parent IDs in the patch filename, by
          default "#".
      loc_split : str, optional
          The string used to separate patch location and level in the patch
          filename, by default "-".
      overwrite : bool, optional
          Flag indicating whether to overwrite existing parent files, by
          default False.

      Returns
      -------
      None

      Notes
      -----
      Parhugin is a Python package for parallelizing computations across
      multiple CPU cores. The method uses Parhugin to parallelize the
      computation of saving parent patches to disk. When Parhugin is
      installed and ``use_parhugin`` is set to True, the method parallelizes
      the calling of the ``save_parents_idx`` method and its corresponding
      arguments. If Parhugin is not installed or ``use_parhugin`` is set to
      False, the method executes the loop over patch indices sequentially
      instead.


   .. py:method:: save_parents_idx(idx, par_split = '#', loc_split = '-', overwrite = False, return_image = False)

      Save the parents of a specific patch to the specified location.

      Parameters
      ----------
          idx : int
              Index of the patch in the dataset.
          par_split : str, optional
              Delimiter to split the parent names in the file path. Default
              is "#".
          loc_split : str, optional
              Delimiter to split the location of the patch in the file path.
              Default is "-".
          overwrite : bool, optional
              Whether to overwrite the existing parent files. Default is
              False.

      Raises
      ------
      ValueError
          If the patch is not found in the dataset.

      Returns
      -------
      None


   .. py:method:: return_orig_image(idx)

      Return the original image associated with the given index.

      Parameters
      ----------
      idx : int or Tensor
          The index of the desired image, or a Tensor containing the index.

      Returns
      -------
      PIL.Image.Image
          The original image associated with the given index.

      Notes
      -----
      This method returns the original image associated with the given index
      by loading the image file using the file path stored in the
      ``input_col`` column of the ``patchframe`` DataFrame at the given
      index. The loaded image is then converted to the format specified by
      the ``convert2`` attribute of the object. The resulting
      ``PIL.Image.Image`` object is returned.


   .. py:method:: plot_sample(indx)

      Plot a sample patch and its corresponding context from the dataset.

      Parameters
      ----------
      indx : int
          The index of the sample to plot.

      Returns
      -------
      None
          Displays the plot of the sample patch and its corresponding
          context.

      Notes
      -----
      This method plots a sample patch and its corresponding context side-by-
      side in a single figure with two subplots. The figure size is set to
      10in x 5in, and the titles of the subplots are set to "Patch" and
      "Context", respectively. The resulting figure is displayed using
      the ``matplotlib`` library (required).



.. py:class:: classifier(device = 'default')

   .. py:method:: set_classnames(classname_dict)

      Set the class names and the number of classes in the object detection
      model.

      Parameters
      ----------
      classname_dict : dict
          A dictionary containing the class IDs (as keys) and their
          corresponding names (as values). E.g.
          ``{0: "rail space", 1: "No rail space"}``

      Returns
      -------
      None


   .. py:method:: add2dataloader(dataset, set_name = None, batch_size = 16, shuffle = True, num_workers = 0, **kwds)

      Adds a PyTorch dataloader to the object's ``dataloader`` dictionary
      property and returns it.

      Parameters
      ----------
      dataset : torch.utils.data.Dataset
          The PyTorch dataset to use for the dataloader.
      set_name : str or None, optional
          The name to use when adding the dataloader to the object's
          ``dataloader`` dictionary property (e.g., ``"train"``, ``"val"``
          or ``"test"``).

          If ``None`` (default), the dataloader is returned without being
          added to the dictionary.
      batch_size : int, optional
          The batch size to use for the dataloader. Default is ``16``.
      shuffle : bool, optional
          Whether to shuffle the dataset during training. Default is
          ``True``.
      num_workers : int, optional
          The number of worker threads to use for loading data. Default is
          ``0``.
      **kwds :
          Additional keyword arguments to pass to PyTorch's ``DataLoader``
          constructor.

      Returns
      -------
      dl : torch.utils.data.DataLoader
          The dataloader that was created.


   .. py:method:: print_classes_dl(set_name = 'train')

      Prints information about the labels and class names (if available)
      associated with a dataloader.

      Parameters
      ----------
      set_name : str, optional
          The name of the dataloader to print information about, normally
          specified in ``self.add2dataloader``. Default is ``"train"``.

      Returns
      -------
      None


   .. py:method:: add_model(model, input_size = 224, is_inception = False)

      Add a PyTorch model to the classifier object.

      Parameters
      ----------
      model : nn.Module
          The PyTorch model to add to the object. See: ``torchvision.models``
      input_size : int, optional
          The expected input size of the model. Default is ``224``.
      is_inception : bool, optional
          Whether the model is an Inception-style model. Default is
          ``False``.

      Raises
      ------
      ValueError
          If the object's ``class_names`` attribute is ``None``. They should
          be specified with the ``set_classnames`` method.

      Returns
      -------
      None


   .. py:method:: del_model()

      Deletes the PyTorch model from the classifier object.

      Parameters
      ----------
      None

      Returns
      -------
      None

      Notes
      -----
      This function deletes the PyTorch model from the object and resets any
      associated metadata, such as the expected input size and whether the
      model is an Inception-style model. It also resets any associated
      metrics and best epoch/loss values.


   .. py:method:: layerwise_lr(min_lr, max_lr, ltype = 'linspace')

      Calculates layer-wise learning rates for a given set of model
      parameters.

      Parameters
      ----------
      min_lr : float
          The minimum learning rate to be used.
      max_lr : float
          The maximum learning rate to be used.
      ltype : str, optional
          The type of sequence to use for spacing the specified interval
          learning rates. Can be either ``"linspace"`` or ``"geomspace"``,
          where `"linspace"` uses evenly spaced learning rates over a
          specified interval and `"geomspace"` uses learning rates spaced
          evenly on a log scale (a geometric progression). Defaults to
          ``"linspace"``.

      Returns
      -------
      list of dicts
          A list of dictionaries containing the parameters and learning
          rates for each layer.


   .. py:method:: initialize_optimizer(optim_type = 'adam', params2optim = 'infer', optim_param_dict = {'lr': 0.001}, add_optim = True)

      Initializes an optimizer for the model and adds it to the classifier
      object.

      Parameters
      ----------
      optim_type : str, optional
          The type of optimizer to use. Can be set to ``"adam"`` (default),
          ``"adamw"``, or ``"sgd"``.
      params2optim : str or iterable, optional
          The parameters to optimize. If set to ``"infer"``, all model
          parameters that require gradients will be optimized, by default
          ``"infer"``.
      optim_param_dict : dict, optional
          The parameters to pass to the optimizer constructor as a
          dictionary, by default ``{"lr": 1e-3}``.
      add_optim : bool, optional
          If ``True``, adds the optimizer to the classifier object, by
          default ``True``.

      Returns
      -------
      optimizer : torch.optim.Optimizer
          The initialized optimizer. Only returned if ``add_optim`` is set to
          ``False``.

      Notes
      -----
      If ``add_optim`` is True, the optimizer will be added to object.

      Note that the first argument of an optimizer is parameters to optimize,
      e.g. ``params2optimize = model_ft.parameters()``:

      - ``model_ft.parameters()``: all parameters are being optimized
      - ``model_ft.fc.parameters()``: only parameters of final layer are being optimized

      Here, we use:

      .. code-block:: python

          filter(lambda p: p.requires_grad, self.model.parameters())


   .. py:method:: add_optimizer(optimizer)

      Add an optimizer to the classifier object.

      Parameters
      ----------
      optimizer : torch.optim.Optimizer
          The optimizer to add to the classifier object.

      Returns
      -------
      None


   .. py:method:: initialize_scheduler(scheduler_type = 'steplr', scheduler_param_dict = {'step_size': 10, 'gamma': 0.1}, add_scheduler = True)

      Initializes a learning rate scheduler for the optimizer and adds it to
      the classifier object.

      Parameters
      ----------
      scheduler_type : str, optional
          The type of learning rate scheduler to use. Can be either
          ``"steplr"`` (default) or ``"onecyclelr"``.
      scheduler_param_dict : dict, optional
          The parameters to pass to the scheduler constructor, by default
          ``{"step_size": 10, "gamma": 0.1}``.
      add_scheduler : bool, optional
          If ``True``, adds the scheduler to the classifier object, by
          default ``True``.

      Raises
      ------
      ValueError
          If the specified ``scheduler_type`` is not implemented.

      Returns
      -------
      scheduler : torch.optim.lr_scheduler._LRScheduler
          The initialized learning rate scheduler. Only returned if
          ``add_scheduler`` is set to False.


   .. py:method:: add_scheduler(scheduler)

      Add a scheduler to the classifier object.

      Parameters
      ----------
      scheduler : torch.optim.lr_scheduler._LRScheduler
          The scheduler to add to the classifier object.

      Raises
      ------
      ValueError
          If no optimizer has been set. Use ``initialize_optimizer`` or
          ``add_optimizer`` to set an optimizer first.

      Returns
      -------
      None


   .. py:method:: add_criterion(criterion)

      Add a loss criterion to the classifier object.

      Parameters
      ----------
      criterion : torch.nn.modules.loss._Loss
          The loss criterion to add to the classifier object.

      Returns
      -------
      None
          The function only modifies the ``criterion`` attribute of the
          classifier and does not return anything.


   .. py:method:: model_summary(only_trainable = False, print_space = [40, 20, 20])

      Print a summary of the model including the modules, the number of
      parameters in each module, and the dimension of the output tensor of
      each module. If ``only_trainable`` is ``True``, it only prints the
      trainable parameters.

      Other ways to check params:

      .. code-block:: python

          sum(p.numel() for p in myclassifier.model.parameters())

      .. code-block:: python

          sum(p.numel() for p in myclassifier.model.parameters()
              if p.requires_grad)

      And:

      .. code-block:: python

          for name, param in self.model.named_parameters():
              n = name.split(".")[0].split("_")[0]
              print(name, param.requires_grad)

      Parameters
      ----------
      only_trainable : bool, optional
          If ``True``, only the trainable parameters will be printed.
          Defaults to ``False``.
      print_space : list, optional
          A list with three integers defining the width of each column in
          the printed table. By default, ``[40, 20, 20]``.

      Returns
      -------
      None

      Notes
      -----
      Credit: this function is the modified version of
      https://stackoverflow.com/a/62508086.


   .. py:method:: freeze_layers(layers_to_freeze = [])

      Freezes the specified layers in the neural network by setting
      ``requires_grad`` attribute to False for their parameters.

      Parameters
      ----------
      layers_to_freeze : list of str, optional
          List of names of the layers to freeze. If a layer name ends with
          an asterisk (``"*"``), then all parameters whose name contains the
          layer name (excluding the asterisk) are frozen. Otherwise,
          only the parameters with an exact match to the layer name
          are frozen. By default, ``[]``.

      Returns
      -------
      None
          The function only modifies the ``requires_grad`` attribute of the
          specified parameters and does not return anything.

      Notes
      -----
      Wildcards are accepted in the ``layers_to_freeze`` parameter.


   .. py:method:: unfreeze_layers(layers_to_unfreeze = [])

      Unfreezes the specified layers in the neural network by setting
      ``requires_grad`` attribute to True for their parameters.

      Parameters
      ----------
      layers_to_unfreeze : list of str, optional
          List of names of the layers to unfreeze. If a layer name ends with
          an asterisk (``"*"``), then all parameters whose name contains the
          layer name (excluding the asterisk) are unfrozen. Otherwise,
          only the parameters with an exact match to the layer name
          are unfrozen. By default, ``[]``.

      Returns
      -------
      None
          The function only modifies the ``requires_grad`` attribute of the
          specified parameters and does not return anything.

      Notes
      -----
      Wildcards are accepted in the ``layers_to_unfreeze`` parameter.


   .. py:method:: only_keep_layers(only_keep_layers_list = [])

      Only keep the specified layers (``only_keep_layers_list``) for
      gradient computation during the backpropagation.

      Parameters
      ----------
      only_keep_layers_list : list, optional
          List of layer names to keep. All other layers will have their
          gradient computation turned off. Default is ``[]``.

      Returns
      -------
      None
          The function only modifies the ``requires_grad`` attribute of the
          specified parameters and does not return anything.


   .. py:method:: inference(set_name = 'infer', verbosity_level = 0, print_info_batch_freq = 5)

      Run inference on a specified dataset (``set_name``).

      Parameters
      ----------
      set_name : str, optional
          The name of the dataset to run inference on, by default
          ``"infer"``.
      verbosity_level : int, optional
          The verbosity level of the output messages, by default ``0``.
      print_info_batch_freq : int, optional
          The frequency of printouts, by default ``5``.

      Returns
      -------
      None

      Notes
      -----
      This method calls the
      :meth:`mapreader.train.classifier.classifier.train` method with the
      ``num_epochs`` set to ``1`` and all the other parameters specified in
      the function arguments.


   .. py:method:: train_component_summary()

      Print a summary of the optimizer, criterion and trainable model
      components.

      Returns:
      --------
      None


   .. py:method:: train(phases = ['train', 'val'], num_epochs = 25, save_model_dir = 'models', verbosity_level = 1, tensorboard_path = None, tmp_file_save_freq = 2, remove_after_load = True, print_info_batch_freq = 5)

      Train the model on the specified phases for a given number of epochs.

      Wrapper function for
      :meth:`mapreader.train.classifier.classifier.train_core` method to
      capture exceptions (``KeyboardInterrupt`` is the only supported
      exception currently).

      Parameters
      ----------
      phases : list of str, optional
          The phases to train the model on for each epoch. Default is
          ``["train", "val"]``.
      num_epochs : int, optional
          The number of epochs to train the model for. Default is ``25``.
      save_model_dir : str or None, optional
          The directory to save the model in. Default is ``"models"``. If
          set to ``None``, the model is not saved.
      verbosity_level : int, optional
          The level of verbosity during training:

          - ``0`` is silent,
          - ``1`` is progress bar and metrics,
          - ``2`` is detailed information.

          Default is ``1``.
      tensorboard_path : str or None, optional
          The path to the directory to save TensorBoard logs in. If set to
          ``None``, no TensorBoard logs are saved. Default is ``None``.
      tmp_file_save_freq : int, optional
          The frequency (in epochs) to save a temporary file of the model.
          Default is ``2``. If set to ``0`` or ``None``, no temporary file
          is saved.
      remove_after_load : bool, optional
          Whether to remove the temporary file after loading it. Default is
          ``True``.
      print_info_batch_freq : int, optional
          The frequency (in batches) to print training information. Default
          is ``5``. If set to ``0`` or ``None``, no training information is
          printed.

      Returns
      -------
      None
          The function saves the model to the ``save_model_dir`` directory,
          and optionally to a temporary file. If interrupted with a
          ``KeyboardInterrupt``, the function tries to load the temporary
          file. If no temporary file is found, it continues without loading.

      Notes
      -----
      Refer to the documentation of
      :meth:`mapreader.train.classifier.classifier.train_core` for more
      information.


   .. py:method:: train_core(phases = ['train', 'val'], num_epochs = 25, save_model_dir = 'models', verbosity_level = 1, tensorboard_path = None, tmp_file_save_freq = 2, print_info_batch_freq = 5)

      Trains/fine-tunes a classifier for the specified number of epochs on
      the given phases using the specified hyperparameters.

      Parameters
      ----------
      phases : list of str, optional
          The phases to train the model on for each epoch. Default is
          ``["train", "val"]``.
      num_epochs : int, optional
          The number of epochs to train the model for. Default is ``25``.
      save_model_dir : str or None, optional
          The directory to save the model in. Default is ``"models"``. If
          set to ``None``, the model is not saved.
      verbosity_level : int, optional
          The level of verbosity during training:

          - ``0`` is silent,
          - ``1`` is progress bar and metrics,
          - ``2`` is detailed information.

          Default is ``1``.
      tensorboard_path : str or None, optional
          The path to the directory to save TensorBoard logs in. If set to
          ``None``, no TensorBoard logs are saved. Default is ``None``.
      tmp_file_save_freq : int, optional
          The frequency (in epochs) to save a temporary file of the model.
          Default is ``2``. If set to ``0`` or ``None``, no temporary file
          is saved.
      print_info_batch_freq : int, optional
          The frequency (in batches) to print training information. Default
          is ``5``. If set to ``0`` or ``None``, no training information is
          printed.

      Raises
      ------
      ValueError
          If the criterion is not set. Use the ``add_criterion`` method to
          set the criterion.

          If the optimizer is not set and the phase is "train". Use the
          ``initialize_optimizer`` or ``add_optimizer`` method to set the
          optimizer.

      KeyError
          If the specified phase cannot be found in the keys of the object's
          ``dataloader`` dictionary property.

      Returns
      -------
      None


   .. py:method:: calculate_add_metrics(y_true, y_pred, y_score, phase, epoch = -1, tboard_writer=None)

      Calculate and add metrics to the classifier's metrics dictionary.

      Parameters
      ----------
      y_true : array-like of shape (n_samples,)
          True binary labels or multiclass labels. Can be considered ground
          truth or (correct) target values.

      y_pred : array-like of shape (n_samples,)
          Predicted binary labels or multiclass labels. The estimated
          targets as returned by a classifier.

      y_score : array-like of shape (n_samples, n_classes)
          Predicted probabilities for each class. Only required when
          ``y_pred`` is not binary.

      phase : str
          Name of the current phase, typically ``"train"`` or ``"val"``. See
          ``train`` function.

      epoch : int, optional
          Current epoch number. Default is ``-1``.

      tboard_writer : object, optional
          TensorBoard SummaryWriter object to write the metrics. Default is
          ``None``.

      Returns
      -------
      None

      Notes
      -----
      This method uses both the
      ``sklearn.metrics.precision_recall_fscore_support`` and
      ``sklearn.metrics.roc_auc_score`` functions from ``scikit-learn`` to
      calculate the metrics for each average type (``"micro"``, ``"macro"``
      and ``"weighted"``). The results are then added to the ``metrics``
      dictionary. It also writes the metrics to the TensorBoard
      SummaryWriter, if ``tboard_writer`` is not None.


   .. py:method:: gen_epoch_msg(phase, epoch_msg)

      Generates a log message for an epoch during training or validation.
      The message includes information about the loss, F-score, and recall
      for a given phase (training or validation).

      Parameters
      ----------
      phase : str
          The training phase, either ``"train"`` or ``"val"``.
      epoch_msg : str
          The message string to be modified with the epoch metrics.

      Returns
      -------
      epoch_msg : str
          The updated message string with the epoch metrics.


   .. py:method:: plot_metric(y_axis, y_label, legends, x_axis = 'epoch', x_label = 'epoch', colors = 5 * ['k', 'tab:red'], styles = 10 * ['-'], markers = 10 * ['o'], figsize = (10, 5), plt_yrange = None, plt_xrange = None)

      Plot the metrics of the classifier object.

      Parameters
      ----------
      y_axis : list of str
          A list of metric names to be plotted on the y-axis.
      y_label : str
          The label for the y-axis.
      legends : list of str
          The legend labels for each metric.
      x_axis : str, optional
          The metric to be used as the x-axis. Can be ``"epoch"`` (default)
          or any other metric name present in the dataset.
      x_label : str, optional
          The label for the x-axis. Defaults to ``"epoch"``.
      colors : list of str, optional
          The colors to be used for the lines of each metric. It must be at
          least the same size as ``y_axis``. Defaults to
          ``5 * ["k", "tab:red"]``.
      styles : list of str, optional
          The line styles to be used for the lines of each metric. It must
          be at least the same size as ``y_axis``. Defaults to
          ``10 * ["-"]``.
      markers : list of str, optional
          The markers to be used for the lines of each metric. It must be at
          least the same size as ``y_axis``. Defaults to ``10 * ["o"]``.
      figsize : tuple of int, optional
          The size of the figure in inches. Defaults to ``(10, 5)``.
      plt_yrange : tuple of float, optional
          The range of values for the y-axis. Defaults to ``None``.
      plt_xrange : tuple of float, optional
          The range of values for the x-axis. Defaults to ``None``.

      Returns
      -------
      None

      Notes
      -----
      This function requires the ``matplotlib`` package.


   .. py:method:: initialize_model(model_name, pretrained = True, last_layer_num_classes = 'default', add_model = True)

      Initializes a PyTorch model with the option to change the number of
      classes in the last layer (``last_layer_num_classes``).

      The function handles six PyTorch models: ResNet, AlexNet, VGG,
      SqueezeNet, DenseNet, and Inception v3.

      Parameters
      ----------
      model_name : str
          Name of a PyTorch model. See
          https://pytorch.org/vision/0.8/models.html
      pretrained : bool, optional
          Use pretrained version, by default ``True``
      last_layer_num_classes : str or int, optional
          Number of elements in the last layer. If ``"default"``, sets it to
          the number of classes. By default, ``"default"``.
      add_model : bool, optional
          If ``True`` (default), adds the initialized model to the instance
          of the class.

      Returns
      -------
      model : PyTorch model
          The initialized PyTorch model with the changed last layer.
      input_size : int
          Input size of the model.
      is_inception : bool
          True if the model is Inception v3.

      Raises
      ------
      ValueError
          If an invalid model name is passed.

      Notes
      -----
      Inception v3 requires the input size to be ``(299, 299)``, whereas all
      of the other models expect ``(224, 224)``.

      See https://pytorch.org/vision/0.8/models.html for available models.


   .. py:method:: show_sample(set_name = 'train', batch_number = 1, print_batch_info = True, figsize = (15, 10))

      Displays a sample of training or validation data in a grid format with
      their corresponding class labels.

      Parameters
      ----------
      set_name : str, optional
          Name of the dataset (``"train"``/``"validation"``) to display the
          sample from, by default ``"train"``.
      batch_number : int, optional
          Number of batches to display, by default ``1``.
      print_batch_info : bool, optional
          Whether to print information about the batch size, by default
          ``True``.
      figsize : tuple, optional
          Figure size (width, height) in inches, by default ``(15, 10)``.

      Returns
      -------
      None
          Displays the sample images with their corresponding class labels.

      Raises
      ------
      StopIteration
          If the specified number of batches to display exceeds the total
          number of batches in the dataset.

      Notes
      -----
      This method uses the dataloader of the ``ImageClassifierData`` class
      and the ``torchvision.utils.make_grid`` function to display the sample
      data in a grid format. It also calls the ``_imshow`` method of the
      ``ImageClassifierData`` class to show the sample data.


   .. py:method:: batch_info(set_name = 'train')

      Print information about a dataset's batches, samples, and batch-size.

      Parameters
      ----------
      set_name : str, optional
          Name of the dataset to display batch information for (default is
          ``"train"``).

      Returns
      -------
      None


   .. py:method:: inference_sample_results(num_samples = 6, class_index = 0, set_name = 'train', min_conf = None, max_conf = None, figsize = (15, 15))

      Performs inference on a given dataset and displays results for a
      specified class.

      Parameters
      ----------
      num_samples : int, optional
          The number of sample results to display. Defaults to ``6``.
      class_index : int, optional
          The index of the class for which to display results. Defaults to
          ``0``.
      set_name : str, optional
          The name of the dataset split to use for inference. Defaults to
          ``"train"``.
      min_conf : float, optional
          The minimum confidence score for a sample result to be displayed.
          Samples with lower confidence scores will be skipped. Defaults to
          ``None``.
      max_conf : float, optional
          The maximum confidence score for a sample result to be displayed.
          Samples with higher confidence scores will be skipped. Defaults to
          ``None``.
      figsize : tuple[int, int], optional
          Figure size (width, height) in inches, displaying the sample
          results. Defaults to ``(15, 15)``.

      Returns
      -------
      None


   .. py:method:: save(save_path = 'default.obj', force = False)

      Save the object to a file.

      Parameters
      ----------
      save_path : str, optional
          The path to the file to write. If the file already exists and
          ``force`` is not ``True``, a ``FileExistsError`` is raised.
          Defaults to ``"default.obj"``.
      force : bool, optional
          Whether to overwrite the file if it already exists. Defaults to
          ``False``.

      Raises
      ------
      FileExistsError
          If the file already exists and ``force`` is not ``True``.

      Notes
      -----
      The object is saved in two parts. First, a serialized copy of the
      object's dictionary is written to the specified file using the
      ``joblib.dump`` function. The object's ``model`` attribute is excluded
      from this dictionary and saved separately using the ``torch.save``
      function, with a filename derived from the original ``save_path``.


   .. py:method:: load(load_path, remove_after_load = False, force_device = False)

      This function loads the state of a class instance from a saved file
      using the joblib library. It also loads a PyTorch model from a
      separate file and maps it to the device used to load the class
      instance.

      Parameters
      ----------
      load_path : str
          Path to the saved file to load.
      remove_after_load : bool, optional
          Whether to remove the saved file after loading. Defaults to
          ``False``.
      force_device : bool or str, optional
          Whether to force the use of a specific device, or the name of the
          device to use. If set to ``True``, the default device is used.
          Defaults to ``False``.

      Raises
      ------
      FileNotFoundError
          If the specified file does not exist.

      Modifies
      ----------
      self.__dict__ : dict
          The state of the class instance is updated with the contents of
          the saved file.
      os.environ["CUDA_VISIBLE_DEVICES"] : str
          The CUDA_VISIBLE_DEVICES environment variable is updated if the
          ``force_device`` argument is specified.

      Returns
      -------
      None


   .. py:method:: get_time()

      Get the current date and time as a formatted string.

      Returns
      -------
      str
          A string representing the current date and time.


   .. py:method:: cprint(type_info, bc_color, text)

      Print colored text with additional information.

      Parameters
      ----------
      type_info : str
          The type of message to display.
      bc_color : str
          The color to use for the message text.
      text : str
          The text to display.

      Returns
      -------
      None
          The colored message is displayed on the standard output stream.


   .. py:method:: update_progress(progress, text = '', barLength = 30)

      Update the progress bar.

      Parameters
      ----------
      progress : float or int
          The progress value to display, between ``0`` and ``1``.
          If an integer is provided, it will be converted to a float.
          If a value outside the range ``[0, 1]`` is provided, it will be
          clamped to the nearest valid value.
      text : str, optional
          Additional text to display after the progress bar, defaults to
          ``""``.
      barLength : int, optional
          The length of the progress bar in characters, defaults to ``30``.

      Raises
      ------
      TypeError
          If progress is not a floating point value or an integer.

      Returns
      -------
      None
          The progress bar is displayed on the standard output stream.



.. py:class:: classifierContext(device = 'default')

   Bases: :py:obj:`mapreader.learn.classifier.classifier`

   .. py:method:: train(phases = ['train', 'val'], num_epochs = 25, save_model_dir = 'models', verbosity_level = 1, tensorboard_path = None, tmp_file_save_freq = 2, remove_after_load = True, print_info_batch_freq = 5)

      Train the model on the specified phases for a given number of epochs.
      Wrapper function for ``train_core`` method to capture exceptions (with
      supported exceptions so far: ``KeyboardInterrupt``). Refer to
      ``train_core`` for more information.

      Parameters
      ----------
      phases : list of str, optional
          The phases to train the model on for each epoch. Default is
          ``["train", "val"]``.
      num_epochs : int, optional
          The number of epochs to train the model for. Default is ``25``.
      save_model_dir : str or None, optional
          The directory to save the model in. Default is ``"models"``. If
          set to ``None``, the model is not saved.
      verbosity_level : int, optional
          The level of verbosity during training:

          - ``0`` is silent,
          - ``1`` is progress bar and metrics,
          - ``2`` is detailed information.

          Default is ``1``.
      tensorboard_path : str or None, optional
          The path to the directory to save TensorBoard logs in. If set to
          ``None``, no TensorBoard logs are saved. Default is ``None``.
      tmp_file_save_freq : int, optional
          The frequency (in epochs) to save a temporary file of the model.
          Default is ``2``. If set to ``0`` or ``None``, no temporary file
          is saved.
      remove_after_load : bool, optional
          Whether to remove the temporary file after loading it. Default is
          ``True``.
      print_info_batch_freq : int, optional
          The frequency (in batches) to print training information. Default
          is ``5``. If set to ``0`` or ``None``, no training information is
          printed.

      Returns
      -------
      None
          The function saves the model to the ``save_model_dir`` directory,
          and optionally to a temporary file. If interrupted with a
          ``KeyboardInterrupt``, the function tries to load the temporary
          file. If no temporary file is found, it continues without loading.


   .. py:method:: train_core(phases = ['train', 'val'], num_epochs = 25, save_model_dir = 'models', verbosity_level = 1, tensorboard_path = None, tmp_file_save_freq = 2, print_info_batch_freq = 5)

      Trains/fine-tunes a classifier for the specified number of epochs on
      the given phases using the specified hyperparameters.

      Parameters
      ----------
      phases : list of str, optional
          The phases to train the model on for each epoch. Default is
          ``["train", "val"]``.
      num_epochs : int, optional
          The number of epochs to train the model for. Default is ``25``.
      save_model_dir : str or None, optional
          The directory to save the model in. Default is ``"models"``. If
          set to ``None``, the model is not saved.
      verbosity_level : int, optional
          The level of verbosity during training:

          - ``0`` is silent,
          - ``1`` is progress bar and metrics,
          - ``2`` is detailed information.

          Default is ``1``.
      tensorboard_path : str or None, optional
          The path to the directory to save TensorBoard logs in. If set to
          ``None``, no TensorBoard logs are saved. Default is ``None``.
      tmp_file_save_freq : int, optional
          The frequency (in epochs) to save a temporary file of the model.
          Default is ``2``. If set to ``0`` or ``None``, no temporary file
          is saved.
      print_info_batch_freq : int, optional
          The frequency (in batches) to print training information. Default
          is ``5``. If set to ``0`` or ``None``, no training information is
          printed.

      Raises
      ------
      ValueError
          If the criterion is not set. Use the ``add_criterion`` method to
          set the criterion.

          If the optimizer is not set and the phase is "train". Use the
          ``initialize_optimizer`` or ``add_optimizer`` method to set the
          optimizer.

      KeyError
          If the specified phase cannot be found in the object's dataloader
          with keys.

      Returns
      -------
      None


   .. py:method:: show_sample(set_name = 'train', batch_number = 1, print_batch_info = True, figsize = (15, 10))

      Displays a sample of training or validation data in a grid format with
      their corresponding class labels.

      Parameters
      ----------
      set_name : str, optional
          Name of the dataset (``train``/``validation``) to display the
          sample from, by default ``"train"``.
      batch_number : int, optional
          Number of batches to display, by default ``1``.
      print_batch_info : bool, optional
          Whether to print information about the batch size, by default
          ``True``.
      figsize : tuple, optional
          Figure size (width, height) in inches, by default ``(15, 10)``.

      Returns
      -------
      None
          Displays the sample images with their corresponding class labels.

      Raises
      ------
      StopIteration
          If the specified number of batches to display exceeds the total
          number of batches in the dataset.

      Notes
      -----
      This method uses the dataloader of the ``ImageClassifierData`` class
      and the ``torchvision.utils.make_grid`` function to display the sample
      data in a grid format. It also calls the ``_imshow`` method of the
      ``ImageClassifierData`` class to show the sample data.


   .. py:method:: layerwise_lr(min_lr, max_lr, ltype = 'linspace', sep_group_names = ['features1', 'features2'])

      Calculates layer-wise learning rates for a given set of model
      parameters.

      Parameters
      ----------
      min_lr : float
          The minimum learning rate to be used.
      max_lr : float
          The maximum learning rate to be used.
      ltype : str, optional
          The type of sequence to use for spacing the specified interval
          learning rates. Can be either ``"linspace"`` or ``"geomspace"``,
          where `"linspace"` uses evenly spaced learning rates over a
          specified interval and `"geomspace"` uses learning rates spaced
          evenly on a log scale (a geometric progression). Defaults to
          ``"linspace"``.
      sep_group_names : list, optional
          A list of strings containing the names of parameter groups. Layers
          belonging to each group will be assigned the same learning rate.
          Defaults to ``["features1", "features2"]``.

      Returns
      -------
      list of dicts
          A list of dictionaries containing the parameters and learning
          rates for each layer.


   .. py:method:: inference_sample_results(num_samples = 6, class_index = 0, set_name = 'train', min_conf = None, max_conf = None, figsize = (15, 15))

      Performs inference on a given dataset and displays results for a
      specified class.

      Parameters
      ----------
      num_samples : int, optional
          The number of sample results to display. Defaults to ``6``.
      class_index : int, optional
          The index of the class for which to display results. Defaults to
          ``0``.
      set_name : str, optional
          The name of the dataset split to use for inference. Defaults to
          ``"train"``.
      min_conf : float, optional
          The minimum confidence score for a sample result to be displayed.
          Samples with lower confidence scores will be skipped. Defaults to
          ``None``.
      max_conf : float, optional
          The maximum confidence score for a sample result to be displayed.
          Samples with higher confidence scores will be skipped. Defaults to
          ``None``.
      figsize : tuple[int, int], optional
          Figure size (width, height) in inches, displaying the sample
          results. Defaults to ``(15, 15)``.

      Returns
      -------
      None




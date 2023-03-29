:py:mod:`mapreader.loader`
==========================

.. py:module:: mapreader.loader


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   images/index.rst
   loader/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   mapreader.loader.mapImages



Functions
~~~~~~~~~

.. autoapisummary::

   mapreader.loader.loader
   mapreader.loader.load_patches



.. py:class:: mapImages(path_images = None, tree_level = 'parent', parent_path = None, **kwds)

   Class to manage a collection of image paths and construct image objects.

   Parameters
   ----------
   path_images : str or None, optional
       Path to the directory containing images (accepts wildcards). By
       default, ``False``
   tree_level : str, optional
       Level of the image hierarchy to construct. The value can be
       ``"parent"`` (default) and ``"child"``.
   parent_path : str, optional
       Path to parent images (if applicable), by default ``None``.
   **kwds : dict, optional
       Additional keyword arguments to be passed to the ``imagesConstructor``
       method.

   Attributes
   ----------
   path_images : list
       List of paths to the image files.
   images : dict
       A dictionary containing the constructed image data. It has two levels
       of hierarchy, ``"parent"`` and ``"child"``, depending on the value of
       the ``tree_level`` parameter.

   .. py:method:: imagesConstructor(image_path, parent_path = None, tree_level = 'child', **kwds)

      Constructs image data from the given image path and parent path and
      adds it to the ``mapImages`` instance's ``images`` attribute.

      Parameters
      ----------
      image_path : str
          Path to the image file.
      parent_path : str, optional
          Path to the parent image (if applicable), by default ``None``.
      tree_level : str, optional
          Level of the image hierarchy to construct, either ``"child"``
          (default) or ``"parent"``.
      **kwds : dict, optional
          Additional keyword arguments to be included in the constructed
          image data.

      Returns
      -------
      None

      Raises
      ------
      ValueError
          If ``tree_level`` is not set to ``"parent"`` or ``"child"``.

          If ``tree_level`` is set to ``"parent"`` and ``parent_path`` is
          not ``None``.

      Notes
      -----
      This method assumes that the ``images`` attribute has been initialized
      on the mapImages instance as a dictionary with two levels of hierarchy,
      ``"parent"`` and ``"child"``. The image data is added to the
      corresponding level based on the value of ``tree_level``.


   .. py:method:: splitImagePath(inp_path)
      :staticmethod:

      Split the input path into basename and dirname.

      Parameters
      ----------
      inp_path : str
          Input path to split.

      Returns
      -------
      tuple
          A tuple containing the basename and dirname of the input path.


   .. py:method:: add_metadata(metadata, columns = None, tree_level = 'parent', index_col = 0, delimiter = '|')

      Add metadata information to the images dictionary.

      Parameters
      ----------
      metadata : str or pandas.DataFrame
          A csv file path (normally created from a pandas DataFrame) or a
          pandas DataFrame that contains the metadata information.
      columns : list, optional
          List of columns to use, by default ``None``.
      tree_level : str, optional
          Determines which images dictionary (``"parent"`` or ``"child"``)
          to add the metadata to, by default ``"parent"``.
      index_col : int, optional
          Column to use as the index when reading the csv file into a pandas
          DataFrame, by default ``0``.

          Needs only be provided if a csv file path is provided as
          the ``metadata`` parameter.
      delimiter : str, optional
          Delimiter to use for reading the csv file into a pandas DataFrame,
          by default ``"|"``.

          Needs only be provided if a csv file path is provided as
          the ``metadata`` parameter.

      Raises
      ------
      ValueError
          If metadata is not a pandas DataFrame or a csv file path.

          If 'name' or 'image_id' is not one of the columns in the metadata.

      Returns
      -------
      None


   .. py:method:: show_sample(num_samples, tree_level = 'parent', random_seed = 65, **kwds)

      Display a sample of images from a particular level in the image
      hierarchy.

      Parameters
      ----------
      num_samples : int
          The number of images to display.
      tree_level : str, optional
          The level of the hierarchy to display images from, which can be
          ``"child"`` or ``"parent"`` (default).
      random_seed : int, optional
          The random seed to use for reproducibility. Default is ``65``.
      **kwds : dict, optional
          Additional keyword arguments to pass to
          ``matplotlib.pyplot.figure()``.

      Returns
      -------
      None


   .. py:method:: list_parents()

      Return list of all parents


   .. py:method:: list_children()

      Return list of all children


   .. py:method:: add_shape(tree_level = 'parent')

      Add a shape to each image in the specified level of the image
      hierarchy.

      Parameters
      ----------
      tree_level : str, optional
          The level of the hierarchy to add shapes to, either ``"parent"``
          (default) or ``"child"``.

      Returns
      -------
      None

      Notes
      -----
      The method runs :meth:`mapreader.loader.images.mapImages.add_shape_id`
      for each image present at the ``tree_level`` provided.


   .. py:method:: add_coord_increments()

      Adds coordinate increments to each image at the parent level.

      Parameters
      ----------
      None

      Returns
      -------
      None

      Notes
      -----
      The method runs
      :meth:`mapreader.loader.images.mapImages.add_coord_increments_id`
      for each image present at the parent level, which calculates
      pixel-wise delta longitute (``dlon``) and delta latititude (``dlat``)
      for the image and adds the data to it.


   .. py:method:: add_center_coord(tree_level = 'child')

      Adds center coordinates to each image at the specified tree level.

      Parameters
      ----------
      tree_level: str, optional
          The tree level where the center coordinates will be added. It can
          be either ``"parent"`` or ``"child"`` (default).

      Returns
      -------
      None

      Notes
      -----
      The method runs
      :meth:`mapreader.loader.images.mapImages.add_center_coord_id`
      for each image present at the ``tree_level`` provided, which calculates
      central longitude and latitude (``center_lon`` and ``center_lat``) for
      the image and adds the data to it.


   .. py:method:: add_shape_id(image_id, tree_level = 'parent')

      Add shape (image_height, image_width, image_channels) of the image
      with specified ``image_id`` in the given ``tree_level`` to the
      metadata.

      Parameters
      ----------
      image_id : int or str
          The ID of the image to add shape metadata to.
      tree_level : str, optional
          The tree level where the image is located, which can be
          ``"parent"`` (default) or ``"child"``.

      Returns
      -------
      None
          This method does not return anything. It modifies the metadata of
          the ``images`` property in-place.

      Notes
      -----
      The shape of the image is obtained by loading the image from its
      ``image_path`` value and getting its shape.


   .. py:method:: add_coord_increments_id(image_id, verbose = False)

      Add pixel-wise delta longitute (``dlon``) and delta latititude
      (``dlat``) to the metadata of the image with the specified ``image_id``
      in the parent tree level.

      Parameters
      ----------
      image_id : int or str
          The ID of the image to add coordinate increments metadata to.
      verbose : bool, optional
          Whether to print warning messages when coordinate or shape
          metadata cannot be found. Default is ``False``.

      Returns
      -------
      None
          This method does not return anything. It modifies the metadata of
          the image in-place.

      Notes
      -----
      Coordinate increments (dlon and dlat) are calculated using the
      following formula:

      .. code-block:: python

          dlon = abs(lon_max - lon_min) / image_width
          dlat = abs(lat_max - lat_min) / image_height

      ``lon_max``, ``lon_min``, ``lat_max``, ``lat_min`` are the coordinate
      bounds of the image, and ``image_width`` and ``image_height`` are the
      width and height of the image in pixels respectively.

      This method assumes that the coordinate and shape metadata of the
      image have already been added to the metadata.

      If the coordinate metadata cannot be found, a warning message will be
      printed if ``verbose=True``.

      If the shape metadata cannot be found, this method will call the
      :meth:`mapreader.loader.images.mapImages.add_shape_id` method to add
      it.


   .. py:method:: add_center_coord_id(image_id, tree_level = 'child', verbose = False)

      Calculates and adds center coordinates (longitude as ``center_lon``
      and latitude as ``center_lat``) to a given image patch.

      Parameters
      ----------
      image_id : int or str
          The ID of the image patch to add center coordinates to.
      tree_level : str, optional
          The level of the image patch in the image hierarchy, either
          ``"parent"`` or ``"child"`` (default).
      verbose : bool, optional
          Whether to print warning messages or not. Defaults to ``False``.

      Raises
      ------
      NotImplementedError
          If ``tree_level`` is not set to ``"parent"`` or ``"child"``.

      Returns
      -------
      None


   .. py:method:: calc_pixel_width_height(parent_id, calc_size_in_m = 'great-circle', verbose = False)

      Calculate the width and height of each pixel in a given image in
      meters.

      Parameters
      ----------
      parent_id : int or str
          The ID of the parent image to calculate pixel size.
      calc_size_in_m : str, optional
          Method to use for calculating image size in meters.
          Possible values: ``"great-circle"`` (default), ``"gc"`` (alias for
          ``"great-circle"``), ``"geodesic"``. ``"great-circle"`` and
          ``"gc"`` compute size using the great-circle distance formula,
          while ``"geodesic"`` computes size using the geodesic distance
          formula.
      verbose : bool, optional
          If ``True``, print additional information during the calculation.
          Default is ``False``.

      Returns
      -------
      tuple of floats
          The size of the image in meters as a tuple of bottom, top, left,
          and right distances (in that order).

      Notes
      -----
      This method requires the parent image to have location metadata added
      with either the :meth:`mapreader.loader.images.mapImages.add_metadata`
      or :meth:`mapreader.loader.images.mapImages.addGeoInfo` methods.

      The calculations are performed using the ``geopy.distance.geodesic``
      and ``geopy.distance.great_circle`` methods. Thus, the method requires
      the ``geopy`` package to be installed.


   .. py:method:: sliceAll(method = 'pixel', slice_size = 100, path_save = 'sliced_images', square_cuts = False, resize_factor = False, output_format = 'png', rewrite = False, verbose = False, tree_level = 'parent', add2child = True, id1 = 0, id2 = -1)

      Slice all images in the specified ``tree_level`` and add the sliced
      images to the mapImages instance's ``images`` dictionary.

      Parameters
      ----------
      method : str, optional
          Method used to slice images, choices between ``"pixel"`` (default)
          and ``"meters"`` or ``"meter"``.
      slice_size : int, optional
          Number of pixels/meters in both x and y to use for slicing, by
          default ``100``.
      path_save : str, optional
          Directory to save the sliced images, by default
          ``"sliced_images"``.
      square_cuts : bool, optional
          If True, all sliced images will have the same number of pixels in
          x and y, by default ``False``.
      resize_factor : bool, optional
          If True, resize the images before slicing, by default ``False``.
      output_format : str, optional
          Format to use when writing image files, by default ``"png"``.
      rewrite : bool, optional
          If True, existing slices will be rewritten, by default ``False``.
      verbose : bool, optional
          If True, progress updates will be printed throughout, by default
          ``False``.
      tree_level : str, optional
          Tree level, choices between ``"parent"`` or ``"child``, by default
          ``"parent"``.
      add2child : bool, optional
          If True, sliced images will be added to the mapImages instance's
          ``images`` dictionary, by default ``True``.
      id1 : int, optional
          The start index of the images to slice. Default is ``0``.
      id2 : int, optional
          The end index of the images to slice. Default is ``-1`` (i.e., all
          images after index ``id1`` will be sliced).

      Raises
      ------
      ValueError
          If ``id2 < id1``.

      Returns
      -------
      None


   .. py:method:: addChildren()

      Add children to parent.

      Returns
      -------
      None

      Notes
      -----
      This method adds children to their corresponding parent image. It
      checks if the parent image has any child image, and if not, it creates
      a list of children and assigns it to the parent. If the parent image
      already has a list of children, the method checks if the current child
      is already in the list. If not, the child is added to the list.


   .. py:method:: calc_pixel_stats(parent_id = None, calc_mean = True, calc_std = True)

      Calculate the mean and standard deviation of pixel values for all
      channels (R, G, B, RGB and, if present, Alpha) of all child images of
      a given parent image. Store the results in the mapImages instance's
      ``images`` dictionary.

      Parameters
      ----------
      parent_id : str, int, or None, optional
          The ID of the parent image to calculate pixel stats for. If
          ``None``, calculate pixel stats for all parent images.
      calc_mean : bool, optional
          Whether to calculate mean pixel values. Default is ``True``.
      calc_std : bool, optional
          Whether to calculate standard deviation of pixel values. Default
          is ``True``.

      Returns
      -------
      None

      Notes
      -----
      - Pixel stats are calculated for child images of the parent image
        specified by ``parent_id``.
      - If ``parent_id`` is ``None``, pixel stats are calculated for all
        parent images in the object.
      - If mean or standard deviation of pixel values has already been
        calculated for a child image, the calculation is skipped.
      - Pixel stats are stored in the ``images`` attribute of the
        ``mapImages`` instance, under the ``child`` key for each child image.
      - If no children are found for a parent image, a warning message is
        displayed and the method moves on to the next parent image.


   .. py:method:: convertImages()

      Convert the ``mapImages`` instance's ``images`` dictionary into pandas
      DataFrames for easy manipulation.

      Returns
      -------
      tuple of two pandas DataFrames
          The method returns a tuple of two DataFrames: One for the
          ``parent`` images and one for the ``child`` images.


   .. py:method:: show_par(parent_id, value = False, **kwds)

      A wrapper method for `.show()` which plots all children of a
      specified parent (`parent_id`).

      Parameters
      ----------
      parent_id : int or str
          ID of the parent image to be plotted.
      value : list or bool, optional
          Value to be plotted on each child image, by default False.

      Returns
      -------
      None

      Raises
      ------
      KeyError
          If the parent_id is not found in the image dictionary.

      Notes
      -----
      This is a wrapper method. See the documentation of the
      :meth:`mapreader.loader.images.mapImages.show` method for more detail.


   .. py:method:: show(image_ids, value = False, plot_parent = True, border = True, border_color = 'r', vmin = 0.5, vmax = 2.5, colorbar = 'viridis', alpha = 1.0, discrete_colorbar = 256, tree_level = 'child', grid_plot = (20000, 20000), plot_histogram = True, save_kml_dir = False, image_width_resolution = None, kml_dpi_image = None, **kwds)

      Plot images from a list of `image_ids`.

      Parameters
      ----------
      image_ids : str or list
          Image ID or list of image IDs to be plotted.
      value : str, list or bool, optional
          Value to plot on child images, by default ``False``.
      plot_parent : bool, optional
          If ``True``, parent image will be plotted in background, by
          default ``True``.
      border : bool, optional
          If ``True``, a border will be placed around each child image, by
          default ``True``.
      border_color : str, optional
          The color of the border. Default is ``"r"``.
      vmin : float or list, optional
          The minimum value for the colormap. By default ``0.5``.

          If a list is provided, it must be the same length as ``image_ids``.
      vmax : float or list, optional
          The maximum value for the colormap. By default ``2.5``.

          If a list is provided, it must be the same length as ``image_ids``.
      colorbar : str or list, optional
          Colorbar used to visualise chosen ``value``, by default
          ``"viridis"``.

          If a list is provided, it must be the same length as ``image_ids``.
      alpha : float or list, optional
          Transparency level for plotting ``value`` with floating point
          values ranging from 0.0 (transparent) to 1 (opaque). By default,
          ``1.0``.

          If a list is provided, it must be the same length as ``image_ids``.
      discrete_colorbar : int or list, optional
          Number of discrete colurs to use in colorbar, by default ``256``.

          If a list is provided, it must be the same length as ``image_ids``.
      tree_level : str, optional
          The level of the image tree to be plotted. Must be either
          ``"child"`` (default) or ``"parent"``.
      grid_plot : tuple, optional
          The size of the grid (number of rows and columns) to be used to
          plot images. Later adjusted to the true min/max of all subplots.
          By default ``(20000, 20000)``.
      plot_histogram : bool, optional
          If ``True``, plot histograms of the ``value`` of images. By
          default ``True``.
      save_kml_dir : str or bool, optional
          If ``True``, save KML files of the images. If a string is provided,
          it is the path to the directory in which to save the KML files. If
          set to ``False``, no files are saved. By default ``False``.
      image_width_resolution : int or None, optional
          The pixel width to be used for plotting. If ``None``, the
          resolution is not changed. Default is ``None``.

          Note: Only relevant when ``tree_level="parent"``.
      kml_dpi_image : int or None, optional
          The resolution, in dots per inch, to create KML images when
          ``save_kml_dir`` is specified (as either ``True`` or with path).
          By default ``None``.

      Returns
      -------
      None


   .. py:method:: loadPatches(patch_paths, parent_paths = False, add_geo_par = False, clear_images = False)

      Loads patch images from the given paths and adds them to the ``images``
      dictionary in the ``mapImages`` instance.

      Parameters
      ----------
      patch_paths : str
          The file path of the patches to be loaded.

          *Note: The ``patch_paths`` parameter accepts wildcards.*
      parent_paths : str or bool, optional
          The file path of the parent images to be loaded. If set to
          ``False``, no parents are loaded. Default is ``False``.

          *Note: The ``parent_paths`` parameter accepts wildcards.*
      add_geo_par : bool, optional
          If ``True``, adds geographic information to the parent image.
          Default is ``False``.
      clear_images : bool, optional
          If ``True``, clears the images from the ``images`` dictionary
          before loading. Default is ``False``.

      Returns
      -------
      None


   .. py:method:: detectParIDfromPath(image_id, parent_delimiter = '#')
      :staticmethod:

      Detect parent IDs from ``image_id``.

      Parameters
      ----------
      image_id : int or str
          ID of child image.
      parent_delimiter : str, optional
          Delimiter used to separate parent ID when naming child image, by
          default ``"#"``.

      Returns
      -------
      str
          Parent ID.


   .. py:method:: detectBorderFromPath(image_id)
      :staticmethod:

      Detects borders from the path assuming child image is named using the
      following format: ``...-min_x-min_y-max_x-max_y-...``

      Parameters
      ----------
      image_id : int or str
          ID of image

      ..
          border_delimiter : str, optional
              Delimiter used to separate border values when naming child
              image, by default ``"-"``.

      Returns
      -------
      tuple of min_x, min_y, max_x, max_y
          Border (min_x, min_y, max_x, max_y) of image


   .. py:method:: loadParents(parent_paths = False, parent_ids = False, update = False, add_geo = False)

      Load parent images from file paths (``parent_paths``).

      If ``parent_paths`` is not given, only ``parent_ids``, no image path
      will be added to the images.

      Parameters
      ----------
      parent_paths : str or bool, optional
          Path to parent images, by default ``False``.
      parent_ids : list, str or bool, optional
          ID(s) of parent images. Ignored if ``parent_paths`` are specified.
          By default ``False``.
      update : bool, optional
          If ``True``, current parents will be overwritten, by default
          ``False``.
      add_geo : bool, optional
          If ``True``, geographical info will be added to parents, by
          default ``False``.

      Returns
      -------
      None


   .. py:method:: loadDataframe(parents = None, children_df = None, clear_images = True)

      Form images variable from pandas DataFrame(s).

      Parameters
      ----------
      parents : pandas.DataFrame, str or None, optional
          DataFrame containing parents or path to parents, by default
          ``None``.
      children_df : pandas.DataFrame or None, optional
          DataFrame containing children (patches), by default ``None``.
      clear_images : bool, optional
          If ``True``, clear images before reading the dataframes, by
          default ``True``.

      Returns
      -------
      None


   .. py:method:: load_csv_file(parent_path = None, child_path = None, clear_images = False, index_col_child = 0, index_col_parent = 0)

      Load CSV files containing information about parent and child images,
      and update the ``images`` attribute of the ``mapImages`` instance with
      the loaded data.

      Parameters
      ----------
      parent_path : str, optional
          Path to the CSV file containing parent image information.
      child_path : str, optional
          Path to the CSV file containing child image information.
      clear_images : bool, optional
          If True, clear all previously loaded image information before
          loading new information. Default is ``False``.
      index_col_child : int, optional
          Column to set as index for the child DataFrame, by default ``0``.
      index_col_parent : int, optional
          Column to set as index for the parent DataFrame, by default ``0``.

      Returns
      -------
      None


   .. py:method:: addGeoInfo(proj2convert = 'EPSG:4326', calc_method = 'great-circle', verbose = False)

      Add geographic information (shape, coords, reprojected to EPSG:4326,
      and size in meters) to the ``images`` attribute of the ``mapImages``
      instance from image metadata.

      Parameters
      ----------
      proj2convert : str, optional
          Projection to convert coordinates into, by default ``"EPSG:4326"``.
      calc_method : str, optional
          Method to use for calculating image size in meters. Possible
          values: ``"great-circle"`` (default), ``"gc"`` (alias for
          ``"great-circle"``), ``"geodesic"``. ``"great-circle"`` and
          ``"gc"`` compute size using the great-circle distance formula,
          while ``"geodesic"`` computes size using the geodesic distance
          formula.
      verbose : bool, optional
          Whether to print progress messages or not. The default is
          ``False``.

      Returns
      -------
      None

      Notes
      -----
      This method reads the image files specified in the ``image_path`` key
      of each dictionary in the ``parent`` dictionary.

      It then checks if the image has geographic coordinates in its metadata,
      if not it prints a warning message and skips to the next image.

      If coordinates are present, this method converts them to the specified
      projection ``proj2convert`` and calculates the size of each pixel
      based on the method specified in ``calc_method``.

      The resulting information is then added to the dictionary in the
      ``parent`` dictionary corresponding to each image.

      Note that the calculations are performed using the
      ``geopy.distance.geodesic`` and ``geopy.distance.great_circle``
      methods. Thus, the method requires the ``geopy`` package to be
      installed.



.. py:function:: loader(path_images = None, tree_level = 'parent', parent_path = None, **kwds)

   Creates a ``mapImages`` class to manage a collection of image paths and
   construct image objects.

   Parameters
   ----------
   path_images : str or None, optional
       Path to the directory containing images (accepts wildcards). By
       default, ``None``
   tree_level : str, optional
       Level of the image hierarchy to construct. The value can be
       ``"parent"`` (default) and ``"child"``.
   parent_path : str, optional
       Path to parent images (if applicable), by default ``None``.
   **kwds : dict, optional
       Additional keyword arguments to be passed to the ``imagesConstructor``
       method.

   Returns
   -------
   mapImages
       The ``mapImages`` class which can manage a collection of image paths
       and construct image objects.

   Notes
   -----
   This is a wrapper method. See the documentation of the
   :class:`mapreader.loader.images.mapImages` class for more detail.


.. py:function:: load_patches(patch_paths, parent_paths = False, add_geo_par = False, clear_images = False)

   Creates a ``mapImages`` class to manage a collection of image paths and
   construct image objects. Then loads patch images from the given paths and
   adds them to the ``images`` dictionary in the ``mapImages`` instance.

   Parameters
   ----------
   patch_paths : str
       The file path of the patches to be loaded.

       *Note: The ``patch_paths`` parameter accepts wildcards.*
   parent_paths : str or bool, optional
       The file path of the parent images to be loaded. If set to
       ``False``, no parents are loaded. Default is ``False``.

       *Note: The ``parent_paths`` parameter accepts wildcards.*
   add_geo_par : bool, optional
       If ``True``, adds geographic information to the parent image.
       Default is ``False``.
   clear_images : bool, optional
       If ``True``, clears the images from the ``images`` dictionary
       before loading. Default is ``False``.

   Returns
   -------
   mapImages
       The ``mapImages`` class which can manage a collection of image paths
       and construct image objects.

   Notes
   -----
   This is a wrapper method. See the documentation of the
   :class:`mapreader.loader.images.mapImages` class for more detail.

   This function in particular, also calls the
   :meth:`mapreader.loader.images.mapImages.loadPatches` method. Please see
   the documentation for that method for more information as well.



mapreader.load
==============

.. py:module:: mapreader.load


Submodules
----------

.. toctree::
   :maxdepth: 1

   /in-depth-resources/api/mapreader/load/geo_utils/index
   /in-depth-resources/api/mapreader/load/images/index
   /in-depth-resources/api/mapreader/load/loader/index


Classes
-------

.. autoapisummary::

   mapreader.load.MapImages


Functions
---------

.. autoapisummary::

   mapreader.load.loader
   mapreader.load.load_patches


Package Contents
----------------

.. py:class:: MapImages(path_images = None, file_ext = False, tree_level = 'parent', parent_path = None, **kwargs)

   Class to manage a collection of image paths and construct image objects.

   :param path_images: Path to the directory containing images (accepts wildcards). By
                       default, ``False``
   :type path_images: str or None, optional
   :param file_ext: The file extension of the image files to be loaded, ignored if file types are specified in ``path_images`` (e.g. with ``"./path/to/dir/*png"``).
                    By default ``False``.
   :type file_ext: str or False, optional
   :param tree_level: Level of the image hierarchy to construct. The value can be
                      ``"parent"`` (default) and ``"patch"``.
   :type tree_level: str, optional
   :param parent_path: Path to parent images (if applicable), by default ``None``.
   :type parent_path: str, optional
   :param \*\*kwargs: Additional keyword arguments to be passed to the ``_images_constructor``
                      method.
   :type \*\*kwargs: dict, optional

   .. attribute:: path_images

      List of paths to the image files.

      :type: list

   .. attribute:: images

      A dictionary containing the constructed image data. It has two levels
      of hierarchy, ``"parent"`` and ``"patch"``, depending on the value of
      the ``tree_level`` parameter.

      :type: dict


   .. py:method:: add_metadata(metadata, index_col = 0, delimiter = ',', columns = None, tree_level = 'parent', ignore_mismatch = False)

      Add metadata information to the images dictionary.

      :param metadata: Path to a ``csv`` (or similar), ``xls`` or ``xlsx`` file or a pandas DataFrame that contains the metadata information.
      :type metadata: str or pandas.DataFrame
      :param index_col: Column to use as the index when reading the file and converting into a panda.DataFrame.
                        Accepts column indices or column names.
                        By default ``0`` (first column).

                        Only used if a file path is provided as the ``metadata`` parameter.
                        Ignored if ``columns`` parameter is passed.
      :type index_col: int or str, optional
      :param delimiter: Delimiter used in the ``csv`` file, by default ``","``.

                        Only used if a ``csv`` file path is provided as
                        the ``metadata`` parameter.
      :type delimiter: str, optional
      :param columns: List of columns indices or names to add to MapImages.
                      If ``None`` is passed, all columns will be used.
                      By default ``None``.
      :type columns: list, optional
      :param tree_level: Determines which images dictionary (``"parent"`` or ``"patch"``)
                         to add the metadata to, by default ``"parent"``.
      :type tree_level: str, optional
      :param ignore_mismatch: Whether to error if metadata with mismatching information is passed.
                              By default ``False``.
      :type ignore_mismatch: bool, optional

      :raises ValueError: If metadata is not a pandas DataFrame or a ``csv``, ``xls`` or ``xlsx`` file path.

          If 'name' or 'image_id' is not one of the columns in the metadata.

      :rtype: None

      .. rubric:: Notes

      Your metadata file must contain an column which contains the image IDs (filenames) of your images.
      This should have a column name of either ``name`` or ``image_id``.

      Existing information in your ``MapImages`` object will be overwritten if there are overlapping column headings in your metadata file/dataframe.



   .. py:method:: show_sample(num_samples, tree_level = 'patch', random_seed = 65, **kwargs)

      Display a sample of images from a particular level in the image
      hierarchy.

      :param num_samples: The number of images to display.
      :type num_samples: int
      :param tree_level: The level of the hierarchy to display images from, which can be
                         ``"patch"`` or ``"parent"``. By default "patch".
      :type tree_level: str, optional
      :param random_seed: The random seed to use for reproducibility. Default is ``65``.
      :type random_seed: int, optional
      :param \*\*kwargs: Additional keyword arguments to pass to
                         ``matplotlib.pyplot.figure()``.
      :type \*\*kwargs: dict, optional

      :returns: The figure generated
      :rtype: matplotlib.Figure



   .. py:method:: list_parents()

      Return list of all parents



   .. py:method:: list_patches()

      Return list of all patches



   .. py:method:: add_shape(tree_level = 'parent')

      Add a shape to each image in the specified level of the image
      hierarchy.

      :param tree_level: The level of the hierarchy to add shapes to, either ``"parent"``
                         (default) or ``"patch"``.
      :type tree_level: str, optional

      :rtype: None

      .. rubric:: Notes

      The method runs :meth:`mapreader.load.images.MapImages._add_shape_id`
      for each image present at the ``tree_level`` provided.



   .. py:method:: add_coords_from_grid_bb(verbose = False)


   .. py:method:: add_coord_increments(verbose = False)

      Adds coordinate increments to each image at the parent level.

      :param verbose: Whether to print verbose outputs, by default ``False``.
      :type verbose: bool, optional

      :rtype: None

      .. rubric:: Notes

      The method runs
      :meth:`mapreader.load.images.MapImages._add_coord_increments_id`
      for each image present at the parent level, which calculates
      pixel-wise delta longitude (``dlon``) and delta latitude (``dlat``)
      for the image and adds the data to it.



   .. py:method:: add_patch_coords(verbose = False)

      Add coordinates to all patches in patches dictionary.

      :param verbose: Whether to print verbose outputs.
                      By default, ``False``
      :type verbose: bool, optional



   .. py:method:: add_patch_polygons(verbose = False)

      Add polygon to all patches in patches dictionary.

      :param verbose: Whether to print verbose outputs.
                      By default, ``False``
      :type verbose: bool, optional



   .. py:method:: add_center_coord(tree_level = 'patch', verbose = False)

      Adds center coordinates to each image at the specified tree level.

      :param tree_level: The tree level where the center coordinates will be added. It can
                         be either ``"parent"`` or ``"patch"`` (default).
      :type tree_level: str, optional
      :param verbose: Whether to print verbose outputs, by default ``False``.
      :type verbose: bool, optional

      :rtype: None

      .. rubric:: Notes

      The method runs
      :meth:`mapreader.load.images.MapImages._add_center_coord_id`
      for each image present at the ``tree_level`` provided, which calculates
      central longitude and latitude (``center_lon`` and ``center_lat``) for
      the image and adds the data to it.



   .. py:method:: patchify_all(method = 'pixel', patch_size = 100, tree_level = 'parent', path_save = None, add_to_parents = True, square_cuts = False, resize_factor = False, output_format = 'png', rewrite = False, verbose = False, overlap = 0)

      Patchify all images in the specified ``tree_level`` and (if ``add_to_parents=True``) add the patches to the MapImages instance's ``images`` dictionary.

      :param method: Method used to patchify images, choices between ``"pixel"`` (default)
                     and ``"meters"`` or ``"meter"``.
      :type method: str, optional
      :param patch_size: Number of pixels/meters in both x and y to use for slicing, by
                         default ``100``.
      :type patch_size: int, optional
      :param tree_level: Tree level, choices between ``"parent"`` or ``"patch``, by default
                         ``"parent"``.
      :type tree_level: str, optional
      :param path_save: Directory to save the patches.
                        If None, will be set as f"patches_{patch_size}_{method}" (e.g. "patches_100_pixel").
                        By default None.
      :type path_save: str, optional
      :param add_to_parents: If True, patches will be added to the MapImages instance's
                             ``images`` dictionary, by default ``True``.
      :type add_to_parents: bool, optional
      :param square_cuts: If True, all patches will have the same number of pixels in
                          x and y, by default ``False``.
      :type square_cuts: bool, optional
      :param resize_factor: If True, resize the images before patchifying, by default ``False``.
      :type resize_factor: bool, optional
      :param output_format: Format to use when writing image files, by default ``"png"``.
      :type output_format: str, optional
      :param rewrite: If True, existing patches will be rewritten, by default ``False``.
      :type rewrite: bool, optional
      :param verbose: If True, progress updates will be printed throughout, by default
                      ``False``.
      :type verbose: bool, optional
      :param overlap: Fractional overlap between patches, by default ``0``.
      :type overlap: int, optional

      :rtype: None



   .. py:method:: calc_pixel_stats(parent_id = None, calc_mean = True, calc_std = True, verbose = False)

      Calculate the mean and standard deviation of pixel values for all
      channels of all patches of
      a given parent image. Store the results in the MapImages instance's
      ``images`` dictionary.

      :param parent_id: The ID of the parent image to calculate pixel stats for.
                        If ``None``, calculate pixel stats for all parent images.
                        By default, None
      :type parent_id: str or None, optional
      :param calc_mean: Whether to calculate mean pixel values. By default, ``True``.
      :type calc_mean: bool, optional
      :param calc_std: Whether to calculate standard deviation of pixel values.
                       By default, ``True``.
      :type calc_std: bool, optional
      :param verbose: Whether to print verbose outputs. By default, ``False``.
      :type verbose: bool, optional

      :rtype: None

      .. rubric:: Notes

      - Pixel stats are calculated for patches of the parent image
        specified by ``parent_id``.
      - If ``parent_id`` is ``None``, pixel stats are calculated for all
        parent images in the object.
      - If mean or standard deviation of pixel values has already been
        calculated for a patch, the calculation is skipped.
      - Pixel stats are stored in the ``images`` attribute of the
        ``MapImages`` instance, under the ``patch`` key for each patch.
      - If no patches are found for a parent image, a warning message is
        displayed and the method moves on to the next parent image.



   .. py:method:: convert_images(save = False, save_format = 'csv', delimiter = ',')

      Convert the ``MapImages`` instance's ``images`` dictionary into pandas
      DataFrames for easy manipulation.

      :param save: Whether to save the dataframes as files. By default ``False``.
      :type save: bool, optional
      :param save_format: If ``save = True``, the file format to use when saving the dataframes.
                          Options of csv ("csv") or excel ("excel" or "xlsx").
                          By default, "csv".
      :type save_format: str, optional
      :param delimiter: The delimiter to use when saving the dataframe. By default ``","``.
      :type delimiter: str, optional

      :returns: The method returns a tuple of two DataFrames: One for the
                ``parent`` images and one for the ``patch`` images.
      :rtype: tuple of two pandas DataFrames



   .. py:method:: show_parent(parent_id, column_to_plot = None, **kwargs)

      A wrapper method for `.show()` which plots all patches of a
      specified parent (`parent_id`).

      :param parent_id: ID of the parent image to be plotted.
      :type parent_id: str
      :param column_to_plot: Column whose values will be plotted on patches, by default ``None``.
      :type column_to_plot: str, optional
      :param \*\*kwargs: Key words to pass to ``show`` method.
                         See help text for ``show`` for more information.
      :type \*\*kwargs: Dict

      :returns: A list of figures created by the method.
      :rtype: list

      .. rubric:: Notes

      This is a wrapper method. See the documentation of the
      :meth:`mapreader.load.images.MapImages.show` method for more detail.



   .. py:method:: show(image_ids, column_to_plot = None, figsize = (10, 10), plot_parent = True, patch_border = True, border_color = 'r', vmin = None, vmax = None, alpha = 1.0, cmap = 'viridis', discrete_cmap = 256, plot_histogram = False, save_kml_dir = False, image_width_resolution = None, kml_dpi_image = None)

      Plot images from a list of `image_ids`.

      :param image_ids: Image ID or list of image IDs to be plotted.
      :type image_ids: str or list
      :param column_to_plot: Column whose values will be plotted on patches, by default ``None``.
      :type column_to_plot: str, optional
      :param plot_parent: If ``True``, parent image will be plotted in background, by
                          default ``True``.
      :type plot_parent: bool, optional
      :param figsize: The size of the figure to be plotted. By default, ``(10,10)``.
      :type figsize: tuple, optional
      :param patch_border: If ``True``, a border will be placed around each patch, by
                           default ``True``.
      :type patch_border: bool, optional
      :param border_color: The color of the border. Default is ``"r"``.
      :type border_color: str, optional
      :param vmin: The minimum value for the colormap.
                   If ``None``, will be set to minimum value in ``column_to_plot``, by default ``None``.
      :type vmin: float, optional
      :param vmax: The maximum value for the colormap.
                   If ``None``, will be set to the maximum value in ``column_to_plot``, by default ``None``.
      :type vmax: float, optional
      :param alpha: Transparency level for plotting ``value`` with floating point
                    values ranging from 0.0 (transparent) to 1 (opaque), by default ``1.0``.
      :type alpha: float, optional
      :param cmap: Color map used to visualize chosen ``column_to_plot`` values, by default ``"viridis"``.
      :type cmap: str, optional
      :param discrete_cmap: Number of discrete colors to use in color map, by default ``256``.
      :type discrete_cmap: int, optional
      :param plot_histogram: If ``True``, plot histograms of the ``value`` of images. By default ``False``.
      :type plot_histogram: bool, optional
      :param save_kml_dir: If ``True``, save KML files of the images. If a string is provided,
                           it is the path to the directory in which to save the KML files. If
                           set to ``False``, no files are saved. By default ``False``.
      :type save_kml_dir: str or bool, optional
      :param image_width_resolution: The pixel width to be used for plotting. If ``None``, the
                                     resolution is not changed. Default is ``None``.

                                     Note: Only relevant when ``tree_level="parent"``.
      :type image_width_resolution: int or None, optional
      :param kml_dpi_image: The resolution, in dots per inch, to create KML images when
                            ``save_kml_dir`` is specified (as either ``True`` or with path).
                            By default ``None``.
      :type kml_dpi_image: int or None, optional

      :returns: A list of figures created by the method.
      :rtype: list



   .. py:method:: load_patches(patch_paths, patch_file_ext = False, parent_paths = False, parent_file_ext = False, add_geo_info = False, clear_images = False)

      Loads patch images from the given paths and adds them to the ``images``
      dictionary in the ``MapImages`` instance.

      :param patch_paths: The file path of the patches to be loaded.

                          *Note: The ``patch_paths`` parameter accepts wildcards.*
      :type patch_paths: str
      :param patch_file_ext: The file extension of the patches to be loaded, ignored if file extensions are specified in ``patch_paths`` (e.g. with ``"./path/to/dir/*png"``)
                             By default ``False``.
      :type patch_file_ext: str or bool, optional
      :param parent_paths: The file path of the parent images to be loaded. If set to
                           ``False``, no parents are loaded. Default is ``False``.

                           *Note: The ``parent_paths`` parameter accepts wildcards.*
      :type parent_paths: str or bool, optional
      :param parent_file_ext: The file extension of the parent images, ignored if file extensions are specified in ``parent_paths`` (e.g. with ``"./path/to/dir/*png"``)
                              By default ``False``.
      :type parent_file_ext: str or bool, optional
      :param add_geo_info: If ``True``, adds geographic information to the parent image.
                           Default is ``False``.
      :type add_geo_info: bool, optional
      :param clear_images: If ``True``, clears the images from the ``images`` dictionary
                           before loading. Default is ``False``.
      :type clear_images: bool, optional

      :rtype: None



   .. py:method:: detect_parent_id_from_path(image_id, parent_delimiter = '#')
      :staticmethod:


      Detect parent IDs from ``image_id``.

      :param image_id: ID of patch.
      :type image_id: int or str
      :param parent_delimiter: Delimiter used to separate parent ID when naming patch, by
                               default ``"#"``.
      :type parent_delimiter: str, optional

      :returns: Parent ID.
      :rtype: str



   .. py:method:: detect_pixel_bounds_from_path(image_id)
      :staticmethod:


      Detects borders from the path assuming patch is named using the
      following format: ``...-min_x-min_y-max_x-max_y-...``

      :param image_id: ID of image
      :type image_id: int or str
      :param ..:
                 border_delimiter : str, optional
                     Delimiter used to separate border values when naming patch
                     image, by default ``"-"``.

      :returns: Border (min_x, min_y, max_x, max_y) of image
      :rtype: tuple of min_x, min_y, max_x, max_y



   .. py:method:: load_parents(parent_paths = False, parent_ids = False, parent_file_ext = False, overwrite = False, add_geo_info = False)

      Load parent images from file paths (``parent_paths``).

      If ``parent_paths`` is not given, only ``parent_ids``, no image path
      will be added to the images.

      :param parent_paths: Path to parent images, by default ``False``.
      :type parent_paths: str or bool, optional
      :param parent_ids: ID(s) of parent images. Ignored if ``parent_paths`` are specified.
                         By default ``False``.
      :type parent_ids: list, str or bool, optional
      :param parent_file_ext: The file extension of the parent images, ignored if file extensions are specified in ``parent_paths`` (e.g. with ``"./path/to/dir/*png"``)
                              By default ``False``.
      :type parent_file_ext: str or bool, optional
      :param overwrite: If ``True``, current parents will be overwritten, by default
                        ``False``.
      :type overwrite: bool, optional
      :param add_geo_info: If ``True``, geographical info will be added to parents, by
                           default ``False``.
      :type add_geo_info: bool, optional

      :rtype: None



   .. py:method:: load_df(parent_df = None, patch_df = None, clear_images = True)

      Create ``MapImages`` instance by loading data from pandas DataFrame(s).

      :param parent_df: DataFrame containing parents or path to parents, by default
                        ``None``.
      :type parent_df: pandas.DataFrame, optional
      :param patch_df: DataFrame containing patches, by default ``None``.
      :type patch_df: pandas.DataFrame, optional
      :param clear_images: If ``True``, clear images before reading the dataframes, by
                           default ``True``.
      :type clear_images: bool, optional

      :rtype: None



   .. py:method:: load_csv(parent_path = None, patch_path = None, clear_images = False, index_col_patch = 0, index_col_parent = 0, delimiter = ',')

      Load CSV files containing information about parent and patches,
      and update the ``images`` attribute of the ``MapImages`` instance with
      the loaded data.

      :param parent_path: Path to the CSV file containing parent image information.
      :type parent_path: str, optional
      :param patch_path: Path to the CSV file containing patch information.
      :type patch_path: str, optional
      :param clear_images: If True, clear all previously loaded image information before
                           loading new information. Default is ``False``.
      :type clear_images: bool, optional
      :param index_col_patch: Column to set as index for the patch DataFrame, by default ``0``.
      :type index_col_patch: int, optional
      :param index_col_parent: Column to set as index for the parent DataFrame, by default ``0``.
      :type index_col_parent: int, optional
      :param delimiter: The delimiter to use when reading the dataframe. By default ``","``.
      :type delimiter: str, optional

      :rtype: None



   .. py:method:: add_geo_info(target_crs = 'EPSG:4326', verbose = True)

      Add coordinates (reprojected to EPSG:4326) to all parents images using image metadata.

      :param target_crs: Projection to convert coordinates into, by default ``"EPSG:4326"``.
      :type target_crs: str, optional
      :param verbose: Whether to print verbose output, by default ``True``
      :type verbose: bool, optional

      :rtype: None

      .. rubric:: Notes

      For each image in the parents dictionary, this method calls ``_add_geo_info_id`` and coordinates (if present) to the image in the ``parent`` dictionary.



   .. py:method:: save_parents_as_geotiffs(rewrite = False, verbose = False, crs = None)

      Save all parents in MapImages instance as geotiffs.

      :param rewrite: Whether to rewrite files if they already exist, by default False
      :type rewrite: bool, optional
      :param verbose: Whether to print verbose outputs, by default False
      :type verbose: bool, optional
      :param crs: The CRS of the coordinates.
                  If None, the method will first look for ``crs`` in the parents dictionary and use those. If ``crs`` cannot be found in the dictionary, the method will use "EPSG:4326".
                  By default None.
      :type crs: str, optional



   .. py:method:: save_patches_as_geotiffs(rewrite = False, verbose = False, crs = None)

      Save all patches in MapImages instance as geotiffs.

      :param rewrite: Whether to rewrite files if they already exist, by default False
      :type rewrite: bool, optional
      :param verbose: Whether to print verbose outputs, by default False
      :type verbose: bool, optional
      :param crs: The CRS of the coordinates.
                  If None, the method will first look for ``crs`` in the patches dictionary and use those. If ``crs`` cannot be found in the dictionary, the method will use "EPSG:4326".
                  By default None.
      :type crs: str, optional



   .. py:method:: save_patches_to_geojson(geojson_fname = 'patches.geojson', rewrite = False, crs = None)

      Saves patches to a geojson file.

      :param geojson_fname: The name of the geojson file, by default "patches.geojson"
      :type geojson_fname: Optional[str], optional
      :param rewrite: Whether to overwrite an existing file, by default False.
      :type rewrite: Optional[bool], optional
      :param crs: The CRS to use when writing the geojson.
                  If None, the method will look for "crs" in the patches dictionary and, if found, will use that. Otherwise it will set the crs to the default value of "EPSG:4326".
                  By default None
      :type crs: Optional[str], optional



.. py:function:: loader(path_images = None, tree_level = 'parent', parent_path = None, **kwargs)

   Creates a ``MapImages`` class to manage a collection of image paths and
   construct image objects.

   :param path_images: Path to the directory containing images (accepts wildcards). By
                       default, ``None``
   :type path_images: str or None, optional
   :param tree_level: Level of the image hierarchy to construct. The value can be
                      ``"parent"`` (default) and ``"patch"``.
   :type tree_level: str, optional
   :param parent_path: Path to parent images (if applicable), by default ``None``.
   :type parent_path: str, optional
   :param \*\*kwargs: Additional keyword arguments to be passed to the ``_images_constructor()``
                      method.
   :type \*\*kwargs: dict, optional

   :returns: The ``MapImages`` class which can manage a collection of image paths
             and construct image objects.
   :rtype: MapImages

   .. rubric:: Notes

   This is a wrapper method. See the documentation of the
   :class:`mapreader.load.images.MapImages` class for more detail.


.. py:function:: load_patches(patch_paths, patch_file_ext = False, parent_paths = False, parent_file_ext = False, add_geo_info = False, clear_images = False)

   Creates a ``MapImages`` class to manage a collection of image paths and
   construct image objects. Then loads patch images from the given paths and
   adds them to the ``images`` dictionary in the ``MapImages`` instance.

   :param patch_paths: The file path of the patches to be loaded.

                       *Note: The ``patch_paths`` parameter accepts wildcards.*
   :type patch_paths: str
   :param patch_file_ext: The file extension of the patches, ignored if file extensions are specified in ``patch_paths`` (e.g. with ``"./path/to/dir/*png"``)
                          By default ``False``.
   :type patch_file_ext: str or bool, optional
   :param parent_paths: The file path of the parent images to be loaded. If set to
                        ``False``, no parents are loaded. Default is ``False``.

                        *Note: The ``parent_paths`` parameter accepts wildcards.*
   :type parent_paths: str or bool, optional
   :param parent_file_ext: The file extension of the parent images, ignored if file extensions are specified in ``parent_paths`` (e.g. with ``"./path/to/dir/*png"``)
                           By default ``False``.
   :type parent_file_ext: str or bool, optional
   :param add_geo_info: If ``True``, adds geographic information to the parent image.
                        Default is ``False``.
   :type add_geo_info: bool, optional
   :param clear_images: If ``True``, clears the images from the ``images`` dictionary
                        before loading. Default is ``False``.
   :type clear_images: bool, optional

   :returns: The ``MapImages`` class which can manage a collection of image paths
             and construct image objects.
   :rtype: MapImages

   .. rubric:: Notes

   This is a wrapper method. See the documentation of the
   :class:`mapreader.load.images.MapImages` class for more detail.

   This function in particular, also calls the
   :meth:`mapreader.load.images.MapImages.loadPatches` method. Please see
   the documentation for that method for more information as well.

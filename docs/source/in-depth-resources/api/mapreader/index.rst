mapreader
=========

.. py:module:: mapreader


Subpackages
-----------

.. toctree::
   :maxdepth: 1

   /in-depth-resources/api/mapreader/annotate/index
   /in-depth-resources/api/mapreader/classify/index
   /in-depth-resources/api/mapreader/download/index
   /in-depth-resources/api/mapreader/load/index
   /in-depth-resources/api/mapreader/process/index
   /in-depth-resources/api/mapreader/spot_text/index
   /in-depth-resources/api/mapreader/utils/index


Classes
-------

.. autoapisummary::

   mapreader.MapImages
   mapreader.SheetDownloader
   mapreader.Downloader
   mapreader.AnnotationsLoader
   mapreader.PatchDataset
   mapreader.PatchContextDataset
   mapreader.ClassifierContainer
   mapreader.Annotator


Functions
---------

.. autoapisummary::

   mapreader.loader
   mapreader.load_patches
   mapreader.create_polygon_from_latlons
   mapreader.create_line_from_latlons
   mapreader.print_version


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


.. py:class:: SheetDownloader(metadata_path, download_url)

   A class to download map sheets using metadata.


   .. py:method:: get_polygons()

      For each map in metadata, creates a polygon from map geometry and saves to ``features`` dictionary.



   .. py:method:: get_grid_bb(zoom_level = 14)

      For each map in metadata, creates a grid bounding box from map polygons and saves to ``features`` dictionary.

      :param zoom_level: The zoom level to use when creating the grid bounding box.
                         Later used when downloading maps, by default 14.
      :type zoom_level: int, optional



   .. py:method:: extract_wfs_id_nos()

      For each map in metadata, extracts WFS ID numbers from WFS information and saves to ``features`` dictionary.



   .. py:method:: extract_published_dates(date_col = None)

      For each map in metadata, extracts publication date and saves to ``features`` dictionary.

      :param date_col: A key or list of keys which map to the metadata field containing the publication date.
                       Multilayer keys should be passed as a list. e.g.:

                       - "key1" will extract ``self.features[i]["key1"]``
                       - ["key1","key2"] will search for ``self.features[i]["key1"]["key2"]``

                       If  None, ["properties"]["WFS_TITLE"] will be used as keys. Date will then be extracted by regex searching for "Published: XXX".
                       By default None.
      :type date_col: str or list, optional



   .. py:method:: get_merged_polygon()

      Creates a multipolygon representing all maps in metadata.



   .. py:method:: get_minmax_latlon()

      Prints minimum and maximum latitudes and longitudes of all maps in metadata.



   .. py:method:: query_map_sheets_by_wfs_ids(wfs_ids, append = False, print = False)

      Find map sheets by WFS ID numbers.

      :param wfs_ids: The WFS ID numbers of the maps to download.
      :type wfs_ids: Union[list, int]
      :param append: Whether to append to current query results list or, if False, start a new list.
                     By default False
      :type append: bool, optional
      :param print: Whether to print query results or not.
                    By default False
      :type print: bool, optional



   .. py:method:: query_map_sheets_by_polygon(polygon, mode = 'within', append = False, print = False)

      Find map sheets which are found within or intersecting with a defined polygon.

      :param polygon: shapely Polygon
      :type polygon: Polygon
      :param mode: The mode to use when finding maps.
                   Options of ``"within"``, which returns all map sheets which are completely within the defined polygon,
                   and ``"intersects""``, which returns all map sheets which intersect/overlap with the defined polygon.
                   By default "within".
      :type mode: str, optional
      :param append: Whether to append to current query results list or, if False, start a new list.
                     By default False
      :type append: bool, optional
      :param print: Whether to print query results or not.
                    By default False
      :type print: bool, optional

      .. rubric:: Notes

      Use ``create_polygon_from_latlons()`` to create polygon.



   .. py:method:: query_map_sheets_by_coordinates(coords, append = False, print = False)

      Find maps sheets which contain a defined set of coordinates.
      Coordinates are (x,y).

      :param coords: Coordinates in ``(x,y)`` format.
      :type coords: tuple
      :param append: Whether to append to current query results list or, if False, start a new list.
                     By default False
      :type append: bool, optional
      :param print: Whether to print query results or not.
                    By default False
      :type print: bool, optional



   .. py:method:: query_map_sheets_by_line(line, append = False, print = False)

      Find maps sheets which intersect with a line.

      :param line: shapely LineString
      :type line: LineString
      :param append: Whether to append to current query results list or, if False, start a new list.
                     By default False
      :type append: bool, optional
      :param print: Whether to print query results or not.
                    By default False
      :type print: bool, optional

      .. rubric:: Notes

      Use ``create_line_from_latlons()`` to create line.



   .. py:method:: query_map_sheets_by_string(string, keys = None, append = False, print = False)

      Find map sheets by searching for a string in a chosen metadata field.

      :param string: The string to search for.
                     Can be raw string and use regular expressions.
      :type string: str
      :param keys: A key or list of keys used to get the metadata field to search in.

                   Key(s) will be passed to each features dictionary.
                   Multilayer keys should be passed as a list. e.g. ["key1","key2"] will search for ``self.features[i]["key1"]["key2"]``.

                   If ``None``, will search in all metadata fields. By default ``None``.
      :type keys: str or list, optional
      :param append: Whether to append to current query results list or, if False, start a new list.
                     By default False
      :type append: bool, optional
      :param print: Whether to print query results or not.
                    By default False
      :type print: bool, optional

      .. rubric:: Notes

      ``string`` is case insensitive.



   .. py:method:: print_found_queries()

      Prints query results.



   .. py:method:: download_all_map_sheets(path_save = 'maps', metadata_fname = 'metadata.csv', overwrite = False, download_in_parallel = True, **kwargs)

      Downloads all map sheets in metadata.

      :param path_save: Path to save map sheets, by default "maps"
      :type path_save: str, optional
      :param metadata_fname: Name to use for metadata file, by default "metadata.csv"
      :type metadata_fname: str, optional
      :param overwrite: Whether to overwrite existing maps, by default ``False``.
      :type overwrite: bool, optional
      :param download_in_parallel: Whether to download tiles in parallel, by default ``True``.
      :type download_in_parallel: bool, optional
      :param \*\*kwargs: Keyword arguments to pass to the ``_download_map_sheets()`` method.
      :type \*\*kwargs: dict, optional



   .. py:method:: download_map_sheets_by_wfs_ids(wfs_ids, path_save = 'maps', metadata_fname = 'metadata.csv', overwrite = False, download_in_parallel = True, **kwargs)

      Downloads map sheets by WFS ID numbers.

      :param wfs_ids: The WFS ID numbers of the maps to download.
      :type wfs_ids: Union[list, int]
      :param path_save: Path to save map sheets, by default "maps"
      :type path_save: str, optional
      :param metadata_fname: Name to use for metadata file, by default "metadata.csv"
      :type metadata_fname: str, optional
      :param overwrite: Whether to overwrite existing maps, by default ``False``.
      :type overwrite: bool, optional
      :param download_in_parallel: Whether to download tiles in parallel, by default ``True``.
      :type download_in_parallel: bool, optional
      :param \*\*kwargs: Keyword arguments to pass to the ``_download_map_sheets()`` method.
      :type \*\*kwargs: dict, optional



   .. py:method:: download_map_sheets_by_polygon(polygon, path_save = 'maps', metadata_fname = 'metadata.csv', mode = 'within', overwrite = False, download_in_parallel = True, **kwargs)

      Downloads any map sheets which are found within or intersecting with a defined polygon.

      :param polygon: shapely Polygon
      :type polygon: Polygon
      :param path_save: Path to save map sheets, by default "maps"
      :type path_save: str, optional
      :param metadata_fname: Name to use for metadata file, by default "metadata.csv"
      :type metadata_fname: str, optional
      :param mode: The mode to use when finding maps.
                   Options of ``"within"``, which returns all map sheets which are completely within the defined polygon,
                   and ``"intersects""``, which returns all map sheets which intersect/overlap with the defined polygon.
                   By default "within".
      :type mode: str, optional
      :param overwrite: Whether to overwrite existing maps, by default ``False``.
      :type overwrite: bool, optional
      :param download_in_parallel: Whether to download tiles in parallel, by default ``True``.
      :type download_in_parallel: bool, optional
      :param \*\*kwargs: Keyword arguments to pass to the ``_download_map_sheets()`` method.
      :type \*\*kwargs: dict, optional

      .. rubric:: Notes

      Use ``create_polygon_from_latlons()`` to create polygon.



   .. py:method:: download_map_sheets_by_coordinates(coords, path_save = 'maps', metadata_fname = 'metadata.csv', overwrite = False, download_in_parallel = True, **kwargs)

      Downloads any maps sheets which contain a defined set of coordinates.
      Coordinates are (x,y).

      :param coords: Coordinates in ``(x,y)`` format.
      :type coords: tuple
      :param path_save: Path to save map sheets, by default "maps"
      :type path_save: str, optional
      :param metadata_fname: Name to use for metadata file, by default "metadata.csv"
      :type metadata_fname: str, optional
      :param overwrite: Whether to overwrite existing maps, by default ``False``.
      :type overwrite: bool, optional
      :param download_in_parallel: Whether to download tiles in parallel, by default ``True``.
      :type download_in_parallel: bool, optional
      :param \*\*kwargs: Keyword arguments to pass to the ``_download_map_sheets()`` method.
      :type \*\*kwargs: dict, optional



   .. py:method:: download_map_sheets_by_line(line, path_save = 'maps', metadata_fname = 'metadata.csv', overwrite = False, download_in_parallel = True, **kwargs)

      Downloads any maps sheets which intersect with a line.

      :param line: shapely LineString
      :type line: LineString
      :param path_save: Path to save map sheets, by default "maps"
      :type path_save: str, optional
      :param metadata_fname: Name to use for metadata file, by default "metadata.csv"
      :type metadata_fname: str, optional
      :param overwrite: Whether to overwrite existing maps, by default ``False``
      :type overwrite: bool, optional
      :param download_in_parallel: Whether to download tiles in parallel, by default ``True``.
      :type download_in_parallel: bool, optional
      :param \*\*kwargs: Keyword arguments to pass to the ``_download_map_sheets()`` method.
      :type \*\*kwargs: dict, optional

      .. rubric:: Notes

      Use ``create_line_from_latlons()`` to create line.



   .. py:method:: download_map_sheets_by_string(string, keys = None, path_save = 'maps', metadata_fname = 'metadata.csv', overwrite = False, download_in_parallel = True, **kwargs)

      Download map sheets by searching for a string in a chosen metadata field.

      :param string: The string to search for.
                     Can be raw string and use regular expressions.
      :type string: str
      :param keys: A key or list of keys used to get the metadata field to search in.

                   Key(s) will be passed to each features dictionary.
                   Multilayer keys should be passed as a list. e.g. ["key1","key2"] will search for ``self.features[i]["key1"]["key2"]``.

                   If ``None``, will search in all metadata fields. By default ``None``.
      :type keys: str or list, optional
      :param path_save: Path to save map sheets, by default "maps"
      :type path_save: str, optional
      :param metadata_fname: Name to use for metadata file, by default "metadata.csv"
      :type metadata_fname: str, optional
      :param overwrite: Whether to overwrite existing maps, by default ``False``.
      :type overwrite: bool, optional
      :param download_in_parallel: Whether to download tiles in parallel, by default ``True``.
      :type download_in_parallel: bool, optional
      :param \*\*kwargs: Keyword arguments to pass to the ``_download_map_sheets()`` method.
      :type \*\*kwargs: dict, optional

      .. rubric:: Notes

      ``string`` is case insensitive.



   .. py:method:: download_map_sheets_by_queries(path_save = 'maps', metadata_fname = 'metadata.csv', overwrite = False, download_in_parallel = True, **kwargs)

      Downloads map sheets saved as query results.

      :param path_save: Path to save map sheets, by default "maps"
      :type path_save: str, optional
      :param metadata_fname: Name to use for metadata file, by default "metadata.csv"
      :type metadata_fname: str, optional
      :param overwrite: Whether to overwrite existing maps, by default ``False``.
      :type overwrite: bool, optional
      :param download_in_parallel: Whether to download tiles in parallel, by default ``True``.
      :type download_in_parallel: bool, optional
      :param \*\*kwargs: Keyword arguments to pass to the ``_download_map_sheets()`` method.
      :type \*\*kwargs: dict, optional



   .. py:method:: hist_published_dates(**kwargs)

      Plots a histogram of the publication dates of maps in metadata.

      :param \*\*kwargs: A dictionary containing keyword arguments to pass to plotting function.
                         See matplotlib.pyplot.hist() for acceptable values.

                         e.g. ``**dict(fc='c', ec='k')``
      :type \*\*kwargs: dict, optional

      .. rubric:: Notes

      bins and range already set when plotting so are invalid kwargs.



   .. py:method:: plot_features_on_map(features, map_extent = None, add_id = True)

      Plots boundaries of map sheets on a map using ``cartopy`` library, (if available).

      :param map_extent: The extent of the underlying map to be plotted.

                         If a tuple or list, must be of the format ``[lon_min, lon_max, lat_min, lat_max]``.
                         If a string, only ``"uk"``, ``"UK"`` or ``"United Kingdom"`` are accepted and will limit the map extent to the UK's boundaries.
                         If None, the map extent will be set automatically.
                         By default None.
      :type map_extent: Union[str, list, tuple, None], optional
      :param add_id: Whether to add an ID (WFS ID number) to each map sheet, by default True.
      :type add_id: bool, optional



   .. py:method:: plot_all_metadata_on_map(map_extent = None, add_id = True)

      Plots boundaries of all map sheets in metadata on a map using ``cartopy`` library (if available).

      :param map_extent: The extent of the underlying map to be plotted.

                         If a tuple or list, must be of the format ``[lon_min, lon_max, lat_min, lat_max]``.
                         If a string, only ``"uk"``, ``"UK"`` or ``"United Kingdom"`` are accepted and will limit the map extent to the UK's boundaries.
                         If None, the map extent will be set automatically.
                         By default None.
      :type map_extent: Union[str, list, tuple, None], optional
      :param add_id: Whether to add an ID (WFS ID number) to each map sheet, by default True.
      :type add_id: bool, optional



   .. py:method:: plot_queries_on_map(map_extent = None, add_id = True)

      Plots boundaries of query results on a map using ``cartopy`` library (if available).

      :param map_extent: The extent of the underlying map to be plotted.

                         If a tuple or list, must be of the format ``[lon_min, lon_max, lat_min, lat_max]``.
                         If a string, only ``"uk"``, ``"UK"`` or ``"United Kingdom"`` are accepted and will limit the map extent to the UK's boundaries.
                         If None, the map extent will be set automatically.
                         By default None.
      :type map_extent: Union[str, list, tuple, None], optional
      :param add_id: Whether to add an ID (WFS ID number) to each map sheet, by default True.
      :type add_id: bool, optional



.. py:class:: Downloader(download_url)

   A class to download maps (without using metadata)


   .. py:method:: download_map_by_polygon(polygon, zoom_level = 14, path_save = 'maps', overwrite = False, map_name = None)

      Downloads a map contained within a polygon.

      :param polygon: A polygon defining the boundaries of the map
      :type polygon: Polygon
      :param zoom_level: The zoom level to use, by default 14
      :type zoom_level: int, optional
      :param path_save: Path to save map sheets, by default "maps"
      :type path_save: str, optional
      :param overwrite: Whether to overwrite existing maps, by default ``False``.
      :type overwrite: bool, optional
      :param map_name: Name to use when saving the map, by default None
      :type map_name: str, optional



.. py:function:: create_polygon_from_latlons(min_lat, min_lon, max_lat, max_lon)

   Creates a polygon from latitudes and longitudes.

   :param min_lat: minimum latitude
   :type min_lat: float
   :param min_lon: minimum longitude
   :type min_lon: float
   :param max_lat: maximum latitude
   :type max_lat: float
   :param max_lon: maximum longitude
   :type max_lon: float

   :returns: shapely Polgyon
   :rtype: Polygon


.. py:function:: create_line_from_latlons(lat1_lon1, lat2_lon2)

   Creates a line between two points.

   :param lat1_lon1: Tuple defining first point
   :type lat1_lon1: tuple
   :param lat2: Tuple defining second point
   :type lat2: tuple

   :returns: shapely LineString
   :rtype: LineString


.. py:class:: AnnotationsLoader

   .. py:method:: load(annotations, delimiter = ',', images_dir = None, remove_broken = True, ignore_broken = False, patch_paths_col = 'image_path', label_col = 'label', append = True, scramble_frame = False, reset_index = False)

      Loads annotations from a csv file or dataframe and can be used to set the ``patch_paths_col`` and ``label_col`` attributes.

      :param annotations: The annotations.
                          Can either be the path to a csv file or a pandas.DataFrame.
      :type annotations: Union[str, pd.DataFrame]
      :param delimiter: The delimiter to use when loading the csv file as a dataframe, by default ",".
      :type delimiter: Optional[str], optional
      :param images_dir: The path to the directory in which patches are stored.
                         This argument should be passed if image paths are different from the path saved in annotations dataframe/csv.
                         If None, no updates will be made to the image paths in the annotations dataframe/csv.
                         By default None.
      :type images_dir: Optional[str], optional
      :param remove_broken: Whether to remove annotations with broken image paths.
                            If False, annotations with broken paths will remain in annotations dataframe and may cause issues!
                            By default True.
      :type remove_broken: Optional[bool], optional
      :param ignore_broken: Whether to ignore broken image paths (only valid if remove_broken=False).
                            If True, annotations with broken paths will remain in annotations dataframe and no error will be raised. This may cause issues!
                            If False, annotations with broken paths will raise error. By default, False.
      :type ignore_broken: Optional[bool], optional
      :param patch_paths_col: The name of the column containing the image paths, by default "image_path".
      :type patch_paths_col: Optional[str], optional
      :param label_col: The name of the column containing the image labels, by default "label".
      :type label_col: Optional[str], optional
      :param append: Whether to append the annotations to a pre-existing ``annotations`` dataframe.
                     If False, existing dataframe will be overwritten.
                     By default True.
      :type append: Optional[bool], optional
      :param scramble_frame: Whether to shuffle the rows of the dataframe, by default False.
      :type scramble_frame: Optional[bool], optional
      :param reset_index: Whether to reset the index of the dataframe (e.g. after shuffling), by default False.
      :type reset_index: Optional[bool], optional

      :raises ValueError: If ``annotations`` is passed as something other than a string or pd.DataFrame.



   .. py:method:: show_patch(patch_id)

      Display a patch and its label.

      :param patch_id: The image ID of the patch to show.
      :type patch_id: str

      :rtype: None



   .. py:method:: print_unique_labels()

      Prints unique labels

      :raises ValueError: If no annotations are found.



   .. py:method:: review_labels(label_to_review = None, chunks = 8 * 3, num_cols = 8, exclude_df = None, include_df = None, deduplicate_col = 'image_id')

      Perform image review on annotations and update labels for a given
      label or all labels.

      :param label_to_review: The target label to review. If not provided, all labels will be
                              reviewed, by default ``None``.
      :type label_to_review: str, optional
      :param chunks: The number of images to display at a time, by default ``24``.
      :type chunks: int, optional
      :param num_cols: The number of columns in the display, by default ``8``.
      :type num_cols: int, optional
      :param exclude_df: A DataFrame of images to exclude from review, by default ``None``.
      :type exclude_df: pandas.DataFrame, optional
      :param include_df: A DataFrame of images to include for review, by default ``None``.
      :type include_df: pandas.DataFrame, optional
      :param deduplicate_col: The column to use for deduplicating reviewed images, by default
                              ``"image_id"``.
      :type deduplicate_col: str, optional

      :rtype: None

      .. rubric:: Notes

      This method reviews images with their corresponding labels and allows
      the user to change the label for each image.

      Updated labels are saved in ``self.annotations`` and in a newly created ``self.reviewed`` DataFrame.
      If ``exclude_df`` is provided, images found in this df are skipped in the review process.
      If ``include_df`` is provided, only images found in this df are reviewed.
      The ``self.reviewed`` DataFrame is deduplicated based on the ``deduplicate_col``.



   .. py:method:: show_sample(label_to_show, num_samples = 9)

      Show a random sample of images with the specified label (tar_label).

      :param label_to_show: The label of the images to show.
      :type label_to_show: str, optional
      :param num_sample: The number of images to show.
                         If ``None``, all images with the specified label will be shown. Default is ``9``.
      :type num_sample: int, optional

      :rtype: None



   .. py:method:: create_datasets(frac_train = 0.7, frac_val = 0.15, frac_test = 0.15, random_state = 1364, train_transform = 'train', val_transform = 'val', test_transform = 'test', context_datasets = False, context_df = None)

      Splits the dataset into three subsets: training, validation, and test sets (DataFrames) and saves them as a dictionary in ``self.datasets``.

      :param frac_train: Fraction of the dataset to be used for training.
                         By default ``0.70``.
      :type frac_train: float, optional
      :param frac_val: Fraction of the dataset to be used for validation.
                       By default ``0.15``.
      :type frac_val: float, optional
      :param frac_test: Fraction of the dataset to be used for testing.
                        By default ``0.15``.
      :type frac_test: float, optional
      :param random_state: Random seed to ensure reproducibility. The default is ``1364``.
      :type random_state: int, optional
      :param train_transform: The transform to use on the training dataset images.
                              Options are "train", "test" or "val" or, a callable object (e.g. a torchvision transform or torchvision.transforms.Compose).
                              By default "train".
      :type train_transform: str, tochvision.transforms.Compose or Callable, optional
      :param val_transform: The transform to use on the validation dataset images.
                            Options are "train", "test" or "val" or, a callable object (e.g. a torchvision transform or torchvision.transforms.Compose).
                            By default "val".
      :type val_transform: str, tochvision.transforms.Compose or Callable, optional
      :param test_transform: The transform to use on the test dataset images.
                             Options are "train", "test" or "val" or, a callable object (e.g. a torchvision transform or torchvision.transforms.Compose).
                             By default "test".
      :type test_transform: str, tochvision.transforms.Compose or Callable, optional
      :param context_datasets: Whether to create context datasets or not. By default False.
      :type context_datasets: bool, optional
      :param context_df: The dataframe containing all patches if using context datasets.
                         Used to create context images. By default None.
      :type context_df: str or pandas.DataFrame, optional

      :raises ValueError: If the sum of fractions of training, validation and test sets does
          not add up to 1.

      :rtype: None

      .. rubric:: Notes

      This method saves the split datasets as a dictionary in ``self.datasets``.

      Following fractional ratios provided by the user, where each subset is
      stratified by the values in a specific column (that is, each subset has
      the same relative frequency of the values in the column). It performs
      this splitting by running ``train_test_split()`` twice.

      See ``PatchDataset`` for more information on transforms.



   .. py:method:: create_patch_datasets(train_transform, val_transform, test_transform, df_train, df_val, df_test)


   .. py:method:: create_patch_context_datasets(context_df, train_transform, val_transform, test_transform, df_train, df_val, df_test)


   .. py:method:: create_dataloaders(batch_size = 16, sampler = 'default', shuffle = False, num_workers = 0, **kwargs)

      Creates a dictionary containing PyTorch dataloaders
      saves it to as ``self.dataloaders`` and returns it.

      :param batch_size: The batch size to use for the dataloader. By default ``16``.
      :type batch_size: int, optional
      :param sampler: The sampler to use when creating batches from the training dataset.
      :type sampler: Sampler, str or None, optional
      :param shuffle: Whether to shuffle the dataset during training. By default ``False``.
      :type shuffle: bool, optional
      :param num_workers: The number of worker threads to use for loading data. By default ``0``.
      :type num_workers: int, optional
      :param \*\*kwds: Additional keyword arguments to pass to PyTorch's ``DataLoader`` constructor.

      :returns: Dictionary containing dataloaders.
      :rtype: Dict

      .. rubric:: Notes

      ``sampler`` will only be applied to the training dataset (datasets["train"]).



.. py:class:: PatchDataset(patch_df, transform, delimiter = ',', patch_paths_col = 'image_path', label_col = None, label_index_col = None, image_mode = 'RGB')

   Bases: :py:obj:`torch.utils.data.Dataset`


   An abstract class representing a :class:`Dataset`.

   All datasets that represent a map from keys to data samples should subclass
   it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
   data sample for a given key. Subclasses could also optionally overwrite
   :meth:`__len__`, which is expected to return the size of the dataset by many
   :class:`~torch.utils.data.Sampler` implementations and the default options
   of :class:`~torch.utils.data.DataLoader`. Subclasses could also
   optionally implement :meth:`__getitems__`, for speedup batched samples
   loading. This method accepts list of indices of samples of batch and returns
   list of samples.

   .. note::
     :class:`~torch.utils.data.DataLoader` by default constructs an index
     sampler that yields integral indices.  To make it work with a map-style
     dataset with non-integral indices/keys, a custom sampler must be provided.


   .. py:method:: return_orig_image(idx)

      Return the original image associated with the given index.

      :param idx: The index of the desired image, or a Tensor containing the index.
      :type idx: int or Tensor

      :returns: The original image associated with the given index.
      :rtype: PIL.Image.Image

      .. rubric:: Notes

      This method returns the original image associated with the given index
      by loading the image file using the file path stored in the
      ``patch_paths_col`` column of the ``patch_df`` DataFrame at the given
      index. The loaded image is then converted to the format specified by
      the ``image_mode`` attribute of the object. The resulting
      ``PIL.Image.Image`` object is returned.



   .. py:method:: create_dataloaders(set_name = 'infer', batch_size = 16, shuffle = False, num_workers = 0, **kwargs)

      Creates a dictionary containing a PyTorch dataloader.

      :param set_name: The name to use for the dataloader.
      :type set_name: str, optional
      :param batch_size: The batch size to use for the dataloader. By default ``16``.
      :type batch_size: int, optional
      :param shuffle: Whether to shuffle the PatchDataset, by default False
      :type shuffle: bool, optional
      :param num_workers: The number of worker threads to use for loading data. By default ``0``.
      :type num_workers: int, optional
      :param \*\*kwargs: Additional keyword arguments to pass to PyTorch's ``DataLoader`` constructor.

      :returns: Dictionary containing dataloaders.
      :rtype: Dict



.. py:class:: PatchContextDataset(patch_df, total_df, transform, delimiter = ',', patch_paths_col = 'image_path', label_col = None, label_index_col = None, image_mode = 'RGB', context_dir = './maps/maps_context', create_context = False, parent_path = './maps')

   Bases: :py:obj:`PatchDataset`


   An abstract class representing a :class:`Dataset`.

   All datasets that represent a map from keys to data samples should subclass
   it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
   data sample for a given key. Subclasses could also optionally overwrite
   :meth:`__len__`, which is expected to return the size of the dataset by many
   :class:`~torch.utils.data.Sampler` implementations and the default options
   of :class:`~torch.utils.data.DataLoader`. Subclasses could also
   optionally implement :meth:`__getitems__`, for speedup batched samples
   loading. This method accepts list of indices of samples of batch and returns
   list of samples.

   .. note::
     :class:`~torch.utils.data.DataLoader` by default constructs an index
     sampler that yields integral indices.  To make it work with a map-style
     dataset with non-integral indices/keys, a custom sampler must be provided.


   .. py:method:: save_context(processors = 10, sleep_time = 0.001, use_parhugin = True, overwrite = False)

      Save context images for all patches in the patch_df.

      :param processors: The number of required processors for the job, by default 10.
      :type processors: int, optional
      :param sleep_time: The time to wait between jobs, by default 0.001.
      :type sleep_time: float, optional
      :param use_parhugin: Whether to use Parhugin to parallelize the job, by default True.
      :type use_parhugin: bool, optional
      :param overwrite: Whether to overwrite existing parent files, by default False.
      :type overwrite: bool, optional

      :rtype: None

      .. rubric:: Notes

      Parhugin is a Python package for parallelizing computations across
      multiple CPU cores. The method uses Parhugin to parallelize the
      computation of saving parent patches to disk. When Parhugin is
      installed and ``use_parhugin`` is set to True, the method parallelizes
      the calling of the ``get_context_id`` method and its corresponding
      arguments. If Parhugin is not installed or ``use_parhugin`` is set to
      False, the method executes the loop over patch indices sequentially
      instead.



   .. py:method:: get_context_id(id, overwrite = False, save_context = False, return_image = True)

      Save the parents of a specific patch to the specified location.

      :param id: Index of the patch in the dataset.
      :param overwrite: Whether to overwrite the existing parent files. Default is
                        False.
      :type overwrite: bool, optional
      :param save_context: Whether to save the context image. Default is False.
      :type save_context: bool, optional
      :param return_image: Whether to return the context image. Default is True.
      :type return_image: bool, optional

      :raises ValueError: If the patch is not found in the dataset.

      :rtype: None



   .. py:method:: plot_sample(idx)

      Plot a sample patch and its corresponding context from the dataset.

      :param idx: The index of the sample to plot.
      :type idx: int

      :returns: Displays the plot of the sample patch and its corresponding
                context.
      :rtype: None

      .. rubric:: Notes

      This method plots a sample patch and its corresponding context side-by-
      side in a single figure with two subplots. The figure size is set to
      10in x 5in, and the titles of the subplots are set to "Patch" and
      "Context", respectively. The resulting figure is displayed using
      the ``matplotlib`` library (required).



.. py:class:: ClassifierContainer(model, labels_map, dataloaders = None, device = 'default', input_size = (224, 224), is_inception = False, load_path = None, force_device = False, **kwargs)

   .. py:method:: generate_layerwise_lrs(min_lr, max_lr, spacing = 'linspace')

      Calculates layer-wise learning rates for a given set of model
      parameters.

      :param min_lr: The minimum learning rate to be used.
      :type min_lr: float
      :param max_lr: The maximum learning rate to be used.
      :type max_lr: float
      :param spacing: The type of sequence to use for spacing the specified interval
                      learning rates. Can be either ``"linspace"`` or ``"geomspace"``,
                      where `"linspace"` uses evenly spaced learning rates over a
                      specified interval and `"geomspace"` uses learning rates spaced
                      evenly on a log scale (a geometric progression). By default ``"linspace"``.
      :type spacing: str, optional

      :returns: A list of dictionaries containing the parameters and learning
                rates for each layer.
      :rtype: list of dicts



   .. py:method:: initialize_optimizer(optim_type = 'adam', params2optimize = 'default', optim_param_dict = None, add_optim = True)

      Initializes an optimizer for the model and adds it to the classifier
      object.

      :param optim_type: The type of optimizer to use. Can be set to ``"adam"`` (default),
                         ``"adamw"``, or ``"sgd"``.
      :type optim_type: str, optional
      :param params2optimize: The parameters to optimize. If set to ``"default"``, all model
                              parameters that require gradients will be optimized.
                              Default is ``"default"``.
      :type params2optimize: str or iterable, optional
      :param optim_param_dict: The parameters to pass to the optimizer constructor as a
                               dictionary, by default ``{"lr": 1e-3}``.
      :type optim_param_dict: dict, optional
      :param add_optim: If ``True``, adds the optimizer to the classifier object, by
                        default ``True``.
      :type add_optim: bool, optional

      :returns: **optimizer** -- The initialized optimizer. Only returned if ``add_optim`` is set to
                ``False``.
      :rtype: torch.optim.Optimizer

      .. rubric:: Notes

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

      :param optimizer: The optimizer to add to the classifier object.
      :type optimizer: torch.optim.Optimizer

      :rtype: None



   .. py:method:: initialize_scheduler(scheduler_type = 'steplr', scheduler_param_dict = None, add_scheduler = True)

      Initializes a learning rate scheduler for the optimizer and adds it to
      the classifier object.

      :param scheduler_type: The type of learning rate scheduler to use. Can be either
                             ``"steplr"`` (default) or ``"onecyclelr"``.
      :type scheduler_type: str, optional
      :param scheduler_param_dict: The parameters to pass to the scheduler constructor, by default
                                   ``{"step_size": 10, "gamma": 0.1}``.
      :type scheduler_param_dict: dict, optional
      :param add_scheduler: If ``True``, adds the scheduler to the classifier object, by
                            default ``True``.
      :type add_scheduler: bool, optional

      :raises ValueError: If the specified ``scheduler_type`` is not implemented.

      :returns: **scheduler** -- The initialized learning rate scheduler. Only returned if
                ``add_scheduler`` is set to False.
      :rtype: torch.optim.lr_scheduler._LRScheduler



   .. py:method:: add_scheduler(scheduler)

      Add a scheduler to the classifier object.

      :param scheduler: The scheduler to add to the classifier object.
      :type scheduler: torch.optim.lr_scheduler._LRScheduler

      :raises ValueError: If no optimizer has been set. Use ``initialize_optimizer`` or
          ``add_optimizer`` to set an optimizer first.

      :rtype: None



   .. py:method:: add_criterion(criterion = 'cross entropy')

      Add a loss criterion to the classifier object.

      :param criterion: The loss criterion to add to the classifier object.
                        Accepted string values are "cross entropy" or "ce" (cross-entropy), "bce" (binary cross-entropy) and "mse" (mean squared error).
      :type criterion: str or torch.nn.modules.loss._Loss

      :returns: The function only modifies the ``criterion`` attribute of the
                classifier and does not return anything.
      :rtype: None



   .. py:method:: model_summary(input_size = None, trainable_col = False, **kwargs)

      Print a summary of the model.

      :param input_size: The size of the input data.
                         If None, input size is taken from "train" dataloader (``self.dataloaders["train"]``).
      :type input_size: tuple or list, optional
      :param trainable_col: If ``True``, adds a column showing which parameters are trainable.
                            Defaults to ``False``.
      :type trainable_col: bool, optional
      :param \*\*kwargs: Keyword arguments to pass to ``torchinfo.summary()`` (see https://github.com/TylerYep/torchinfo).
      :type \*\*kwargs: Dict

      .. rubric:: Notes

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



   .. py:method:: freeze_layers(layers_to_freeze = None)

      Freezes the specified layers in the neural network by setting
      ``requires_grad`` attribute to False for their parameters.

      :param layers_to_freeze: List of names of the layers to freeze. If a layer name ends with
                               an asterisk (``"*"``), then all parameters whose name contains the
                               layer name (excluding the asterisk) are frozen. Otherwise,
                               only the parameters with an exact match to the layer name
                               are frozen. By default, ``[]``.
      :type layers_to_freeze: list of str, optional

      :returns: The function only modifies the ``requires_grad`` attribute of the
                specified parameters and does not return anything.
      :rtype: None

      .. rubric:: Notes

      Wildcards are accepted in the ``layers_to_freeze`` parameter.



   .. py:method:: unfreeze_layers(layers_to_unfreeze = None)

      Unfreezes the specified layers in the neural network by setting
      ``requires_grad`` attribute to True for their parameters.

      :param layers_to_unfreeze: List of names of the layers to unfreeze. If a layer name ends with
                                 an asterisk (``"*"``), then all parameters whose name contains the
                                 layer name (excluding the asterisk) are unfrozen. Otherwise,
                                 only the parameters with an exact match to the layer name
                                 are unfrozen. By default, ``[]``.
      :type layers_to_unfreeze: list of str, optional

      :returns: The function only modifies the ``requires_grad`` attribute of the
                specified parameters and does not return anything.
      :rtype: None

      .. rubric:: Notes

      Wildcards are accepted in the ``layers_to_unfreeze`` parameter.



   .. py:method:: only_keep_layers(only_keep_layers_list = None)

      Only keep the specified layers (``only_keep_layers_list``) for
      gradient computation during the backpropagation.

      :param only_keep_layers_list: List of layer names to keep. All other layers will have their
                                    gradient computation turned off. Default is ``[]``.
      :type only_keep_layers_list: list, optional

      :returns: The function only modifies the ``requires_grad`` attribute of the
                specified parameters and does not return anything.
      :rtype: None



   .. py:method:: inference(set_name = 'infer', verbose = False, print_info_batch_freq = 5)

      Run inference on a specified dataset (``set_name``).

      :param set_name: The name of the dataset to run inference on, by default
                       ``"infer"``.
      :type set_name: str, optional
      :param verbose: Whether to print verbose outputs, by default False.
      :type verbose: bool, optional
      :param print_info_batch_freq: The frequency of printouts, by default ``5``.
      :type print_info_batch_freq: int, optional

      :rtype: None

      .. rubric:: Notes

      This method calls the
      :meth:`mapreader.train.classifier.classifier.train` method with the
      ``num_epochs`` set to ``1`` and all the other parameters specified in
      the function arguments.



   .. py:method:: train_component_summary()

      Print a summary of the optimizer, criterion, and trainable model
      components.

      Returns:
      --------
      None



   .. py:method:: train(phases = None, num_epochs = 25, save_model_dir = 'models', verbose = False, tensorboard_path = None, tmp_file_save_freq = 2, remove_after_load = True, print_info_batch_freq = 5)

      Train the model on the specified phases for a given number of epochs.

      Wrapper function for
      :meth:`mapreader.train.classifier.classifier.train_core` method to
      capture exceptions (``KeyboardInterrupt`` is the only supported
      exception currently).

      :param phases: The phases to run through during each training iteration. Default is
                     ``["train", "val"]``.
      :type phases: list of str, optional
      :param num_epochs: The number of epochs to train the model for. Default is ``25``.
      :type num_epochs: int, optional
      :param save_model_dir: The directory to save the model in. Default is ``"models"``. If
                             set to ``None``, the model is not saved.
      :type save_model_dir: str or None, optional
      :param verbose: Whether to print verbose outputs, by default ``False``.
      :type verbose: int, optional
      :param tensorboard_path: The path to the directory to save TensorBoard logs in. If set to
                               ``None``, no TensorBoard logs are saved. Default is ``None``.
      :type tensorboard_path: str or None, optional
      :param tmp_file_save_freq: The frequency (in epochs) to save a temporary file of the model.
                                 Default is ``2``. If set to ``0`` or ``None``, no temporary file
                                 is saved.
      :type tmp_file_save_freq: int, optional
      :param remove_after_load: Whether to remove the temporary file after loading it. Default is
                                ``True``.
      :type remove_after_load: bool, optional
      :param print_info_batch_freq: The frequency (in batches) to print training information. Default
                                    is ``5``. If set to ``0`` or ``None``, no training information is
                                    printed.
      :type print_info_batch_freq: int, optional

      :returns: The function saves the model to the ``save_model_dir`` directory,
                and optionally to a temporary file. If interrupted with a
                ``KeyboardInterrupt``, the function tries to load the temporary
                file. If no temporary file is found, it continues without loading.
      :rtype: None

      .. rubric:: Notes

      Refer to the documentation of
      :meth:`mapreader.train.classifier.classifier.train_core` for more
      information.



   .. py:method:: train_core(phases = None, num_epochs = 25, save_model_dir = 'models', verbose = False, tensorboard_path = None, tmp_file_save_freq = 2, print_info_batch_freq = 5)

      Trains/fine-tunes a classifier for the specified number of epochs on
      the given phases using the specified hyperparameters.

      :param phases: The phases to run through during each training iteration. Default is
                     ``["train", "val"]``.
      :type phases: list of str, optional
      :param num_epochs: The number of epochs to train the model for. Default is ``25``.
      :type num_epochs: int, optional
      :param save_model_dir: The directory to save the model in. Default is ``"models"``. If
                             set to ``None``, the model is not saved.
      :type save_model_dir: str or None, optional
      :param verbose: Whether to print verbose outputs, by default ``False``.
      :type verbose: bool, optional
      :param tensorboard_path: The path to the directory to save TensorBoard logs in. If set to
                               ``None``, no TensorBoard logs are saved. Default is ``None``.
      :type tensorboard_path: str or None, optional
      :param tmp_file_save_freq: The frequency (in epochs) to save a temporary file of the model.
                                 Default is ``2``. If set to ``0`` or ``None``, no temporary file
                                 is saved.
      :type tmp_file_save_freq: int, optional
      :param print_info_batch_freq: The frequency (in batches) to print training information. Default
                                    is ``5``. If set to ``0`` or ``None``, no training information is
                                    printed.
      :type print_info_batch_freq: int, optional

      :raises ValueError: If the criterion is not set. Use the ``add_criterion`` method to
          set the criterion.

          If the optimizer is not set and the phase is "train". Use the
          ``initialize_optimizer`` or ``add_optimizer`` method to set the
          optimizer.
      :raises KeyError: If the specified phase cannot be found in the keys of the object's
          ``dataloaders`` dictionary property.

      :rtype: None



   .. py:method:: calculate_add_metrics(y_true, y_pred, y_score, phase, epoch = -1, tboard_writer=None)

      Calculate and add metrics to the classifier's metrics dictionary.

      :param y_true: True binary labels or multiclass labels. Can be considered ground
                     truth or (correct) target values.
      :type y_true: array-like of shape (n_samples,)
      :param y_pred: Predicted binary labels or multiclass labels. The estimated
                     targets as returned by a classifier.
      :type y_pred: array-like of shape (n_samples,)
      :param y_score: Predicted probabilities for each class. Only required when
                      ``y_pred`` is not binary.
      :type y_score: array-like of shape (n_samples, n_classes)
      :param phase: Name of the current phase, typically ``"train"`` or ``"val"``. See
                    ``train`` function.
      :type phase: str
      :param epoch: Current epoch number. Default is ``-1``.
      :type epoch: int, optional
      :param tboard_writer: TensorBoard SummaryWriter object to write the metrics. Default is
                            ``None``.
      :type tboard_writer: object, optional

      :rtype: None

      .. rubric:: Notes

      This method uses both the
      ``sklearn.metrics.precision_recall_fscore_support`` and
      ``sklearn.metrics.roc_auc_score`` functions from ``scikit-learn`` to
      calculate the metrics for each average type (``"micro"``, ``"macro"``
      and ``"weighted"``). The results are then added to the ``metrics``
      dictionary. It also writes the metrics to the TensorBoard
      SummaryWriter, if ``tboard_writer`` is not None.



   .. py:method:: plot_metric(y_axis, y_label, legends, x_axis = 'epoch', x_label = 'epoch', colors = 5 * ['k', 'tab:red'], styles = 10 * ['-'], markers = 10 * ['o'], figsize = (10, 5), plt_yrange = None, plt_xrange = None)

      Plot the metrics of the classifier object.

      :param y_axis: A list of metric names to be plotted on the y-axis.
      :type y_axis: list of str
      :param y_label: The label for the y-axis.
      :type y_label: str
      :param legends: The legend labels for each metric.
      :type legends: list of str
      :param x_axis: The metric to be used as the x-axis. Can be ``"epoch"`` (default)
                     or any other metric name present in the dataset.
      :type x_axis: str, optional
      :param x_label: The label for the x-axis. Defaults to ``"epoch"``.
      :type x_label: str, optional
      :param colors: The colors to be used for the lines of each metric. It must be at
                     least the same size as ``y_axis``. Defaults to
                     ``5 * ["k", "tab:red"]``.
      :type colors: list of str, optional
      :param styles: The line styles to be used for the lines of each metric. It must
                     be at least the same size as ``y_axis``. Defaults to
                     ``10 * ["-"]``.
      :type styles: list of str, optional
      :param markers: The markers to be used for the lines of each metric. It must be at
                      least the same size as ``y_axis``. Defaults to ``10 * ["o"]``.
      :type markers: list of str, optional
      :param figsize: The size of the figure in inches. Defaults to ``(10, 5)``.
      :type figsize: tuple of int, optional
      :param plt_yrange: The range of values for the y-axis. Defaults to ``None``.
      :type plt_yrange: tuple of float, optional
      :param plt_xrange: The range of values for the x-axis. Defaults to ``None``.
      :type plt_xrange: tuple of float, optional

      :rtype: None

      .. rubric:: Notes

      This function requires the ``matplotlib`` package.



   .. py:method:: show_sample(set_name = 'train', batch_number = 1, print_batch_info = True, figsize = (15, 10))

      Displays a sample of training or validation data in a grid format with
      their corresponding class labels.

      :param set_name: Name of the dataset (``"train"``/``"validation"``) to display the
                       sample from, by default ``"train"``.
      :type set_name: str, optional
      :param batch_number: Which batch to display, by default ``1``.
      :type batch_number: int, optional
      :param print_batch_info: Whether to print information about the batch size, by default
                               ``True``.
      :type print_batch_info: bool, optional
      :param figsize: Figure size (width, height) in inches, by default ``(15, 10)``.
      :type figsize: tuple, optional

      :returns: Displays the sample images with their corresponding class labels.
      :rtype: None

      :raises StopIteration: If the specified number of batches to display exceeds the total
          number of batches in the dataset.

      .. rubric:: Notes

      This method uses the dataloader of the ``ImageClassifierData`` class
      and the ``torchvision.utils.make_grid`` function to display the sample
      data in a grid format. It also calls the ``_imshow`` method of the
      ``ImageClassifierData`` class to show the sample data.



   .. py:method:: print_batch_info(set_name = 'train')

      Print information about a dataset's batches, samples, and batch-size.

      :param set_name: Name of the dataset to display batch information for (default is
                       ``"train"``).
      :type set_name: str, optional

      :rtype: None



   .. py:method:: show_inference_sample_results(label, num_samples = 6, set_name = 'test', min_conf = None, max_conf = None, figsize = (15, 15))

      Shows a sample of the results of the inference.

      :param label: The label for which to display results.
      :type label: str, optional
      :param num_samples: The number of sample results to display. Defaults to ``6``.
      :type num_samples: int, optional
      :param set_name: The name of the dataset split to use for inference. Defaults to
                       ``"test"``.
      :type set_name: str, optional
      :param min_conf: The minimum confidence score for a sample result to be displayed.
                       Samples with lower confidence scores will be skipped. Defaults to
                       ``None``.
      :type min_conf: float, optional
      :param max_conf: The maximum confidence score for a sample result to be displayed.
                       Samples with higher confidence scores will be skipped. Defaults to
                       ``None``.
      :type max_conf: float, optional
      :param figsize: Figure size (width, height) in inches, displaying the sample
                      results. Defaults to ``(15, 15)``.
      :type figsize: tuple[int, int], optional

      :rtype: None



   .. py:method:: save(save_path = 'default.obj', force = False)

      Save the object to a file.

      :param save_path: The path to the file to write.
                        If the file already exists and ``force`` is not ``True``, a ``FileExistsError`` is raised.
                        Defaults to ``"default.obj"``.
      :type save_path: str, optional
      :param force: Whether to overwrite the file if it already exists. Defaults to
                    ``False``.
      :type force: bool, optional

      :raises FileExistsError: If the file already exists and ``force`` is not ``True``.

      .. rubric:: Notes

      The object is saved in two parts. First, a serialized copy of the
      object's dictionary is written to the specified file using the
      ``joblib.dump`` function. The object's ``model`` attribute is excluded
      from this dictionary and saved separately using the ``torch.save``
      function, with a filename derived from the original ``save_path``.



   .. py:method:: save_predictions(set_name, save_path = None, delimiter = ',')


   .. py:method:: load_dataset(dataset, set_name, batch_size = 16, sampler = None, shuffle = False, num_workers = 0, **kwargs)

      Creates a DataLoader from a PatchDataset and adds it to the ``dataloaders`` dictionary.

      :param dataset: The dataset to add
      :type dataset: PatchDataset
      :param set_name: The name to use for the dataset
      :type set_name: str
      :param batch_size: The batch size to use when creating the DataLoader, by default 16
      :type batch_size: Optional[int], optional
      :param sampler: The sampler to use when creating the DataLoader, by default None
      :type sampler: Optional[Union[Sampler, None]], optional
      :param shuffle: Whether to shuffle the PatchDataset, by default False
      :type shuffle: Optional[bool], optional
      :param num_workers: The number of worker threads to use for loading data, by default 0.
      :type num_workers: Optional[int], optional



   .. py:method:: load(load_path, force_device = False)

      This function loads the state of a class instance from a saved file
      using the joblib library. It also loads a PyTorch model from a
      separate file and maps it to the device used to load the class
      instance.

      :param load_path: Path to the saved file to load.
      :type load_path: str
      :param force_device: Whether to force the use of a specific device, or the name of the
                           device to use. If set to ``True``, the default device is used.
                           Defaults to ``False``.
      :type force_device: bool or str, optional

      :raises FileNotFoundError: If the specified file does not exist.

      :rtype: None



   .. py:method:: cprint(type_info, bc_color, text)

      Print colored text with additional information.

      :param type_info: The type of message to display.
      :type type_info: str
      :param bc_color: The color to use for the message text.
      :type bc_color: str
      :param text: The text to display.
      :type text: str

      :returns: The colored message is displayed on the standard output stream.
      :rtype: None



   .. py:method:: update_progress(progress, text = '', barLength = 30)

      Update the progress bar.

      :param progress: The progress value to display, between ``0`` and ``1``.
                       If an integer is provided, it will be converted to a float.
                       If a value outside the range ``[0, 1]`` is provided, it will be
                       clamped to the nearest valid value.
      :type progress: float or int
      :param text: Additional text to display after the progress bar, defaults to
                   ``""``.
      :type text: str, optional
      :param barLength: The length of the progress bar in characters, defaults to ``30``.
      :type barLength: int, optional

      :raises TypeError: If progress is not a floating point value or an integer.

      :returns: The progress bar is displayed on the standard output stream.
      :rtype: None



.. py:class:: Annotator(patch_df = None, parent_df = None, labels = None, patch_paths = None, parent_paths = None, metadata_path = None, annotations_dir = './annotations', patch_paths_col = 'image_path', label_col = 'label', show_context = False, auto_save = True, delimiter = ',', sortby = None, ascending = True, username = None, task_name = None, min_values = None, max_values = None, filter_for = None, surrounding = 1, max_size = 1000, resize_to = None)

   Annotator class for annotating patches with labels.

   :param patch_df: Path to a CSV file or a pandas DataFrame containing patch data, by default None
   :type patch_df: str or pd.DataFrame or None, optional
   :param parent_df: Path to a CSV file or a pandas DataFrame containing parent data, by default None
   :type parent_df: str or pd.DataFrame or None, optional
   :param labels: List of labels for annotation, by default None
   :type labels: list, optional
   :param patch_paths: Path to patch images, by default None
                       Ignored if patch_df is provided.
   :type patch_paths: str or None, optional
   :param parent_paths: Path to parent images, by default None
                        Ignored if parent_df is provided.
   :type parent_paths: str or None, optional
   :param metadata_path: Path to metadata CSV file, by default None
   :type metadata_path: str or None, optional
   :param annotations_dir: Directory to store annotations, by default "./annotations"
   :type annotations_dir: str, optional
   :param patch_paths_col: Name of the column in which image paths are stored in patch DataFrame, by default "image_path"
   :type patch_paths_col: str, optional
   :param label_col: Name of the column in which labels are stored in patch DataFrame, by default "label"
   :type label_col: str, optional
   :param show_context: Whether to show context when loading patches, by default False
   :type show_context: bool, optional
   :param auto_save: Whether to automatically save annotations, by default True
   :type auto_save: bool, optional
   :param delimiter: Delimiter used in CSV files, by default ","
   :type delimiter: str, optional
   :param sortby: Name of the column to use to sort the patch DataFrame, by default None.
                  Default sort order is ``ascending=True``. Pass ``ascending=False`` keyword argument to sort in descending order.
   :type sortby: str or None, optional
   :param ascending: Whether to sort the DataFrame in ascending order when using the ``sortby`` argument, by default True.
   :type ascending: bool, optional
   :param username: Username to use when saving annotations file, by default None.
                    If not provided, a random string is generated.
   :type username: str or None, optional
   :param task_name: Name of the annotation task, by default None.
   :type task_name: str or None, optional
   :param min_values: A dictionary consisting of column names (keys) and minimum values as floating point values (values), by default None.
   :type min_values: dict, optional
   :param max_values: A dictionary consisting of column names (keys) and maximum values as floating point values (values), by default None.
   :type max_values: dict, optional
   :param filter_for: A dictionary consisting of column names (keys) and values to filter for (values), by default None.
   :type filter_for: dict, optional
   :param surrounding: The number of surrounding images to show for context, by default 1.
   :type surrounding: int, optional
   :param max_size: The size in pixels for the longest side to which constrain each patch image, by default 1000.
   :type max_size: int, optional
   :param resize_to: The size in pixels for the longest side to which resize each patch image, by default None.
   :type resize_to: int or None, optional

   :raises FileNotFoundError: If the provided patch_df or parent_df file path does not exist
   :raises ValueError: If patch_df or parent_df is not a valid path to a CSV file or a pandas DataFrame
       If patch_df or patch_paths is not provided
       If the DataFrame does not have the required columns
       If sortby is not a string or None
       If labels provided are not in the form of a list
   :raises SyntaxError: If labels provided are not in the form of a list


   .. py:method:: get_queue(as_type = 'list')

      Gets the indices of rows which are eligible for annotation.

      :param as_type: The format in which to return the indices. Options: "list",
                      "index". Default is "list". If any other value is provided, it
                      returns a pandas.Series.
      :type as_type: str, optional

      :returns: Depending on "as_type", returns either a list of indices, a
                pd.Index object, or a pd.Series of legible rows.
      :rtype: List[int] or pandas.Index or pandas.Series



   .. py:method:: get_context()

      Provides the surrounding context for the patch to be annotated.

      :returns: An IPython VBox widget containing the surrounding patches for
                context.
      :rtype: ipywidgets.VBox



   .. py:method:: annotate(show_context = None, sortby = None, ascending = None, min_values = None, max_values = None, surrounding = None, resize_to = None, max_size = None)

      Annotate at the patch-level of the current patch.
      Renders the annotation interface for the first image.

      :param show_context: Whether or not to display the surrounding context for each image.
                           Default is None.
      :type show_context: bool or None, optional
      :param sortby: Name of the column to use to sort the patch DataFrame, by default None.
                     Default sort order is ``ascending=True``. Pass ``ascending=False`` keyword argument to sort in descending order.
      :type sortby: str or None, optional
      :param ascending: Whether to sort the DataFrame in ascending order when using the ``sortby`` argument, by default True.
      :type ascending: bool, optional
      :param min_values: Minimum values for each property to filter images for annotation.
                         It should be provided as a dictionary consisting of column names
                         (keys) and minimum values as floating point values (values).
                         Default is None.
      :type min_values: dict or None, optional
      :param max_values: Maximum values for each property to filter images for annotation.
                         It should be provided as a dictionary consisting of column names
                         (keys) and minimum values as floating point values (values).
                         Default is None
      :type max_values: dict or None, optional
      :param surrounding: The number of surrounding images to show for context. Default: 1.
      :type surrounding: int or None, optional
      :param max_size: The size in pixels for the longest side to which constrain each
                       patch image. Default: 100.
      :type max_size: int or None, optional

      .. rubric:: Notes

      This method is a wrapper for the ``_annotate`` method.



   .. py:method:: render()

      Displays the image at the current index in the annotation interface.

      If the current index is greater than or equal to the length of the
      dataframe, the method disables the "next" button and saves the data.

      :rtype: None



   .. py:method:: get_patch_image(ix)

      Returns the image at the given index.

      :param ix: The index of the image in the dataframe.
      :type ix: int | str

      :returns: A PIL.Image object of the image at the given index.
      :rtype: PIL.Image



   .. py:method:: get_labelled_data(sort = True, index_labels = False, include_paths = True)

      Returns the annotations made so far.

      :param sort: Whether to sort the dataframe by the order of the images in the
                   input data, by default True
      :type sort: bool, optional
      :param index_labels: Whether to return the label's index number (in the labels list
                           provided in setting up the instance) or the human-readable label
                           for each row, by default False
      :type index_labels: bool, optional
      :param include_paths: Whether to return a column containing the full path to the
                            annotated image or not, by default True
      :type include_paths: bool, optional

      :returns: A dataframe containing the labelled images and their associated
                label index.
      :rtype: pandas.DataFrame



   .. py:property:: filtered
      :type: pandas.DataFrame



   .. py:method:: render_complete()

      Renders the completion message once all images have been annotated.

      :rtype: None



.. py:function:: print_version()

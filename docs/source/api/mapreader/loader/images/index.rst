:py:mod:`mapreader.loader.images`
=================================

.. py:module:: mapreader.loader.images


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   mapreader.loader.images.mapImages




.. py:class:: mapImages(path_images=False, tree_level='parent', parent_path=None, **kwds)

   mapImages class

   .. py:method:: imagesConstructor(image_path, parent_path=None, tree_level='child', **kwds)

      Construct images instance variable

      Parameters
      ----------
      image_path : str or None
          Path to image
      parent_path : str or None, optional
          Path to parent image (if applicable), by default None
      tree_level : str, optional
          Tree level, choices between "parent" or "child", by default "child"


   .. py:method:: splitImagePath(inp_path)
      :staticmethod:

      Split 'inp_path' into basename and dirname


   .. py:method:: add_metadata(metadata, columns=None, tree_level='parent', index_col=0, delimiter='|')

      Add metadata to images at tree_level

      Parameters
      ----------
      metadata : str or DataFrame
          Path to csv file (normally created from a pd.DataFrame) or pd.DataFrame containing metadata
      columns : list, optional
          List of columns to be added, by default None
      tree_level : str, optional
          Tree level, choices between "parent" or "child", by default "parent"
      index_col : int, optional
          Column in csv file to use as 'index' when creating pd.DataFrame, by default 0
      delimiter : str, optional
          Delimiter to use when creating pd.DataFrame, by default "|"


   .. py:method:: show_sample(num_samples, tree_level='parent', random_seed=65, **kwds)

      Show sample images

      Parameters
      ----------
      num_samples : int
          Number of samples to be plotted
      tree_level : str, optional
          Tree level, choices between "parent" or "child", by default "parent"
      random_seed : int, optional
          Random seed to use for reproducibility, by default 65


   .. py:method:: list_parents()

      Return list of all parents


   .. py:method:: list_children()

      Return list of all children


   .. py:method:: add_shape(tree_level='parent')

      Run add_shape_id for all tree_level items


   .. py:method:: add_coord_increments(tree_level='parent', verbose=False)

      Run `add_coord_increments_id` for each image at "tree_level"

      Parameters
      ----------
      tree_level : str, optional
          Tree level, choices between "parent" or "child, by default "parent"
      verbose : bool, optional
          If true, print verbose outputs, by default False


   .. py:method:: add_center_coord(tree_level='child', verbose=False)

      Run `add_center_coord_id` for each image at "tree_level"

      Parameters
      ----------
      tree_level : str, optional
          Tree level, choices of "parent" or "child, by default "child"
      verbose : bool, optional
          If True, print verbose outputs, by default False


   .. py:method:: add_shape_id(image_id, tree_level='parent')

      Add an image/array shape to image

      Parameters
      ----------
      image_id : str
          Image ID
      tree_level : str, optional
          Tree level, choices of "parent" or "child, by default "parent"


   .. py:method:: add_coord_increments_id(image_id, tree_level='parent', verbose=False)

      Add pixel-wise dlon and dlat to image

      Parameters
      ----------
      image_id : str
          Image ID
      tree_level : str, optional
          Tree level, choices between "parent" or "child, by default "parent"
      verbose : bool, optional
          If True, print verbose outputs, by default False


   .. py:method:: add_center_coord_id(image_id, tree_level='child', verbose=False)

      Add center_lon and center_lat to image

      Parameters
      ----------
      image_id :str
          Image ID
      tree_level : str, optional
          Tree level, choices between "parent" or "child, by default "child"
      verbose : bool, optional
          If True, print verbose outputs, by default False


   .. py:method:: calc_pixel_width_height(parent_id, calc_size_in_m='great-circle', verbose=False)

      Calculate width and height of pixels

      Parameters
      ----------
      parent_id : str
          ID of the parent image
      calc_size_in_m : str, optional
          Method to compute pixel widths and heights, choices between "geodesic" and "great-circle" or "gc", by default "great-circle"
      verbose : bool, optional
          If true, print verbose outputs, by default False

      Returns
      -------
      tuple
          size_in_m (bottom, top, left, right)



   .. py:method:: sliceAll(method='pixel', slice_size=100, path_save='sliced_images', square_cuts=False, resize_factor=False, output_format='png', rewrite=False, verbose=False, tree_level='parent', add2child=True, id1=0, id2=-1)

      Slice all images at the specified 'tree_level'

      Parameters
      ----------
      method : str, optional
          Method used to slice images, choices between "pixel" and "meters" or "meter", by default "pixel"
      slice_size : int, optional
          Number of pixels/meters in both x and y to use for slicing, by default 100
      path_save : str, optional
          Directory to save the sliced images, by default "sliced_images"
      square_cuts : bool, optional
          If True, all sliced images will have the same number of pixels in x and y, by default False
      resize_factor : bool, optional
          If True, resize the images before slicing, by default False
      output_format : str, optional
          Format to use when writing image files, by default "png"
      rewrite : bool, optional
          If True, existing slices will be rewritten, by default False
      verbose : bool, optional
          If True, progress updates will be printed throughout, by default False
      tree_level : str, optional
          Tree level, choices between "parent" or "child, by default "parent"
      add2child : bool, optional
          If True, sliced images will be added to `self.images` dictionary, by default True
      id1 : int, optional
          First image to slice, by default 0
      id2 : int, optional
          Last image to slice, by default -1


   .. py:method:: addChildren()

      Add children to parent


   .. py:method:: calc_pixel_stats(parent_id=None, calc_mean=True, calc_std=True)

      Calculate pixel stats (R, G, B, RGB and, if present, Alpha) of each child in a parent_id and store the results

      Parameters
      ----------
      parent_id : str, list or None, optional
          ID of the parent image(s). 
          If None, all parents will be used, by default None
      calc_mean : bool, optional
          Calculate mean values, by default True
      calc_std : bool, optional
          Calculate standard deviations, by default True


   .. py:method:: convertImages()

      Convert images dictionary into a pd.DataFrame

      Returns
      -------
      list
          list of dataframes containing information about parents and children.


   .. py:method:: show_par(parent_id, value=False, **kwds)

      A wrapper function for `.show()` which plots all children of a specified parent

      Parameters
      ----------
      parent_id : str
          ID of the parent image to be plotted
      value : list or bool, optional
          Value to be plotted on each child image, by default False
          See `.show()` for more detail.


   .. py:method:: show(image_ids, value=False, plot_parent=True, border=True, border_color='r', vmin=0.5, vmax=2.5, colorbar='viridis', alpha=1.0, discrete_colorbar=256, tree_level='child', grid_plot=(20000, 20000), plot_histogram=True, save_kml_dir=False, image_width_resolution=None, kml_dpi_image=None, **kwds)

      Plot images from a list of image ids

      Parameters
      ----------
      image_ids : str or list
          Image or list of images to be plotted
      value : str, list or bool, optional
          Value to plot on child images, by default False
      plot_parent : bool, optional
          If true, parent image will be plotted in background, by default True
      border : bool, optional
          If true, border will be placed around each child image, by default True
      border_color : str, optional
          Border colour, by default "r"
      vmin : float, optional
          Minimum value for the colorbar, by default 0.5
      vmax : float, optional
          Maximum value for the colorbar, by default 2.5
      colorbar : str, optional
          Colorbar used to visualise chosen `value`, by default "viridis"
      alpha : float, optional
          Transparancy level for plotting `value` (0 = transparent, 1 = opaque), by default 1.0
      discrete_colorbar : int, optional
          Number of discrete colurs to use in colorbar, by default 256
      grid_plot : tuple, optional
          Number of rows and columns to use in the image, later adjusted to the true min/max of all subplots, by default (20000, 20000)
      plot_histogram : bool, optional
          Plot a histogram of `value`, by default True
      save_kml_dir : str or bool, optional
          Directory to save KML files
          If False, no files are saved, by default False
      image_width_resolution : int or None, optional
          Pixel width to be used for plotting, only when tree_level="parent"
          If None,  by default None
      kml_dpi_image : int or None, optional
          The resolution, in dots per inch, to create KML images when `save_kml_dir` is specified, by default None


   .. py:method:: loadPatches(patch_paths, parent_paths=False, add_geo_par=False, clear_images=False)

      Load patches from path and, if parent_paths specified, add parents

      Parameters
      ----------
      patch_paths : str
          Path to patches, accepts wildcards
      parent_paths : str or bool, optional
          Path to parents, accepts wildcards
          If False, no parents are loaded, by default False
      add_geo_par : bool, optional
          Add geographical info to parents, by default False
      clear_images : bool, optional
          Clear images variable before loading, by default False


   .. py:method:: detectParIDfromPath(image_id, parent_delimiter='#')
      :staticmethod:

      Detect parent IDs from image ID

      Parameters
      ----------
      image_id : str
          ID of child image
      parent_delimiter : str, optional
          Delimiter used to separate parent ID when naming child image, by default "#"

      Returns
      -------
      str
          Parent ID


   .. py:method:: detectBorderFromPath(image_id, border_delimiter='-')
      :staticmethod:

      Detects borders from the path assuming child image is named using the following format:
      str-min_x-min_y-max_x-max_y-str

      Parameters
      ----------
      image_id : str
          ID of image
      border_delimiter : str, optional
          Delimiter used to separate border values when naming child image, by default "-"

      Returns
      -------
      tuple
          Border (min_x, min_y, max_x, max_y) of image


   .. py:method:: loadParents(parent_paths=False, parent_ids=False, update=False, add_geo=False)

      Load parent images from file paths.
      If no path is given, only `parent_ids`, no image_path will be added to the images.

      Parameters
      ----------
      parent_paths : str or bool, optional
          Path to parent images, by default False
      parent_ids : list, str or bool, optional
          ID(s) of parent images
          Ignored if parent_paths are specified, by default False
      update : bool, optional
          If true, current parents will be overwritten, by default False
      add_geo : bool, optional
          If true, geographical info will be added to parents, by default False


   .. py:method:: loadDataframe(parents=None, children_df=None, clear_images=True)

      Form images variable from dataframe(s)

      Parameters
      ----------
      parents : DataFrame, str or None, optional
          DataFrame containing parents or path to parents, by default None
      children_df : DataFrame or None, optional
          DataFrame containing children (patches), by default None
      clear_images : bool, optional
          If true, clear images before reading the dataframes, by default True


   .. py:method:: load_csv_file(parent_path=None, child_path=None, clear_images=False, index_col_child=0, index_col_parent=0)

      Form images variable from csv file(s)

      Parameters
      ----------
      parent_path : _type_, optional
          Path to parent csv file, by default None
      child_path : _type_, optional
          Path to child csv file, by default None
      clear_images : bool, optional
          If true, clear images before reading the csv files, by default False
      index_col_child : int, optional
          Column in child csv file to use as index, by default 0
      index_col_parent : int, optional
          Column in parent csv file to use as index, by default 0


   .. py:method:: addGeoInfo(proj2convert='EPSG:4326', calc_method='great-circle', verbose=False)

      Add geographic information (shape, coords, size in m) to images from image metadata

      Parameters
      ----------
      proj2convert : str, optional
          Projection to convert coordinates into, by default "EPSG:4326"
      calc_method : str, optional
          Method to compute pixel widths and heights, choices between "geodesic" and "great-circle" or "gc", by default "great-circle"
      verbose : bool, optional
          If True, print verbose outputs, by default False




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

      Construct images instance variable,

      Arguments:
          image_path {str or None} -- Path to the image

      Keyword Arguments:
          parent_path {str or None} -- Path to the parent of image (default: {None})
          tree_level {str} -- Tree level, choices between parent and child (default: {"child"})


   .. py:method:: splitImagePath(inp_path)
      :staticmethod:

      split image path into basename and dirname,


   .. py:method:: add_metadata(metadata, columns=None, tree_level='parent', index_col=0, delimiter='|')

      Add metadata to images at tree_level,

      Args:
          metadata_path (path): path to a csv file, normally created from a pandas dataframe
          columns (list, optional): list of columns to be used. If None (default), all columns are used.
          tree_level (str, optional): parent/child tree level. Defaults to "parent".
          index_col (int, optional): index column


   .. py:method:: show_sample(num_samples, tree_level='parent', random_seed=65, **kwds)

      Show sample images,

      Arguments:
          num_samples {int} -- Number of samples to be plotted

      Keyword Arguments:
          tree_level {str} -- XXX (default: {"child"})
          random_seed {int} -- Random seed for reproducibility (default: {65})


   .. py:method:: list_parents()

      Return list of all parents


   .. py:method:: list_children()

      Return list of all children


   .. py:method:: add_shape(tree_level='parent')

      Run add_shape_id for all tree_level items


   .. py:method:: add_coord_increments(tree_level='parent')

      Run add_coord_increments_id for all tree_level items


   .. py:method:: add_center_coord(tree_level='child')

      Run add_center_coord_id for all tree_level items


   .. py:method:: add_shape_id(image_id, tree_level='parent')

      Add an image/array shape to self.images[tree_level][image_id]

      Parameters
      ----------
      image_id : str
          image ID
      tree_level : str, optional
          Tree level, choices between parent and child (default: {"child"})


   .. py:method:: add_coord_increments_id(image_id, tree_level='parent')

      Add pixel-wise dlon and dlat to self.images[tree_level][image_id]

      Parameters
      ----------
      image_id : str
          image ID
      tree_level : str, optional
          Tree level, choices between parent and child (default: {"child"})


   .. py:method:: add_center_coord_id(image_id, tree_level='child')

      Add center_lon and center_lat to self.images[tree_level][image_id]

      Parameters
      ----------
      image_id : str
          image ID
      tree_level : str, optional
          Tree level, choices between parent and child (default: {"child"})


   .. py:method:: calc_pixel_width_height(parent_id, calc_size_in_m='great-circle')

      Calculate width and height of pixels

      Args:
          parent_id (str): ID of the parent image
          calc_size_in_m (str, optional): How to compute the width/heigh, options: geodesic and great-circle (default).


   .. py:method:: sliceAll(method='pixel', slice_size=100, path_save='test', square_cuts=False, resize_factor=False, output_format='PNG', rewrite=False, verbose=False, tree_level='parent', add2child=True, id1=0, id2=-1)

      Slice all images in the object (the list can be accessed via .images variable)

      Keyword Arguments:
          method {str} -- method to slice an image (default: {"pixel"})
          slice_size {int} -- Number of pixels in both x and y directions (default: {100})
          path_save {str} -- Directory to save the sliced images (default: {"test"})
          square_cuts {bool} -- All sliced images will have the same number of pixels in x and y (default: {True})
          resize_factor {bool} -- Resize image before slicing (default: {False})
          output_format {str} -- Output format (default: {"PNG"})
          tree_level {str} -- image group to be sliced (default: {"parent"})
          verbose {bool} -- Print the progress (default: {False})


   .. py:method:: addChildren()

      Add children to parent


   .. py:method:: calc_pixel_stats(parent_id=None, calc_mean=True, calc_std=True)

      Calculate stats of each child in a parent_id and
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


   .. py:method:: show(image_ids, value=False, plot_parent=True, border=True, border_color='r', vmin=0.5, vmax=2.5, colorbar='jet', alpha=1.0, discrete_colorbar=256, tree_level='child', grid_plot=(20000, 20000), plot_histogram=True, save_kml_dir=False, image_width_resolution=None, kml_dpi_image=None, **kwds)

      Plot a list of image ids,

      Arguments:
          image_ids {list} -- List of image ids to be plotted

      Keyword Arguments:
          value {False or list} -- Value to be plotted on child images
          plot_parent {bool} -- Plot parent image in the background (default: {True})
          border {bool} -- Plot a border for each image id (default: {True})
          border_color {str} -- color of patch borders (default: {r})
          vmin {float or list} -- min. value for the colorbar (default: {0.5})
          vmax {float or list} -- max. value for the colorbar (default: {2.5})
          colorbar {str or list} -- colorbar to visualize "value" on maps (default: {jet})
          alpha {float or list} -- set transparency level for plotting "value" on maps (default: {1.})
          discrete_colorbar {int or list} -- number of discrete colors to be used (default: {256})
          tree_level {str} -- Tree level for the plot XXX (default: {"child"})
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
      Here, the assumption is that the child image is named:
      NOTE: STRING-min_x-min_y-max_x-max_y-STRING


   .. py:method:: loadParents(parent_paths=False, parent_ids=False, update=False, add_geo=False)

      load parent images from files (parent_paths)
         if only parent_ids is specified, self.images["parent"] will be filled with no image_path.
         NOTE: if parent_paths is given, parent_ids will be omitted as ids will be
               detected from the basename

      Keyword Arguments:
          parent_paths {False or str, wildcard accepted} -- path to parents (default: {False})
          parent_ids {False or list/tuple} -- list of parent ids (default: {False})


   .. py:method:: loadDataframe(parents=None, children_df=None, clear_images=True)

      Read dataframes and form images variable

      Keyword Arguments:
          parents_df {dataframe or path} -- Parents dataframe or path to parents (default: {None})
          children_df {dataframe} -- Children/slices dataframe (default: {None})
          clear_images {bool} -- clear images before reading dataframes (default: {True})


   .. py:method:: load_csv_file(parent_path=None, child_path=None, clear_images=False, index_col_child=0, index_col_parent=0)

      Read parent and child from CSV files




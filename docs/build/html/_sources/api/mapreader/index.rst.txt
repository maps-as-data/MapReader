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
   loader/index.rst
   process/index.rst
   slicers/index.rst
   train/index.rst
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



.. py:function:: loader(path_images=False, tree_level='parent', parent_path=None, **kwds)

   Construct mapImages object by passing the image path,

   Keyword Arguments:
       path_images {str or False} -- path to one or many images

   Returns:
       [mapImages object] -- mapImages object contains various methods to work with images


.. py:function:: load_patches(patch_paths, parent_paths=False, add_geo_par=False, clear_images=False)


.. py:class:: TileServer(metadata_path, geometry='polygone', download_url='https://mapseries-tilesets.s3.amazonaws.com/1inch_2nd_ed/{z}/{x}/{y}.png')

   .. py:method:: create_info()

      Extract information from metadata and create metadata_info_list and metadata_coord_arr.

      This is a helper function for other methods in this class


   .. py:method:: modify_metadata(remove_image_ids=[], only_keep_image_ids=[])

      Modify metadata using metadata[...]["properties"]["IMAGE"]

      Parameters
      ----------
      remove_image_ids : list, optional
          Image IDs to be removed from metadata variable


   .. py:method:: query_point(latlon_list, append=False)

      Query maps from a list of lats/lons using metadata file

      Args:
          latlon_list (list): a list that contains lats/lons: [lat, lon] or [[lat1, lon1], [lat2, lon2], ...]
          append (bool, optional): If True, append the new query to the list of queries. Defaults to False.


   .. py:method:: print_found_queries()

      Print found queries


   .. py:method:: detect_rectangle_boundary(coords)

      Detect rectangular boundary given a set of coordinates


   .. py:method:: create_metadata_query()

      Create a metadata type variable out of all queries.
      This will be later used in download_tileserver method


   .. py:method:: minmax_latlon()

      Method to return min/max of lats/lons


   .. py:method:: download_tileserver(mode='queries', num_img2test=-1, zoom_level=14, adjust_mult=0.005, retries=10, scraper_max_connections=4, failed_urls_path='failed_urls.txt', tile_tmp_dir='tiles', output_maps_dirname='maps', output_metadata_filename='metadata.csv', pixel_closest=None, redownload=False, id1=0, id2=-1, error_path='errors.txt', max_num_errors=20)

      Download maps via tileserver

      Args:
          mode (str): specify the set of maps to be downloaded:
                      mode = query or queries: this will download the queried maps
                      mode = all: download all maps in the metadata file
          num_img2test (int, optional): Number of images to download for testing. Defaults to -1 (all maps).
          zoom_level (int, optional): Zoom level for maps to be downloaded. Defaults to 14.
          adjust_mult (float, optional): If some tiles cannot be downloaded, shrink the requested bounding box.
                                         by this factor. Defaults to 0.005.
          retries (int, optional): If a tile cannot be downloaded, retry these many times. Defaults to 1.
          failed_urls_path (str, optional): File that contains info about failed download attempts. Defaults to "failed_urls.txt".
          tile_tmp_dir (str, optional): Save tmp files in this directory. Defaults to "tiles".
          output_maps_dirname (str, optional): Path to save downloaded maps. Defaults to "maps".
          output_metadata_filename (str, optional): Path to save metada for downloaded maps. Defaults to "metadata.csv".
                                                    this file will be saved at output_maps_dirname/output_metadata_filename
          pixel_closest (int): adjust the number of pixels in both directions (width and height) after downloading a map
                               for example, if pixel_closest = 100, number of pixels in both directions will be multiples of 100
                               this helps to create only square tiles in processing step
          redownload (bool): if False, only maps that do not exist in the local directory will be retrieved
          id1, id2: consider metadata[id1:id2]


   .. py:method:: extract_region_dates_metadata(one_item)

      Extract name of the region and surveyed/revised/published dates

      Parameters
      ----------
      one_item : dict
          dictionary which contains at least properties/WFS_TITLE


   .. py:method:: find_and_clean_date(ois, ois_key='surveyed')
      :staticmethod:

      Given a string (ois) and a key (ois_key), extract date

      Parameters
      ----------
      ois : str
          string that contains date info
      ois_key : str, optional
          type of date, e.g., surveyed/revised/published


   .. py:method:: plot_metadata_on_map(list2remove=[], map_extent=None, add_text=False)

      Plot the map boundaries specified in metadata

      Args:
          list2remove (list, optional): List of IDs to be removed. Defaults to [].
          map_extent (list or None, optional): Extent of the main map [min_lon, max_lon, min_lat, max_lat]. Defaults to None.
          add_text (bool, optional): Add image IDs to the figure


   .. py:method:: hist_published_dates(min_date=None, max_date=None)

      Plot a histogram for published dates

      Parameters
      ----------
      min_date : int, None
          min date for histogram
      max_date : int, None
          max date for histogram


   .. py:method:: download_tileserver_rect(mode='queries', num_img2test=-1, zoom_level=14, adjust_mult=0.005, retries=1, failed_urls_path='failed_urls.txt', tile_tmp_dir='tiles', output_maps_dirname='maps', output_metadata_filename='metadata.csv', pixel_closest=None, redownload=False, id1=0, id2=-1, min_lat_len=0.05, min_lon_len=0.05)

      Download maps via tileserver

      Args:
          mode (str): specify the set of maps to be downloaded:
                      mode = query or queries: this will download the queried maps
                      mode = all: download all maps in the metadata file
          num_img2test (int, optional): Number of images to download for testing. Defaults to -1 (all maps).
          zoom_level (int, optional): Zoom level for maps to be downloaded. Defaults to 14.
          adjust_mult (float, optional): If some tiles cannot be downloaded, shrink the requested bounding box.
                                         by this factor. Defaults to 0.005.
          retries (int, optional): If a tile cannot be downloaded, retry these many times. Defaults to 1.
          failed_urls_path (str, optional): File that contains info about failed download attempts. Defaults to "failed_urls.txt".
          tile_tmp_dir (str, optional): Save tmp files in this directory. Defaults to "tiles".
          output_maps_dirname (str, optional): Path to save downloaded maps. Defaults to "maps".
          output_metadata_filename (str, optional): Path to save metada for downloaded maps. Defaults to "metadata.csv".
                                                    this file will be saved at output_maps_dirname/output_metadata_filename
          pixel_closest (int): adjust the number of pixels in both directions (width and height) after downloading a map
                               for example, if pixel_closest = 100, number of pixels in both directions will be multiples of 100
                               this helps to create only square tiles in processing step
          redownload (bool): if False, only maps that do not exist in the local directory will be retrieved
          id1, id2: consider metadata[id1:id2]



.. py:class:: loadAnnotations

   .. py:method:: load_all(csv_paths, **kwds)


   .. py:method:: load(csv_path, path2dir=None, col_path='image_id', keep_these_cols=False, append=True, col_label='label', shuffle_rows=True, reset_index=True, random_state=1234)

      Read/append annotation file(s)

      Parameters
      ----------
      csv_path : str
          path to an annotation file in CSV format
      path2dir : str, optional
          update col_path by adding path2dir/col_path, by default None
      col_path : str, optional
          column that contains image paths, by default "image_id"
      keep_these_cols : bool, optional
          only keep these columns, if False (default), all columns will be kept
      append : bool, optional
          append a newly read csv file to self.annotations, by default True
      col_label : str
          Name of the column that contains labels
      shuffle_rows : bool
          Shuffle rows after reading annotations


   .. py:method:: set_col_label(new_label: str = 'label')

      Set the name of the column that contains labels

      Parameters
      ----------
      new_label : str, optional
          Name of the column that contains labels, by default "label"


   .. py:method:: show_image(indx: int, cmap='viridis')

      Show an image by its index (i.e., iloc in pandas)

      Parameters
      ----------
      indx : int
          Index of the image to be plotted


   .. py:method:: adjust_labels(shiftby: int = -1)

      Shift labels by the specified value (shiftby)

      Parameters
      ----------
      shiftby : int, optional
          shift values of self.col_label by shiftby, i.e., self.annotations[self.col_label] + shiftby, by default -1


   .. py:method:: review_labels(tar_label: Union[None, int] = None, start_indx: int = 1, chunks: int = 8 * 6, num_cols: int = 8, figsize: Union[list, tuple] = (8 * 3, 8 * 2), exclude_df=None, include_df=None, deduplicate_col: str = 'image_id')

      Review/edit labels

      Parameters
      ----------
      tar_label : Union[None, int], optional
      start_indx : int, optional
      chunks : int, optional
      num_cols : int, optional
      figsize : Union[list, tuple], optional


   .. py:method:: show_image_labels(tar_label=1, num_sample=10)

      Show sample images for the specified label

      Parameters
      ----------
      tar_label : int, optional
          target label to be used in plotting, by default 1
      num_sample : int, optional
          number of samples to plot, by default 10


   .. py:method:: split_annotations(stratify_colname='label', frac_train=0.7, frac_val=0.15, frac_test=0.15, random_state=1364)

      Split pandas dataframe into three subsets.

      CREDIT: https://stackoverflow.com/a/60804119 (with minor changes)

      Following fractional ratios provided by the user, where each subset is
      stratified by the values in a specific column (that is, each subset has
      the same relative frequency of the values in the column). It performs this
      splitting by running train_test_split() twice.

      Parameters
      ----------
      stratify_colname : str
          The name of the column that will be used for stratification.
      frac_train : float
      frac_val   : float
      frac_test  : float
          The ratios with which the dataframe will be split into train, val, and
          test data. The values should be expressed as float fractions and should
          sum to 1.0.
      random_state : int, None, or RandomStateInstance
          Value to be passed to train_test_split().

      Returns
      -------
      df_train, df_val, df_test :
          Dataframes containing the three splits.


   .. py:method:: sample_labels(tar_label, num_samples, random_state=12345)



.. py:class:: patchTorchDataset(patchframe, transform=None, label_col='label', convert2='RGB', input_col=0)

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



.. py:class:: patchContextDataset(patchframe, transform1=None, transform2=None, label_col='label', convert2='RGB', input_col=0, context_save_path='./maps/maps_context', create_context=False, par_path='./maps', x_offset=1.0, y_offset=1.0, slice_method='scale')

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

   .. py:method:: save_parents(num_req_p=10, sleep_time=0.001, use_parhugin=True, par_split='#', loc_split='-', overwrite=False)


   .. py:method:: save_parents_idx(idx, par_split='#', loc_split='-', overwrite=False, return_image=False)


   .. py:method:: return_orig_image(idx)


   .. py:method:: plot_sample(indx)



.. py:class:: classifier(device='default')

   .. py:method:: set_classnames(classname_dict)

      Set names of the classes in the dataset

      Parameters
      ----------
      classname_dict : dictionary
          name of the classes in the dataset,
          e.g., {0: "rail space", 1: "No rail space"}


   .. py:method:: add2dataloader(dataset, set_name=None, batch_size=16, shuffle=True, num_workers=0, **kwds)

      Create and add a dataloader

      Parameters
      ----------
      dataset : pytorch dataset
      set_name : name of the dataset, e.g., train/val/test, optional
      batch_size : int, optional
      shuffle : bool, optional
      num_workers : int, optional


   .. py:method:: print_classes_dl(set_name: str = 'train')

      Print classes and classnames (if available)

      Parameters
      ----------
      set_name : str, optional
          Name of the dataset (normally specified in self.add2dataloader), by default "train"


   .. py:method:: add_model(model, input_size=224, is_inception=False)

      Add a model to classifier object

      Parameters
      ----------
      model : PyTorch model
          See: from torchvision import models
      input_size : int, optional
          input size, by default 224
      is_inception : bool, optional
          is this a inception-type model?, by default False


   .. py:method:: del_model()

      Delete the model


   .. py:method:: layerwise_lr(min_lr: float, max_lr: float, ltype: str = 'linspace')

      Define layer-wise learning rates

      linspace: use evenly spaced learning rates over a specified interval
      geomspace: use learning rates spaced evenly on a log scale (a geometric progression)

      Parameters
      ----------
      min_lr : float
          minimum learning rate
      max_lr : float
          maximum learning rate
      ltype : str, optional
          how to space the specified interval, by default "linspace"


   .. py:method:: initialize_optimizer(optim_type: str = 'adam', params2optim='infer', optim_param_dict: dict = {'lr': 0.001}, add_optim: bool = True)

      Initialize an optimizer
      if add_optim is True, the optimizer will be added to object

      Note that the first argument of an optimizer is:
          parameters to optimize, e.g.,
              model_ft.parameters(): all parameters are being optimized
              model_ft.fc.parameters(): only parameters of final layer are being optimized
              params2optimize = model_ft.parameters()
          Here, we use filter(lambda p: p.requires_grad, self.model.parameters())

      Parameters
      ----------
      optim_type : str, optional
          optimizer type, e.g., adam, sgd, by default "adam"
      optim_param_dict : dict, optional
          optimizer parameters, by default {"lr": 1e-3}
      add_optim : bool, optional
          add optimizer to the object, by default True


   .. py:method:: add_optimizer(optimizer)

      Add an optimizer to the object


   .. py:method:: initialize_scheduler(scheduler_type: str = 'steplr', scheduler_param_dict: dict = {'step_size': 10, 'gamma': 0.1}, add_scheduler: bool = True)

      Initialize a scheduler

      Parameters
      ----------
      scheduler_type : str, optional
          scheduler type, by default "steplr"
      scheduler_param_dict : dict, optional
          scheduler parameters, by default {"step_size": 10, "gamma": 0.1}
      add_scheduler : bool, optional
          add scheduler to the object, by default True


   .. py:method:: add_scheduler(scheduler)

      Add a scheduler to the object


   .. py:method:: add_criterion(criterion)

      Add a criterion to the object


   .. py:method:: model_summary(only_trainable=False, print_space=[40, 20, 20])

      Print model summary

      Credit: this function is the modified version of https://stackoverflow.com/a/62508086

      Other ways to check params:
          sum(p.numel() for p in myclassifier.model.parameters())
          sum(p.numel() for p in myclassifier.model.parameters() if p.requires_grad)

          # Also:
          for name, param in self.model.named_parameters():
              n = name.split(".")[0].split("_")[0]
              print(name, param.requires_grad)

      Parameters
      ----------
      only_trainable : bool, optional
          print only trainable params, by default False
      print_space : list, optional
          print params, how many spaces should be added in each column, by default [40, 20, 20]


   .. py:method:: freeze_layers(layers_to_freeze: list = [])

      Freeze a list of layers, wildcard is accepted

      Parameters
      ----------
      layers_to_freeze : list, optional
          List of layers to freeze, by default []


   .. py:method:: unfreeze_layers(layers_to_unfreeze: list = [])

      Unfreeze a list of layers, wildcard is accepted

      Parameters
      ----------
      layers_to_unfreeze : list, optional
          List of layers to unfreeze, by default []


   .. py:method:: only_keep_layers(only_keep_layers_list: list = [])

      Only keep this list of layers in training

      Parameters
      ----------
      only_keep_layers_list : list, optional
          List of layers to keep, by default []


   .. py:method:: inference(set_name='infer', verbosity_level=0, print_info_batch_freq: int = 5)

      Model inference on dataset: set_name


   .. py:method:: train_component_summary()

      Print some info about optimizer/criterion/model...


   .. py:method:: train(phases: list = ['train', 'val'], num_epochs: int = 25, save_model_dir: Union[None, str] = 'models', verbosity_level: int = 1, tensorboard_path: Union[None, str] = None, tmp_file_save_freq: int = 2, remove_after_load: bool = True, print_info_batch_freq: int = 5)

      Wrapper function for train_core method to capture exceptions. Supported exceptions so far:
      - KeyboardInterrupt

      Refer to train_core for more information.


   .. py:method:: train_core(phases: list = ['train', 'val'], num_epochs: int = 25, save_model_dir: Union[None, str] = 'models', verbosity_level: int = 1, tensorboard_path: Union[None, str] = None, tmp_file_save_freq: int = 2, print_info_batch_freq: int = 5)

      Train/fine-tune a classifier

      Parameters
      ----------
      phases : list, optional
          at each epoch, perform this list of phases, e.g., train and val, by default ["train", "val"]
      num_epochs : int, optional
          number of epochs, by default 25
      save_model_dir : Union[None, str], optional
          Parent directory to save models, by default "models"
      verbosity_level : int, optional
          verbosity level: -1 (quiet), 0 (normal), 1 (verbose), 2 (very verbose), 3 (debug)
      tensorboard_path : Union[None, str], optional
          Parent directory to save tensorboard files, by default None
      tmp_file_save_freq : int, optional
          frequency (in epoch) to save a temporary checkpoint


   .. py:method:: calculate_add_metrics(y_true, y_pred, y_score, phase, epoch=-1, tboard_writer=None)

      Calculate various evaluation metrics (e.g., precision, recall and F1) and add to self.metrics

      Parameters
      ----------
      y_true : list
          Ground truth (correct) target values
      y_pred : list
          Estimated targets as returned by a classifier.
      y_score : list
          Target scores
      phase : str
          Specified phase in training (see train function)
      epoch : int
          Epoch
      tboard_writer : optional
          tensorboard writer initialized by SummaryWriter, by default None


   .. py:method:: gen_epoch_msg(phase, epoch_msg)


   .. py:method:: plot_metric(y_axis, y_label, legends, x_axis='epoch', x_label='epoch', colors=5 * ['k', 'tab:red'], styles=10 * ['-'], markers=10 * ['o'], figsize=(10, 5), plt_yrange=None, plt_xrange=None)

      Plot content of self.metrics

      Parameters
      ----------
      y_axis : list
          items to be plotted on y-axis
      y_label : list
      legends : list
      x_axis : str, optional
          item to be plotted on x-axis, by default "epoch"
      x_label : str, optional
      colors : list, optional
          list of colors, at least the same size as y_axis, by default 5*["k", "tab:red"]
      styles : list, optional
          list of line styles, at least the same size as y_axis, by default 10*["-"]
      markers : list, optional
          list of line markers, at least the same size as y_axis, by default 10*["o"]
      figsize : tuple, optional
      plt_yrange : list, optional
      plt_xrange : list, optional


   .. py:method:: initialize_model(model_name, pretrained=True, last_layer_num_classes='default', add_model=True)

      Initialize a PyTorch model
      This method changes the number of classes in the last layer (see last_layer_num_classes)

      NOTES
      -----
      inception_v3 requires the input size to be (299,299), whereas all of the other models expect (224,224).

      models:see https://pytorch.org/vision/0.8/models.html)

      Parameters
      ----------
      model_name : str
          Name of a PyTorch model, see https://pytorch.org/vision/0.8/models.html
      pretrained : bool, optional
          Use pretrained version, by default True
      last_layer_num_classes : str, optional
          Number of elements in the last layer, by default "default"


   .. py:method:: show_sample(set_name='train', batch_number=1, print_batch_info=True, figsize=(15, 10))

      Show samples from specified dataset

      Parameters
      ----------
      set_name : str, optional
          name of the dataset, by default "train"
      batch_number : int, optional
          batch number to be plotted, by default 1
      figsize : tuple, optional
          size of the figure, by default (15, 10)


   .. py:method:: batch_info(set_name='train')

      Print info about samples/batch-size/...

      Parameters
      ----------
      set_name : str, optional
          name of the dataset, by default "train"


   .. py:method:: inference_sample_results(num_samples: int = 6, class_index: int = 0, set_name: str = 'train', min_conf: Union[None, float] = None, max_conf: Union[None, float] = None, figsize: tuple = (15, 15))

      Plot some samples (specified by num_samples) for inference outputs

      Parameters
      ----------
      num_samples : int, optional
      class_index : int, optional
          class index to be plotted, by default 0
      set_name : str, optional
          name of the dataset, by default "train"
      min_conf : Union[None, float], optional
          min prediction confidence, by default None
      max_conf : Union[None, float], optional
          max prediction confidence, by default None
      figsize : tuple, optional


   .. py:method:: save(save_path='default.obj', force=False)

      Save object


   .. py:method:: load(load_path, remove_after_load=False, force_device=False)

      load class


   .. py:method:: get_time()


   .. py:method:: cprint(type_info, bc_color, text)

      simple print function used for colored logging


   .. py:method:: update_progress(progress, text='', barLength=30)



.. py:class:: classifierContext(device='default')

   Bases: :py:obj:`mapreader.train.classifier.classifier`

   .. py:method:: train(phases: list = ['train', 'val'], num_epochs: int = 25, save_model_dir: Union[None, str] = 'models', verbosity_level: int = 1, tensorboard_path: Union[None, str] = None, tmp_file_save_freq: int = 2, remove_after_load: bool = True, print_info_batch_freq: int = 5)

      Wrapper function for train_core method to capture exceptions. Supported exceptions so far:
      - KeyboardInterrupt

      Refer to train_core for more information.


   .. py:method:: train_core(phases: list = ['train', 'val'], num_epochs: int = 25, save_model_dir: Union[None, str] = 'models', verbosity_level: int = 1, tensorboard_path: Union[None, str] = None, tmp_file_save_freq: int = 2, print_info_batch_freq: int = 5)

      Train/fine-tune a classifier

      Parameters
      ----------
      phases : list, optional
          at each epoch, perform this list of phases, e.g., train and val, by default ["train", "val"]
      num_epochs : int, optional
          number of epochs, by default 25
      save_model_dir : Union[None, str], optional
          Parent directory to save models, by default "models"
      verbosity_level : int, optional
          verbosity level: -1 (quiet), 0 (normal), 1 (verbose), 2 (very verbose), 3 (debug)
      tensorboard_path : Union[None, str], optional
          Parent directory to save tensorboard files, by default None
      tmp_file_save_freq : int, optional
          frequency (in epoch) to save a temporary checkpoint


   .. py:method:: show_sample(set_name='train', batch_number=1, print_batch_info=True, figsize=(15, 10))

      Show samples from specified dataset

      Parameters
      ----------
      set_name : str, optional
          name of the dataset, by default "train"
      batch_number : int, optional
          batch number to be plotted, by default 1
      figsize : tuple, optional
          size of the figure, by default (15, 10)


   .. py:method:: layerwise_lr(min_lr: float, max_lr: float, ltype: str = 'linspace', sep_group_names=['features1', 'features2'])

      Define layer-wise learning rates

      linspace: use evenly spaced learning rates over a specified interval
      geomspace: use learning rates spaced evenly on a log scale (a geometric progression)

      Parameters
      ----------
      min_lr : float
          minimum learning rate
      max_lr : float
          maximum learning rate
      ltype : str, optional
          how to space the specified interval, by default "linspace"


   .. py:method:: inference_sample_results(num_samples: int = 6, class_index: int = 0, set_name: str = 'train', min_conf: Union[None, float] = None, max_conf: Union[None, float] = None, figsize: tuple = (15, 15))

      Plot some samples (specified by num_samples) for inference outputs

      Parameters
      ----------
      num_samples : int, optional
      class_index : int, optional
          class index to be plotted, by default 0
      set_name : str, optional
          name of the dataset, by default "train"
      min_conf : Union[None, float], optional
          min prediction confidence, by default None
      max_conf : Union[None, float], optional
          max prediction confidence, by default None
      figsize : tuple, optional




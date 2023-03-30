:py:mod:`mapreader.annotate.load_annotate`
==========================================

.. py:module:: mapreader.annotate.load_annotate


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   mapreader.annotate.load_annotate.loadAnnotations




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




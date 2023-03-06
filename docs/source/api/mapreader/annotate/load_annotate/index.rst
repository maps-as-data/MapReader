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




mapreader.annotate.annotator
============================

.. py:module:: mapreader.annotate.annotator


Classes
-------

.. autoapisummary::

   mapreader.annotate.annotator.Annotator


Module Contents
---------------

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

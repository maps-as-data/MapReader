mapreader.annotate.utils
========================

.. py:module:: mapreader.annotate.utils


Functions
---------

.. autoapisummary::

   mapreader.annotate.utils.prepare_data
   mapreader.annotate.utils.annotation_interface
   mapreader.annotate.utils.prepare_annotation
   mapreader.annotate.utils.save_annotation


Module Contents
---------------

.. py:function:: prepare_data(df, col_names = None, annotation_set = '001', label_col_name = 'label', redo = False, random_state = 'random', num_samples = 100)

   Prepare data for image annotation by selecting a subset of images from a
   DataFrame.

   :param df: DataFrame containing the image data to be annotated.
   :type df: pandas.DataFrame
   :param col_names: List of column names to include in the output. Default columns are
                     ``["image_path", "parent_id"]``.
   :type col_names: list of str, optional
   :param annotation_set: String specifying the annotation set. Default is ``"001"``.
   :type annotation_set: str, optional
   :param label_col_name: Column name containing the label information for each image. Default
                          is ``"label"``.
   :type label_col_name: str, optional
   :param redo: If ``True``, all images will be annotated even if they already have a
                label. If ``False`` (default), only images without a label will be
                annotated.
   :type redo: bool, optional
   :param random_state: Seed for the random number generator used when selecting images to
                        annotate. If set to ``"random"`` (default), a random seed will be used.
   :type random_state: int or str, optional
   :param num_samples: Maximum number of images to annotate. Default is ``100``.
   :type num_samples: int, optional

   :returns: A list of lists containing the selected image data, with each sublist
             containing the specified columns plus the annotation set and a row
             counter.
   :rtype: list of list of str/int


.. py:function:: annotation_interface(data, list_labels, list_colors = None, annotation_set = '001', method = 'ipyannotate', list_shortcuts = None)

   Create an annotation interface for a list of patches with corresponding
   labels.

   :param data: List of patches to annotate.
   :type data: list
   :param list_labels: List of strings representing the labels for each annotation class.
   :type list_labels: list
   :param list_colors: List of strings representing the colors for each annotation class,
                       by default ``["red", "green", "blue", "green"]``.
   :type list_colors: list, optional
   :param annotation_set: String representing the annotation set, specified in the yaml file or
                          via function argument, by default ``"001"``.
   :type annotation_set: str, optional
   :param method: String representing the method for annotation, by default
                  ``"ipyannotate"``.
   :type method: Literal["ipyannotate", "pigeonxt"], optional
   :param list_shortcuts: List of strings representing the keyboard shortcuts for each
                          annotation class, by default ``None``.
   :type list_shortcuts: list, optional

   :returns: **annotation** -- The annotation object containing the toolbar, tasks and canvas for the
             interface.
   :rtype: Annotation

   :raises SystemExit: If ``method`` parameter is not ``"ipyannotate"`` or ``pigeonxt``.

   .. rubric:: Notes

   This function creates an annotation interface using the ``ipyannotate``
   library, which is a browser-based tool for annotating data.


.. py:function:: prepare_annotation(userID, task, annotation_tasks_file, custom_labels = None, annotation_set = '001', redo_annotation = False, patch_paths = False, parent_paths = False, tree_level = 'patch', sortby = None, min_alpha_channel = None, min_mean_pixel = None, max_mean_pixel = None, min_std_pixel = None, max_std_pixel = None, context_image = False, xoffset = 500, yoffset = 500, urlmain = 'https://maps.nls.uk/view/', random_state = 'random', list_shortcuts = None, method = 'ipyannotate')

   Prepare image data for annotation and launch the annotation interface.

   :param userID: The ID of the user annotating the images. Should be unique as it is
                  used in the name of the output file.
   :type userID: str
   :param task: The task name that the images are associated with. This task should be
                defined in the yaml file (``annotation_tasks_file``), if not,
                ``custom_labels`` will be used instead.
   :type task: str
   :param annotation_tasks_file: The file path to the YAML file containing information about task, image
                                 paths and annotation metadata.
   :type annotation_tasks_file: str
   :param custom_labels: A list of custom label names to be used instead of the label names in
                         the ``annotation_tasks_file``. Default is ``[]``.
   :type custom_labels: list of str, optional
   :param annotation_set: The ID of the annotation set to use in the YAML file
                          (``annotation_tasks_file``). Default is ``"001"``.
   :type annotation_set: str, optional
   :param redo_annotation: If ``True``, allows the user to redo annotations on previously
                           annotated images. Default is ``False``.
   :type redo_annotation: bool, optional
   :param patch_paths: The path to the directory containing patches, if ``custom_labels`` are provided. Default is ``False`` and the information is read from the yaml file.
   :type patch_paths: str or bool, optional
   :param parent_paths: The path to parent images, if ``custom_labels`` are provided. Default
                        is ``False`` and the information is read from the yaml file.
   :type parent_paths: str, optional
   :param tree_level: The level of annotation to be used, either ``"patch"`` or ``"parent"``.
                      Default is ``"patch"``.
   :type tree_level: str, optional
   :param sortby: If ``"mean"``, sort images by mean pixel intensity. Default is
                  ``None``.
   :type sortby: str, optional
   :param min_alpha_channel: The minimum alpha channel value for images to be included in the
                             annotation interface. Only applies to patch level annotations.
                             Default is ``None``.
   :type min_alpha_channel: float, optional
   :param min_mean_pixel: The minimum mean pixel intensity value for images to be included in
                          the annotation interface. Only applies to patch level annotations.
                          Default is ``None``.
   :type min_mean_pixel: float, optional
   :param max_mean_pixel: The maximum mean pixel intensity value for images to be included in
                          the annotation interface. Only applies to patch level annotations.
                          Default is ``None``.
   :type max_mean_pixel: float, optional
   :param min_std_pixel: The minimum standard deviation of pixel intensity value for images to be included in
                         the annotation interface. Only applies to patch level annotations.
                         Default is ``None``.
   :type min_std_pixel: float, optional
   :param max_std_pixel: The maximum standard deviation of pixel intensity value for images to be included in
                         the annotation interface. Only applies to patch level annotations.
                         Default is ``None``.
   :type max_std_pixel: float, optional
   :param context_image: If ``True``, includes a context image with each patch image in the
                         annotation interface. Only applies to patch level annotations. Default
                         is ``False``.
   :type context_image: bool, optional
   :param xoffset: The x-offset in pixels to be used for displaying context images in the
                   annotation interface. Default is ``500``.
   :type xoffset: int, optional
   :param yoffset: The y-offset in pixels to be used for displaying context images in the
                   annotation interface. Default is ``500``.
   :type yoffset: int, optional
   :param urlmain: The main URL to be used for displaying images in the annotation
                   interface. Default is ``"https://maps.nls.uk/view/"``.
   :type urlmain: str, optional
   :param random_state: Seed or state value for the random number generator used for shuffling
                        the image order. Default is ``"random"``.
   :type random_state: int or str, optional
   :param list_shortcuts: A list of tuples containing shortcut key assignments for label names.
                          Default is ``None``.
   :type list_shortcuts: list of tuples, optional
   :param method: String representing the method for annotation, by default
                  ``"ipyannotate"``.
   :type method: Literal["ipyannotate", "pigeonxt"], optional

   :returns: **annotation** -- A dictionary containing the annotation results.
   :rtype: dict

   :raises ValueError: If a specified annotation_set is not a key in the paths dictionary
       of the YAML file with the information about the annotation metadata
       (``annotation_tasks_file``).


.. py:function:: save_annotation(annotation, userID, task, annotation_tasks_file, annotation_set)

   Save annotations for a given task and user to a csv file.

   :param annotation: Annotation object containing the annotations to be saved (output from
                      the annotation tool).
   :type annotation: ipyannotate.annotation.Annotation
   :param userID: User ID of the person performing the annotation. This should be unique
                  as it is used in the name of the output file.
   :type userID: str
   :param task: Name of the task being annotated.
   :type task: str
   :param annotation_tasks_file: Path to the yaml file describing the annotation tasks, paths, etc.
   :type annotation_tasks_file: str
   :param annotation_set: Name of the annotation set to which the annotations belong, defined in
                          the ``annotation_tasks_file``.
   :type annotation_set: str

   :rtype: None

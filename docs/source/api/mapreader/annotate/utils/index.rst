:py:mod:`mapreader.annotate.utils`
==================================

.. py:module:: mapreader.annotate.utils


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   mapreader.annotate.utils.display_record
   mapreader.annotate.utils.prepare_data
   mapreader.annotate.utils.annotation_interface
   mapreader.annotate.utils.prepare_annotation
   mapreader.annotate.utils.save_annotation



.. py:function:: display_record(record)

   Displays an image and optionally, a context image with a patch border.

   Parameters
   ----------
   record : tuple
       A tuple containing the following elements:
           - str : The name of the patch.
           - str : The path to the image to be displayed.
           - str : The path to the parent image, if any.
           - int : The index of the task, if any.
           - int : The number of times this patch has been displayed.

   Returns
   -------
   None

   Notes
   -----
   This function should be called from ``prepare_annotation``, there are
   several global variables that are being set in the function.

   This function uses ``matplotlib`` to display images. If the context image
   is displayed, the border of the patch is highlighted in red.

   Refer to ``ipyannotate`` and ``matplotlib`` for more info.


.. py:function:: prepare_data(df, col_names = ['image_path', 'parent_id'], annotation_set = '001', label_col_name = 'label', redo = False, random_state = 'random', num_samples = 100)

   Prepare data for image annotation by selecting a subset of images from a
   DataFrame.

   Parameters
   ----------
   df : pandas.DataFrame
       DataFrame containing the image data to be annotated.
   col_names : list of str, optional
       List of column names to include in the output. Default columns are
       ``["image_path", "parent_id"]``.
   annotation_set : str, optional
       String specifying the annotation set. Default is ``"001"``.
   label_col_name : str, optional
       Column name containing the label information for each image. Default
       is ``"label"``.
   redo : bool, optional
       If ``True``, all images will be annotated even if they already have a
       label. If ``False`` (default), only images without a label will be
       annotated.
   random_state : int or str, optional
       Seed for the random number generator used when selecting images to
       annotate. If set to ``"random"`` (default), a random seed will be used.
   num_samples : int, optional
       Maximum number of images to annotate. Default is ``100``.

   Returns
   -------
   list of list of str/int
       A list of lists containing the selected image data, with each sublist
       containing the specified columns plus the annotation set and a row
       counter.


.. py:function:: annotation_interface(data, list_labels, list_colors = ['red', 'green', 'blue', 'green'], annotation_set = '001', method = 'ipyannotate', list_shortcuts = None)

   Create an annotation interface for a list of patches with corresponding
   labels.

   Parameters
   ----------
   data : list
       List of patches to annotate.
   list_labels : list
       List of strings representing the labels for each annotation class.
   list_colors : list, optional
       List of strings representing the colors for each annotation class,
       by default ``["red", "green", "blue", "green"]``.
   annotation_set : str, optional
       String representing the annotation set, specified in the yaml file or
       via function argument, by default ``"001"``.
   method : str, optional
       String representing the method for annotation, by default
       ``"ipyannotate"``.
   list_shortcuts : list, optional
       List of strings representing the keyboard shortcuts for each
       annotation class, by default ``None``.

   Returns
   -------
   annotation : Annotation
       The annotation object containing the toolbar, tasks and canvas for the
       interface.

   Raises
   ------
   SystemExit
       If ``method`` parameter is not ``"ipyannotate"``.

   Notes
   -----
   This function creates an annotation interface using the ``ipyannotate``
   library, which is a browser-based tool for annotating data.


.. py:function:: prepare_annotation(userID, task, annotation_tasks_file, custom_labels = [], annotation_set = '001', redo_annotation = False, patch_paths = False, parent_paths = False, tree_level = 'patch', sortby = None, min_alpha_channel = None, min_mean_pixel = None, max_mean_pixel = None, min_std_pixel = None, max_std_pixel = None, context_image = False, xoffset = 500, yoffset = 500, urlmain = 'https://maps.nls.uk/view/', random_state = 'random', list_shortcuts = None)

   Prepare image data for annotation and launch the annotation interface.

   Parameters
   ----------
   userID : str
       The ID of the user annotating the images. Should be unique as it is
       used in the name of the output file.
   task : str
       The task name that the images are associated with. This task should be
       defined in the yaml file (``annotation_tasks_file``), if not,
       ``custom_labels`` will be used instead.
   annotation_tasks_file : str
       The file path to the YAML file containing information about task, image
       paths and annotation metadata.
   custom_labels : list of str, optional
       A list of custom label names to be used instead of the label names in
       the ``annotation_tasks_file``. Default is ``[]``.
   annotation_set : str, optional
       The ID of the annotation set to use in the YAML file
       (``annotation_tasks_file``). Default is ``"001"``.
   redo_annotation : bool, optional
       If ``True``, allows the user to redo annotations on previously
       annotated images. Default is ``False``.
   patch_paths : str or bool, optional
       The path to the directory containing patches, if ``custom_labels`` are provided. Default is ``False`` and the information is read from the yaml file.
   parent_paths : str, optional
       The path to parent images, if ``custom_labels`` are provided. Default
       is ``False`` and the information is read from the yaml file.
   tree_level : str, optional
       The level of annotation to be used, either ``"patch"`` or ``"parent"``.
       Default is ``"patch"``.
   sortby : str, optional
       If ``"mean"``, sort images by mean pixel intensity. Default is
       ``None``.
   min_alpha_channel : float, optional
       The minimum alpha channel value for images to be included in the
       annotation interface. Only applies to patch level annotations.
       Default is ``None``.
   min_mean_pixel : float, optional
       The minimum mean pixel intensity value for images to be included in
       the annotation interface. Only applies to patch level annotations.
       Default is ``None``.
   max_mean_pixel : float, optional
       The maximum mean pixel intensity value for images to be included in
       the annotation interface. Only applies to patch level annotations.
       Default is ``None``.
   min_std_pixel : float, optional
       The minimum standard deviation of pixel intensity value for images to be included in
       the annotation interface. Only applies to patch level annotations.
       Default is ``None``.
   max_std_pixel : float, optional
       The maximum standard deviation of pixel intensity value for images to be included in
       the annotation interface. Only applies to patch level annotations.
       Default is ``None``.
   context_image : bool, optional
       If ``True``, includes a context image with each patch image in the
       annotation interface. Only applies to patch level annotations. Default
       is ``False``.
   xoffset : int, optional
       The x-offset in pixels to be used for displaying context images in the
       annotation interface. Default is ``500``.
   yoffset : int, optional
       The y-offset in pixels to be used for displaying context images in the
       annotation interface. Default is ``500``.
   urlmain : str, optional
       The main URL to be used for displaying images in the annotation
       interface. Default is ``"https://maps.nls.uk/view/"``.
   random_state : int or str, optional
       Seed or state value for the random number generator used for shuffling
       the image order. Default is ``"random"``.
   list_shortcuts : list of tuples, optional
       A list of tuples containing shortcut key assignments for label names.
       Default is ``None``.

   Returns
   -------
   annotation : dict
       A dictionary containing the annotation results.

   Raises
   -------
   ValueError
       If a specified annotation_set is not a key in the paths dictionary
       of the YAML file with the information about the annotation metadata
       (``annotation_tasks_file``).


.. py:function:: save_annotation(annotation, userID, task, annotation_tasks_file, annotation_set)

   Save annotations for a given task and user to a csv file.

   Parameters
   ----------
   annotation : ipyannotate.annotation.Annotation
       Annotation object containing the annotations to be saved (output from
       the annotation tool).
   userID : str
       User ID of the person performing the annotation. This should be unique
       as it is used in the name of the output file.
   task : str
       Name of the task being annotated.
   annotation_tasks_file : str
       Path to the yaml file describing the annotation tasks, paths, etc.
   annotation_set : str
       Name of the annotation set to which the annotations belong, defined in
       the ``annotation_tasks_file``.

   Returns
   -------
   None



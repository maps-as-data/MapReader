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

   Display patches for annotation

   NOTE: This function should be called from prepare_annotation,
         there are several global variables that are being set in the function.

   Refer to ipyannotate for more info.


.. py:function:: prepare_data(df, col_names=['image_path', 'parent_id'], annotation_set='001', label_col_name='label', redo=False, random_state='random', num_samples=100)

   prepare data for annotations

   Args:
       df (pandas dataframe): dataframe which contains information about patches to be annotated
       col_names (list, optional): column names of the dataframe to be used in annotations. Defaults to ["image_path", "parent_id"].
       annotation_set (str, optional): as the suggest. Defaults to "001".
       label_col_name (str, optional): column name related to labels. Defaults to "label".
       redo (bool, optional): redo the annotations. Defaults to False.


.. py:function:: annotation_interface(data, list_labels, list_colors=['red', 'green', 'blue', 'green'], annotation_set='001', method='ipyannotate', list_shortcuts=None)

   Setup the annotation interface

   Args:
       data (list): list of patches to be annotated
       list_labels (list): list of labels
       list_colors (list, optional): list of colors. Defaults to ["green", "blue", "red"].
       annotation_set (str, optional): annotation set, specified in the yaml file or via function argument. Defaults to "001".
       method (str, optional): method to annotate patches. Defaults to "ipyannotate".


.. py:function:: prepare_annotation(userID, task, annotation_tasks_file, custom_labels=[], annotation_set='001', redo_annotation=False, patch_paths=False, parent_paths=False, tree_level='child', sortby=None, min_alpha_channel=None, min_mean_pixel=None, max_mean_pixel=None, min_std_pixel=None, max_std_pixel=None, context_image=False, xoffset=500, yoffset=500, urlmain='https://maps.nls.uk/view/', random_state='random', list_shortcuts=None)

   Prepare annotations

   Args:
       userID (str): unique user-ID. This is used in the name of the output file.
       task (str): name of the task. This task should be defined in the yaml file (annotation_tasks_file), if not,
                   custom_labels will be used instead
       annotation_tasks_file (str, path to yaml file): yaml file describing the tasks/paths/etc
       custom_labels (list, optional): If task is not found in the yaml file, use custom labels. Defaults to [].
       annotation_set (str, optional): Name of the annotation set defined in annotation_tasks_file. Defaults to "001".
       redo_annotation (bool, optional): redo annotations. Defaults to False.
       patch_paths (bool, str, optional): if custom_labels, specify path to patches. Normally, this is set to False and the information is read from the yaml file. Defaults to False.
       parent_paths (bool, str, optional): if custom_labels, specify path to parent images. Normally, this is set to False and the information is read from the yaml file. Defaults to False.
       tree_level (str, optional): parent/child tree level. Defaults to "child".
       sortby (None, mean, optional): sort patches to be annotated. Defaults to None.
       context_image (bool): add a context image or not
       xoffset (int, optional): x-offset for the borders of the context image. Defaults to 500.
       yoffset (int, optional): y-offset for the borders of the context image. Defaults to 500.
       urlmain (str, None, optional): when annotating, the URL in form of url_main/{map_id} will be shown as well.


.. py:function:: save_annotation(annotation, userID, task, annotation_tasks_file, annotation_set)

   Save annotation results

   Args:
       annotation: output from the annotation tool
       userID (str): unique user-ID. This is used in the name of the output file.
       task (str): name of the task. This task should be defined in the yaml file (annotation_tasks_file), if not,
                   custom_labels will be used instead
       annotation_tasks_file (str, path to yaml file): yaml file describing the tasks/paths/etc
       annotation_set (str, optional): Name of the annotation set defined in annotation_tasks_file. Defaults to "001".



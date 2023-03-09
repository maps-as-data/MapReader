Annotate
========

.. note:: You will need to update file paths to reflect your own machine's directory structure.

MapReader's ``annotate`` subpackage is used to interactively annotate images (e.g. maps).

This is done in three simple steps: 

1. :ref:`Edit the annotation tasks file.`
2. :ref:`Annotate your images.`
3. :ref:`Save your annotations.`
 

Edit the annotation tasks file
------------------------------------

.. TODO: let people know they need to create this file from scratch (would be nice to have a template somewhere as the details below get separated out and it's nice to see an example...)
	
The ``annotation_tasks.yaml`` file is used to set up your annotation tasks. It contains two sections - ``tasks`` and ``paths``.
	
The ``tasks`` section is used to specify annotation tasks and their labels. This section can contain as many tasks/labels as you would like and should be formatted as follows:
	
.. code :: yaml
		
	tasks:
		your_task_name: 
			labels: ["your_label_1", "your_label_2", "your_label_3"]
		your_task_name_2: 
			labels: ["your_label_1", "your_label_2"]

.. note:: When annotating, for each patch you will only be able to select one label from your label list. So, if you envisage wanting to label something as "x" **and also** "y", you will need to create a separate label combining "x and y".
	
The ``paths`` section is used to specify file paths to sets of images you would like to annotate (annotation sets). This section can contain as many annotation sets as you would like and should be formatted as follows:

.. code :: yaml

	paths:
		your_annotation_set:
			patch_paths: "./path/to/patches/"
			parent_paths: "./path/to/parents/"
			annot_dir: "./path/to/save/annotations"
		your_annotation_set_2:
			patch_paths: "./path/to/patches_2/"
			parent_paths: "./path/to/parents_2/"
			annot_dir: "./path/to/save/annotations_2"

For example, if you want to annotate 'railspace' (as in our `our paper <https://dl.acm.org/doi/10.1145/3557919.3565812>`_), use: 
	   
.. code :: yaml

	tasks:
		rail_space:
			labels: ["no", "rail_space"]

	paths:
		test_one_inch_maps_001:
			patch_paths: "./maps/slice_50_50/patch-*PNG"
			parent_paths: "./maps/*png"
			annot_dir: "./annotations_one_inch"
		
Annotate your images
----------------------

.. note:: Run these commands in a Jupyter notebook (or other IDE), ensuring you are in your `mr_py38` python environment.

To prepare your annotations, you must specify a ``userID``, ``annotation_tasks_file`` (i.e. the ``./annotation_task.yaml``), tell MapReader which ``task`` you'd like to run and which  ``annotation_set`` you would like to run on. 

To do this, use: 

.. code :: python

	from mapreader.annotate.utils import prepare_annotation
			
	userID="your_name"
	annotation_tasks_file="./annotation_tasks.yaml"
	task="rail_space"
	annotation_set="test_one_inch_maps_001"

	annotation=prepare_annotation(userID, annotation_tasks_file, task, annotation_set)

You can then interactively annotate a sample of your images using:

.. code :: python

	annotation

.. image:: ../figures/annotate.png
	:width: 400px

To help with annotating, you can set the annotation interface to show a context image using ``context_image=True``. This creates a second panel in the annotation interface, showing your patch in the context of a larger region whose size, in pixels, is set by ``xoffset`` and ``yoffset``.
		
.. code :: python
		
	annotation=prepare_annotation(userID, annotation_tasks_file, task, annotation_set=annotation_set,
					 				context_image=True, xoffset=100, yoffset=100)

	annotation 

.. image:: ../figures/annotate_context.png
	:width: 400px

By default, your patches will be shown to you in a random order but, to help with annotating, can be sorted by their mean pixel intesities using ``sorby="mean"``. 
	
You can also specify ``min_mean_pixel`` and ``max_mean_pixel`` to limit the range of mean pixel intensities shown to you and ``min_std_pixel`` and ``max_std_pixel`` to limit the range of standard deviations within the mean pixel intensities shown to you. This is particularly useful if your images (e.g. maps) have collars or margins that you would like to avoid.

Save your annotations
---------------------------
	
Once you have annotated your images, you should save your annotations using:

.. code :: python

	from mapreader.annotate.utils import save_annotation

	save_annotation(annotation, userID, task, annotation_tasks_file, annotation_set)

This saves your annotations as a ``.csv`` file in the ``annot_dir`` specified in your annotation tasks file.

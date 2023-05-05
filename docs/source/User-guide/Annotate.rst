Annotate
=========

.. note:: You will need to update file paths to reflect your own machine's directory structure.

MapReader's ``Annotate`` subpackage is used to interactively annotate images (e.g. maps).

.. _Annotate_images:

Annotate your images
----------------------

.. note:: Run these commands in a Jupyter notebook (or other IDE), ensuring you are in your `mr_py38` python environment.


To prepare your annotations, you must specify a number of parameters when initializing the Annotator class. The example below demonstrates how to do this for a 'rail_space' annotation task:

.. code-block:: python
    # Example
    from mapreader.annotate import Annotator

    annotator = Annotator(
        task_name="railspace",
        labels=["no_rail_space", "rail_space"],
        username="rosie",
        patches="./patches/patch-*.png",
        parents="./maps/*.png",
        annotations_dir="./annotations"
    )

In the above example, the following parameters are specified:

#. ``task_name``: The specific annotation task you want to perform, in this case ``"railspace"``.
#. ``labels``: A list of labels for the annotation task, such as ``"no_rail_space"`` and ``"rail_space"``.
#. ``username``: Your unique identifier, which can be any string (e.g., ``"rosie"``).
#. ``patches``: The file path pattern to access patch images for annotation (e.g., ``"./patches/patch-*.png"``).
#. ``parents``: The file path pattern to access the corresponding parent images (e.g., ``"./maps/*.png"``).
#. ``annotations_dir``: The directory where your annotations will be saved (e.g., ``"./annotations"``).

These are only a few of the settings that you can provide the annotator. We will cover a few more below, but a full inventory of settings can be found in the API documentation.

After setting up the ``Annotator`` instance, you can interactively annotate a sample of your images using:

.. code-block:: python

    annotator.annotate()

To help with annotating, you can set the annotation interface to show a context image using ``show_context=True``. This creates a panel of patches in the annotation interface, highlighting your patch in the middle of its surrounding immediate images. This is how you would pass the ``show_context`` argument:

.. code-block:: python

	#EXAMPLE
    annotator = Annotator(
        task_name="railspace",
        labels=["no_rail_space", "rail_space"],
        username="rosie",
        patches="./patches/patch-*.png"
        parents="./maps/*.png"
        annotations_dir="./annotations",
        show_context=True
    )

    annotator.annotate()

# TODO: This is not currently an option
By default, your patches will be shown to you in a random order but, to help with annotating, can be sorted by their mean pixel intesities using ``sorby="mean"``.

# TODO: This is not currently an option
You can also specify ``min_mean_pixel`` and ``max_mean_pixel`` to limit the range of mean pixel intensities shown to you and ``min_std_pixel`` and ``max_std_pixel`` to limit the range of standard deviations within the mean pixel intensities shown to you. 
This is particularly useful if your images (e.g. maps) have collars or margins that you would like to avoid.

e.g. :

.. code-block:: python

    annotation=prepare_annotation(userID="rosie", annotation_tasks_file="annotation_tasks.yaml", task="rail_space", annotation_set="set_001", context_image=True, xoffset=100, yoffset=100, min_mean_pixel=0.5, max_mean_pixel=0.9)

    annotation

.. _Save_annotations:

Save your annotations
----------------------

Your annotations are automatically saved as you're making progress through the annotation task as a ``csv`` file (unless you've set the ``auto_save`` keyword argument to ``False`` when you set up the ``Annotator`` instance).

If you need to know the name of the annotations file, you may refer to a property on your ``Annotator`` instance:

.. code-block:: python

    annotator.annotations_file

The file will be located in the ``annotations_dir`` that you may have passed as a keyword argument when you set up the ``Annotator`` instance. If you didn't provide a keyword argument, it will be in the ``./annotations`` directory.

For example, if you have downloaded your maps using the default settings of our ``Download`` subpackage or have set up your directory as reccommended in our `Input Guidance <https://mapreader.readthedocs.io/en/latest/Input-guidance.html>`__, and then saved your patches using the default settings:

::

    project
    ├──your_notebook.ipynb
    └──maps
    │   ├── map1.png
    │   ├── map2.png
    │   ├── map3.png
    │   ├── ...
    │   └── metadata.csv
    └──patches
    │   ├── patch-0-100-#map1.png#.png
    │   ├── patch-100-200-#map1.png#.png
    │   ├── patch-200-300-#map1.png#.png
    │   └── ...
    └──annotations
	    └──rail_space_#rosie#-123hjkfr298jIUHfs808da.csv

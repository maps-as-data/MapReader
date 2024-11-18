Annotate
=========

.. note:: You will need to update file paths to reflect your own machine's directory structure.

MapReader's ``Annotate`` subpackage is used to interactively annotate images (e.g. maps).

.. _Annotate_images:

Annotate your images
----------------------

.. note:: Run these commands in a Jupyter notebook (or other IDE), ensuring you are in your `mr_py38` Python environment.


To prepare your annotations, you must specify a number of parameters when initializing the Annotator class.
We will use a 'railspace' annotation task to demonstrate how to set up the annotator.

The simplest way to initialize your annotator is to provide file paths for your patches and parent images using the ``patch_paths`` and ``parent_paths`` arguments, respectively.
e.g. :

.. code-block:: python

    from mapreader import Annotator

    # EXAMPLE
    annotator = Annotator(
        patch_paths="./patches_100_pixel/*.png",
        parent_paths="./maps/*.png",
        metadata="./maps/metadata.csv", # or .geojson
        annotations_dir="./annotations",
        task_name="railspace",
        labels=["no_railspace", "railspace"],
        username="rosie",
    )

Alternatively, if you have created/saved a ``patch_df`` and ``parent_df`` from MapReader's ``Load`` subpackage, you can replace the ``patch_paths`` and ``parent_paths`` arguments with ``patch_df`` and ``parent_df`` arguments, respectively.
e.g. :

.. code-block:: python

    from mapreader import Annotator

    # EXAMPLE
    annotator = Annotator(
        patch_df="./patch_df.csv", # or .geojson
        parent_df="./parent_df.csv", # or .geojson
        annotations_dir="./annotations",
        task_name="railspace",
        labels=["no_railspace", "railspace"],
        username="rosie",
    )

or, if you have the patch and parent DataFrames in memory:

.. code-block:: python

    from mapreader import Annotator

    # EXAMPLE
    annotator = Annotator(
        patch_df=patch_df,
        parent_df=parent_df,
        annotations_dir="./annotations",
        task_name="railspace",
        labels=["no_railspace", "railspace"],
        username="rosie",
    )

In the above examples, the following parameters are also specified:

- ``annotations_dir``: The directory where your annotations will be saved (e.g., ``"./annotations"``).
- ``task_name``: The specific annotation task you want to perform, in this case ``"railspace"``.
- ``labels``: A list of labels for the annotation task, such as ``"no_railspace"`` and ``"railspace"``.
- ``username``: Your unique identifier, which can be any string (e.g., ``"rosie"``).

Other arguments that you may want to be aware of when initializing the ``Annotator`` instance include:

- ``show_context``: Whether to show a context image in the annotation interface (default: ``False``).
- ``border``: Whether to show a border around the central patch when showing context (default: ``False``).
- ``surrounding``: How many surrounding patches to show in the context image (default: ``1``).
- ``sortby``: The name of the column to use to sort the patch Dataframe (e.g. "mean_pixel_R" to sort by red pixel intensities).
- ``ascending``: A boolean indicating whether to sort in ascending or descending order (default: ``True``).
- ``filter_for``: A dictionary containing the name of the column to use for filtering and the value to filter for within this column. (e.g. ``{"predicted_label":"railspace"}``)
- ``delimiter``: The delimiter to use when reading your data files (default: ``","`` for csv).

After setting up the ``Annotator`` instance, you can interactively annotate a sample of your images using:

.. code-block:: python

    annotator.annotate()

Patch size
~~~~~~~~~~

By default, your patches will be shown to you as their original size in pixels.
This can make annotating difficult if your patches are very small.
To resize your patches when viewing them in the annotation interface, you can pass the ``resize_to`` argument when initializing the ``Annotator`` or when calling the ``annotate`` method.

e.g. to resize your patches so that their largest edge is 300 pixels:

.. code-block:: python

    # EXAMPLE
    annotator = Annotator(
        patch_df="./patch_df.csv",
        parent_df="./parent_df.csv",
        annotations_dir="./annotations",
        task_name="railspace",
        labels=["no_railspace", "railspace"],
        username="rosie",
        resize_to=300,
    )

Or, equivalently, :

.. code-block:: python

    annotator.annotate(resize_to=300)

.. note:: Passing the ``resize_to`` argument when calling the ``annotate`` method overrides the ``resize_to`` argument passed when initializing the ``Annotator``.

Context
~~~~~~~

As well as resizing your patches, you can also set the annotation interface to show a context image using ``show_context=True``.
This creates a panel of patches in the annotation interface, highlighting your patch in the middle of its surrounding immediate images.
As above, you can either pass the ``show_context`` argument when initializing the ``Annotator`` or when calling the ``annotate`` method.

e.g. :

.. code-block:: python

    # EXAMPLE
    annotator = Annotator(
        patch_df="./patch_df.csv",
        parent_df="./parent_df.csv",
        annotations_dir="./annotations",
        task_name="railspace",
        labels=["no_railspace", "railspace"],
        username="rosie",
        show_context=True,
    )

    annotator.annotate()

Or, equivalently, :

.. code-block:: python

    annotator.annotate(show_context=True)

.. note:: Passing the ``show_context`` argument when calling the ``annotate`` method overrides the ``show_context`` argument passed when initializing the ``Annotator`` instance.

If you have set ``show_context=True``, you can also choose to show a border around the central patch using the ``border`` argument:

.. code-block:: python

    # EXAMPLE
    annotator = Annotator(
        patch_df="./patch_df.csv",
        parent_df="./parent_df.csv",
        annotations_dir="./annotations",
        task_name="railspace",
        labels=["no_railspace", "railspace"],
        username="rosie",
        show_context=True,
        border=True,
    )

    annotator.annotate()

or, equivalently, :

.. code-block:: python

    annotator.annotate(show_context=True, border=True)

By default, your ``Annotator`` will show one surrounding patch in the context image.
You can change this by passing the ``surrounding`` argument when initializing the ``Annotator`` instance and/or when calling the ``annotate`` method.

e.g. to show two surrounding patches in the context image:

.. code-block:: python

    annotator.annotate(show_context=True, surrounding=2)

Sort order
~~~~~~~~~~

By default, your patches will be shown to you in a random order but, to help with annotating, they can be sorted using the ``sortby`` argument.
This argument takes the name of a column in your patch DataFrame and sorts the patches by the values in that column.
e.g. :

.. code-block:: python

    # EXAMPLE
    annotator = Annotator(
        patch_df="./patch_df.csv",
        parent_df="./parent_df.csv",
        annotations_dir="./annotations"m
        task_name="railspace",
        labels=["no_railspace", "railspace"],
        username="rosie",
        sortby="mean_pixel_R",
    )

This will sort your patches by the mean red pixel intensity in each patch, by default, in ascending order.
This is particularly useful if your images (e.g. maps) have collars, margins or blank regions that you would like to avoid.

.. note:: If you would like to sort in descending order, you can also pass ``ascending=False``.

You can also specify ``min_values`` and ``max_values`` to limit the range of values shown to you.
e.g. To sort your patches by the mean red pixel intensity in each patch but only show you patches with a mean blue pixel intensity between 0.5 and 0.9.

.. code-block:: python

    # EXAMPLE
    annotator = Annotator(
        patch_df="./patch_df.csv",
        parent_df="./parent_df.csv",
        annotations_dir="./annotations",
        task_name="railspace",
        labels=["no_railspace", "railspace"],
        username="rosie",
        sortby="mean_pixel_R",
        min_values={"mean_pixel_B": 0.5},
        max_values={"mean_pixel_B": 0.9},
    )

Filtering
~~~~~~~~~~

You can use the ``filter_for`` argument to filter your patches based on a column in your patch DataFrame.
This can be useful if you want to focus on a particular subset of your patches, or, to look at predictions made by a model.

e.g. to filter for patches that have been predicted to be "railspace":

.. code-block:: python

    # EXAMPLE
    annotator = Annotator(
        patch_df="./patch_df.csv",
        parent_df="./parent_df.csv",
        annotations_dir="./annotations"m
        task_name="railspace",
        labels=["no_railspace", "railspace"],
        username="rosie",
        filter_for={"predicted_label":"railspace"},
    )

This will only show you patches that have been predicted to be "railspace".

You can filter for any column in your patch DataFrame, and you can filter for multiple values by passing multiple key-value pairs as your ``filter_for`` dictionary.

Showing additional information about your patches in the annotation interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you would like to show additional information about your patches in the annotation interface, you can pass the names of the columns you would like to show using the ``show_vals`` argument when calling the ``annotate`` method.

e.g. to show the ``"mean_pixel"`` and ``"std_pixel"`` columns in the annotation interface, you should pass these column names as a list to the ``show_vals`` argument:

.. code-block:: python

    annotator.annotate(show_vals=["mean_pixel", "std_pixel"])

The values in these columns will then be shown below the patch when you are annotating.
This can help you get an idea of which mean pixel values you might want to filter for or use as "min_values" or "max_values" arguments when annotating.

.. _Save_annotations:

Save your annotations
----------------------

Your annotations are automatically saved as you're making progress through the annotation task as a ``csv`` file (unless you've set ``auto_save=False`` when you set up the ``Annotator`` instance).

If you need to know the name of the annotations file, you may refer to a property on your ``Annotator`` instance:

.. code-block:: python

    annotator.annotations_file

The file will be located in the ``annotations_dir`` that you may have passed as a keyword argument when you set up the ``Annotator`` instance.
If you didn't provide a keyword argument, it will be in the ``./annotations`` directory.

For example, if you have downloaded your maps using the default settings of our ``Download`` subpackage or have set up your directory as recommended in our :doc:`Input Guidance </using-mapreader/input-guidance/index>`, and then saved your patches using the default settings:

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
	    └──railspace_#rosie#-123hjkfr298jIUHfs808da.csv

Geospatial images: Maps and earth observation imagery
=====================================================

MapReader was developed for maps and geospatial images.

..
    TODO: Add a note here that says that you should look through step-by-step guidance before engaging with the worked examples to understand the workflow.

Classification of one-inch OS maps
----------------------------------

.. image:: /_static/tutorial_classification_one_inch_maps_001.png
   :width: 400px
   :target: https://github.com/maps-as-data/mapreader-examples/tree/main/notebooks/geospatial/classification_one_inch_maps

We have provided two examples of how to use MapReader to identify railspace patches in one-inch OS maps.
Both examples demonstrate how to use MapReader with maps hosted on a tileserver.

Our examples show a full end-to-end use of the MapReader pipeline, including downloading, loading and patchifying map images, annotating patches to create training data, training a model and using the model to classify patches.

The ``classification_one_inch_maps`` example demonstrates how to use MapReader to classify patches using a standard patch-level classification model in which patches are used as inputs to the model.
It can be found `here <https://github.com/maps-as-data/mapreader-examples/blob/main/notebooks/geospatial/classification_one_inch_maps/Pipeline.ipynb>`__.

The ``context_classification_one_inch_maps`` example demonstrates how to use MapReader to classify patches using a context-level classification model in which patches and their surrounding patches (i.e. context) are used as inputs to the model.
It can be found `here <https://github.com/maps-as-data/mapreader-examples/tree/main/notebooks/geospatial/context_classification_one_inch_maps>`__.


Workshop notebooks
------------------

In the worked examples directory, we also have a number of notebooks that were used in our workshops.
These are **not** updated to align with the most recent version of MapReader, but instead they are dated and contain information about the MapReader version used at the time of the workshop.

If you want to run one of these you will need to install the version of MapReader that was used at the time of the workshop.

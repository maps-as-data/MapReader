Worked Examples
================

We have provided a number of worked examples to demonstrate how to use MapReader.
These examples can be found in the `worked_examples <https://github.com/Living-with-machines/MapReader/tree/main/worked_examples>`_ directory of the repository.

Geospatial images: Maps and earth observation imagery
--------------------------

MapReader was developed for maps and geospatial images.

Classification of one-inch OS maps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: https://raw.githubusercontent.com/Living-with-machines/MapReader/main/figs/tutorial_classification_one_inch_maps_001.png
   :width: 400px
   :target: https://github.com/Living-with-machines/MapReader/tree/main/worked_examples/geospatial

We have provided two examples of how to use MapReader to identify railspace patches in one-inch OS maps.
Both examples demonstrate how to use MapReader with maps hosted on a tileserver.

Our examples show a full end-to-end use of the MapReader pipeline, including downloading, loading and patchifying map images, annotating patches to create training data, training a model and using the model to classify patches.

The first examples demonstrates how to use MapReader to classify patches using a standard patch-level classification model in which patches are used as inputs to the model.
It can be found `here <https://github.com/Living-with-machines/MapReader/blob/main/worked_examples/geospatial/classification_one_inch_maps/Pipeline.ipynb>`__.

The second example demonstrates how to use MapReader to classify patches using a context-level classification model in which patches and their surrounding patches (i.e. context) are used as inputs to the model.
It can be found `here <https://github.com/Living-with-machines/MapReader/blob/main/worked_examples/geospatial/context_classification_one_inch_maps/Pipeline.ipynb>`__.

Non-geospatial images
---------------------

MapReader can also be used for non-geospatial images.
We have provided two examples of this.

Classification of plant phenotypes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: https://raw.githubusercontent.com/Living-with-machines/MapReader/main/figs/tutorial_classification_plant_phenotype.png
   :width: 400px
   :target: https://github.com/Living-with-machines/MapReader/blob/main/worked_examples/non-geospatial/classification_plant_phenotype/Pipeline.ipynb

In our plant phenotypes example, we demonstrate how to use MapReader to classify plant phenotypes in images of plants.
Importantly, this worked example demonstrates how to use MapReader with non-georeferenced images (e.g. non-georeferenced map images).
It can be found `here <ttps://github.com/Living-with-machines/MapReader/blob/main/worked_examples/non-geospatial/classification_plant_phenotype/Pipeline.ipynb>`__.

Classification of MNIST digits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: https://raw.githubusercontent.com/Living-with-machines/MapReader/main/figs/tutorial_classification_mnist.png
   :width: 400px
   :target: https://github.com/Living-with-machines/MapReader/blob/main/worked_examples/non-geospatial/classification_mnist/Pipeline.ipynb

In our MNIST example, we demonstrate how to use MapReader to classify MNIST digits.
Importantly, this example demonstrates how to use MapReader to classify whole images instead of patches and therefore how MapReader can generalize to much broader use cases.
It can be found `here <https://github.com/Living-with-machines/MapReader/blob/main/worked_examples/non-geospatial/classification_mnist/Pipeline.ipynb>`__.

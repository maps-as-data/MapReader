Spot text
=========

MapReader implements three new frameworks for spotting text on maps:

- ``DPTextDETRRunner`` - This is used to detect text on maps using `DPTextDETR <https://github.com/ymy-k/DPText-DETR/tree/main>`__ and outputs bounding boxes and scores.
- ``DeepSoloRunner`` - This is used to detect and recognize text on maps using `DeepSolo <https://github.com/ViTAE-Transformer/DeepSolo/tree/main>`__ and outputs bounding boxes, text and scores.
- ``MapTextPipeline`` - This is used to detect and recognize text on maps using `MapTextPipeline <https://github.com/yyyyyxie/MapTextPipeline>`__ and outputs bounding boxes, text and scores.

We recommend using the ``MapTextPipeline`` for most use cases as it has been used to train a model on a sample of David Rumsey maps and so should work best for map text spotting.

Install dependencies
--------------------

To run text spotting with MapReader, you will need to install the required dependencies. These are listed below:

.. note:: We have our own forks of the ``DPTextDETRRunner``, ``DeepSolo`` and ``MapTextPipeline`` repos to enable them to work on CPU. Please use our forks but remember to cite the original authors in your work!

Detectron2
~~~~~~~~~~~

Detectron2 is a popular object detection library built by Facebook AI Research.
The main repo is available `here <https://github.com/facebookresearch/detectron2>`__.

To install, run the following commands in your terminal:

.. code:: bash

    git clone https://github.com/facebookresearch/detectron2.git
    cd detectron2
    pip install .


.. admonition:: Detectron2 issues for windows users
    :class: dropdown

    If you are on a windows machine and are having trouble installing Detectron2, you can try the following:
    - Install `Visual Studio Build Tools <https://visualstudio.microsoft.com/downloads/?q=build+tools>`__.
    - Follow instructions `here <https://stackoverflow.com/questions/64261546/how-to-solve-error-microsoft-visual-c-14-0-or-greater-is-required-when-inst>`__ to install the required packages. (The format might be different in newer versions of Visual Studio Build Tools, so you might need to look up the specific package names.)

    Once this is done, retry installing Detectron2.

You should then **pick one** of the following to install:

.. note:: Since the DPText-DETR, DeepSolo and MapTextPipeline frameworks are built on top of `AdelaiDet <https://github.com/aim-uofa/AdelaiDet>`__, you won't be able to install them at the same. To get around this, you can set up three different conda environments, one for each framework. This will allow you to switch between them as needed.

DPTextDETR
~~~~~~~~~~~

Our fork for DPText-DETR is available `here <https://github.com/rwood-97/DPText-DETR>`__.

To install, run the following commands in your terminal:

.. code:: bash

    git clone https://github.com/rwood-97/DPText-DETR.git
    cd DPText-DETR
    pip install .

DeepSolo
~~~~~~~~

Our fork for DeepSolo is available `here <https://github.com/rwood-97/DeepSolo>`__

To install, run the following commands in your terminal:

.. code:: bash

    git clone https://github.com/rwood-97/DeepSolo.git
    cd DeepSolo
    pip install .

MapTextPipeline
~~~~~~~~~~~~~~~

Our fork for MapTextPipeline is available `here <https://github.com/rwood-97/MapTextPipeline>`__

To install, run the following commands in your terminal:

.. code:: bash

    git clone https://github.com/rwood-97/MapTextPipeline.git
    cd MapTextPipeline
    pip install .


Advice for patch size
---------------------

When running the text spotting models, we recommend using a patch size of 1024x1024 pixels.
This is the size used as input to the models, and so should give the best results.

You may also want to create some overlap between your patches as this should minimise cut off text at the edges of patches.
MapReader has an algorithm to deduplicate overlapping bounding boxes so this creating an overlap will enable the fullest text to be detected.
You will need to experiment with the amount of overlap to find the best results for your maps.

.. note:: Greater overlaps will create more patches and result in greater computational costs when running.

See the :doc:`Load </using-mapreader/step-by-step-guide/2-load>` user guide for more information on how to create patches.

Set-up the runner
-----------------

Once you have installed the dependencies, you can set up your chosen "runner".

You will need to choose a model configuration and download the corresponding model weights.

- Config files can be found in the ``DPText-DETR``, ``DeepSolo`` and ``MapTextPipeline`` repositories under the ``configs`` directory.
- Weights files should be downloaded from the github repositories (links to the downloads are in the README).

e.g. for the ``DPTextDETRRunner``, if you choose the "ArT/R_50_poly.yaml", you should download the "art_final.pth" model weights file from the DPTextDETR repo.

e.g. for the ``DeepSoloRunner``, if you choose the "R_50/IC15/finetune_150k_tt_mlt_13_15_textocr.yaml", you should download the "ic15_res50_finetune_synth-tt-mlt-13-15-textocr.pth" model weights file from the DeepSolo repo.

e.g. for the ``MapTextPipeline``, if you choose the "ViTAEv2_S/rumsey/final_rumsey.yaml", you should download the "rumsey-finetune.pth" model weights file from the MapTextPipeline repo.

.. note:: We recommend using the "ViTAEv2_S/rumsey/final_rumsey.yaml" configuration and "rumsey-finetune.pth" weights from the ``MapTextPipeline``. But you should choose based on your own use case.

For the DPTextDETRRunner, use:

.. code-block:: python

    from map_reader import DPTextDETRRunner

    #EXAMPLE
    my_runner = DPTextDETR(
        "./patch_df.csv",
        "./parent_df.csv",
        cfg_file = "DPText-DETR/configs/DPText_DETR/ArT/R_50_poly.yaml",
        weights_file = "./art_final.pth",
    )

or, if you have your `patch_df` and `parent_df` already loaded as pandas DataFrames or geopandas GeoDataFrames, you can use:

.. code-block:: python

    #EXAMPLE
    my_runner = DPTextRunner(
        patch_df,
        parent_df,
        cfg_file = "DPText-DETR/configs/DPText_DETR/ArT/R_50_poly.yaml",
        weights_file = "./art_final.pth",
    )

For the DeepSoloRunner, use:

.. code-block:: python

    from map_reader import DeepSoloRunner

    #EXAMPLE
    my_runner = DeepSoloRunner(
        "./patch_df.csv",
        "./parent_df.csv",
        cfg_file = "DeepSolo/configs/R_50/IC15/finetune_150k_tt_mlt_13_15_textocr.yaml",
        weights_file = "./ic15_res50_finetune_synth-tt-mlt-13-15-textocr.pth"
    )

or, if you have your `patch_df` and `parent_df` already loaded as pandas DataFrames or geopandas GeoDataFrames, you can use these directly as shown for the DPTextDETRRunner (above).

For the MapTextPipeline, use:

.. code-block:: python

    from map_reader import MapTextPipeline

    #EXAMPLE
    my_runner = MapTextPipeline(
        "./patch_df.csv",
        "./parent_df.csv",
        cfg_file = "MapTextPipeline/configs/ViTAEv2_S/rumsey/final_rumsey.yaml",
        weights_file = "./rumsey-finetune.pth"
    )

or, if you have your `patch_df` and `parent_df` already loaded as pandas DataFrames or geopandas GeoDataFrames, you can use these directly as shown for the DPTextDETRRunner (above).

.. note:: You'll need to adjust the paths to the config and weights files to match your own set-up!

By default, the runners will set the device to "cuda" if available, otherwise it will use "cpu".
You can explicitly set this using the ``device`` argument:

.. code-block:: python

    #EXAMPLE
    my_runner = MapTextPipeline(
        "./patch_df.csv",
        "./parent_df.csv",
        cfg_file = "MapTextPipeline/configs/ViTAEv2_S/rumsey/final_rumsey.yaml",
        weights_file = "./rumsey-finetune.pth",
        device = "cuda",
    )


Run the runner
--------------

You can then run the runner on all patches in your patch dataframe:

.. code-block:: python

    patch_preds = my_runner.run_all()

By default, this will return a dictionary containing all the predictions for each patch.
If you'd like to return a dataframe instead, use the ``return_dataframe`` argument:

.. code-block:: python

    patch_preds_df = my_runner.run_all(return_dataframe=True)

MapReader will automatically run a deduplication algorithm to remove overlapping bounding boxes, based on a minimum intersection of area (IoA) for each overlapping polygon.
If two polygons overlap with intersection over area greater than the minimum IoA, the the one with the lower IoA will be kept (i.e. the larger of the two polygons).

Below are two examples of this:

.. image:: /_static/IoA.png
    :width: 400px

.. image:: /_static/IoA_0.9.png
    :width: 400px

By default, the minimum IoA is set to 0.7 so the deduplication algorithm will only remove the smaller polygon in the second example.

You can adjust the minimum IoA by setting the ``min_ioa`` argument:

.. code-block:: python

    patch_preds_df = my_runner.run_all(return_dataframe=True, min_ioa=0.9)

Higher ``min_ioa``values will mean a tighter threshold for identifying two polygons as duplicates.

If you'd like to run the runner on a single patch, you can also just run on one image:

.. code-block:: python

    patch_preds = my_runner.run_on_image("path/to/your/image.png", min_ioa=0.7)

Again, this will return a dictionary by default but you can use the ``return_dataframe`` argument to return a dataframe instead.

To view the patch predictions, you can use the ``show`` method.
This takes an image ID as an argument, and will show you all the predictions for that image:

.. code-block:: python

    #EXAMPLE
    my_runner.show(
        "patch-0-0-1000-1000-#map_74488689.png#.png"
    )

By default, this will show the image with the bounding boxes drawn on in red and text in blue.
You can change these by setting the ``border_color`` and ``text_color`` arguments:

.. code-block:: python

    my_runner.show(
        "patch-0-0-1000-1000-#map_74488689.png#.png",
        border_color = "green",
        text_color = "yellow",
    )

You can also change the size of the figure with the ``figsize`` argument.


Scale-up to whole map
---------------------

Once you've got your patch-level predictions, you can scale these up to the parent image using the ``convert_to_parent_pixel_bounds`` method:

.. code-block:: python

    parent_preds = my_runner.convert_to_parent_pixel_bounds()

This will return a dictionary containing the predictions for the parent image.
If you'd like to return a dataframe instead, use the ``return_dataframe`` argument:

.. code-block:: python

    parent_preds_df = my_runner.convert_to_parent_pixel_bounds(return_dataframe=True)

If you have created patches with overlap, then you should deduplicate at the parent level as well.
You can do this by setting the ``deduplicate`` argument and passing a ``min_ioa`` value:

.. code-block:: python

    parent_preds_df = my_runner.convert_to_parent_pixel_bounds(return_dataframe=True, deduplicate=True, min_ioa=0.7)

This will help resolve any issues with predictions being cut-off at the edges of patches since the overlap should help find the full piece of text.

Again, to view the predictions, you can use the ``show`` method.
You should pass a parent image ID as the ``image_id`` argument:

.. code-block:: python

    #EXAMPLE
    my_runner.show(
        "map_74488689.png"
    )

As above, use the ``border_color``, ``text_color`` and ``figsize`` arguments to customize the appearance of the image.

.. code-block:: python

    my_runner.show(
        "map_74488689.png",
        border_color = "green",
        text_color = "yellow",
        figsize = (20, 20),
    )


You can then save these predictions to a csv file:

.. code-block:: python

    parent_preds_df.to_csv("text_preds.csv")

Geo-reference
-------------

If you maps are georeferenced in your ``parent_df``, you can also convert the pixel bounds to georeferenced coordinates using the ``convert_to_coords`` method:

.. code-block:: python

    geo_preds_df = my_runner.convert_to_coords(return_dataframe=True)

Again, you can save these to a csv file as above, or, you can save them to a geojson file for loading into GIS software:

.. code-block:: python

    my_runner.save_to_geojson("text_preds.geojson")

This will save the predictions to a geojson file, with each text prediction as a separate feature.

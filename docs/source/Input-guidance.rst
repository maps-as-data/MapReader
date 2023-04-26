Input guidance
===============

.. contents:: Table of Contents
    :depth: 2

Input options
--------------

The MapReader pipeline is explained in detail `here <https://mapreader.readthedocs.io/en/latest/About.html>`__.
The inputs you will need for MapReader will depend on where you begin within the pipeline.

Option 1 - If you want to download maps from a TileServer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to download maps from a TileServer using MapReaders ``Download`` subpackage, you will need to begin with the 'Download' task. 
For this, you will need:

* A ``json`` file containing metadata for each map you would like to query/download. 
* The base URL of the map layer which you would like to access.

.. TODO: RW - Unsure if the below is true so will need to check. Leaving for now.

The key starting point is to be sure you have metadata for each map, or, "item-level" metadata in a json file. 
This allows every map file to be associated with its georeferencing information, title, publication date, or other basic information that you would like to be preserved and associated with patches.

You may have different kinds of metadata from different sources for your map files (e.g. descriptive or bibliographic metadata from a collection record, or technical metadata about georeferencing). 
We provide detailed guidance about requirements for your metadata file if you are working with maps from a Tile Server service.


.. comment: TODO add guidance about metadata requirement for other file types (not tile server) (Rosie) - need column in metadata that corresponds to image id in images object.

Option 2 - If your files are already saved locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you already have your maps saved locally, you can skip the 'Download' task and move straight to 'Load'.

If you would like to work with georeferenced maps, you will need either:

* Your map images saved as standard, non-georeferenced, image files (e.g. JPEG, PNG or TIFF) along with a separate file containing georeferencing metadata you wish to associate to these map images **OR**
* You map images saved as georeferenced image files (e.g. geoTIFF).

Alternatively, if you would like to work with non-georeferenced maps/images, you will need:

* Your images saved as standard image files (e.g. JPEG, PNG or TIFF).

Recommended directory structure
--------------------------------

If you are using non-georeferenced image files (e.g. PNG files) plus a separate metadata file, we reccomend setting these up in the following directory structure:

::

    project
    ├──your_notebook.ipynb
    └──maps        
        ├── map1.png
        ├── map2.png
        ├── map3.png
        ├── ...
        └── metadata.csv
    
This is the directory structure created by default when downloading maps using MapReader's ``Download`` subpackage.

Alternatively, if you are using geo-referenced image files (eg. geoTIFF files), your will not need a metadata file, and so your files can be set up as follows: 

::

    project
    ├──your_notebook.ipynb
    └──maps        
        ├── map1.tif
        ├── map2.tif
        ├── map3.tif
        └── ...


.. note:: Your map images should be stored in a flat directory. They **cannot be nested** (e.g. if you have states within a nation, or some other hierarchy or division).

.. comment: TODO - Katie to add comment about user needing to have maps accessible either in cloud storage (Azure, etc.) or locally.

Preparing your metadata
------------------------

MapReader uses the file names of your map images as unique identifiers (``image_id``s).
Therefore, if you would like to associate metadata to your map images, then, **at minimum**, your metadata must contain a column/header named ``image_id`` whose contents is the file names of your map images.

To load metadata (e.g. georeferencing information, publication dates or any other information about your images) into MapReader, your metadata must be in a `pandas readable file format <https://pandas.pydata.org/>`_.


Option 1 - Using a ``csv`` file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest option is to save your metadata as a ``csv`` file (this can be created from an excel spreadsheet using ``File > Save As...``) and load it directly into MapReader.

If you are loading metadata from a ``csv`` file, your file should be structures as follows:


+-----------+--------------------------+---------------------+-----------+
| image_id  | col1 (e.g. coords)       | col2 (e.g. region)  | col3      |
+===========+==========================+=====================+===========+
| map1.png  | (-4.8, -4.2, 55.8, 56.4) | Glasgow             | ...       |
+-----------+--------------------------+---------------------+-----------+
| map2.png  | (-2.2, -1.6, 53.2, 53.8) | Manchester          | ...       |
+-----------+--------------------------+---------------------+-----------+
| map3.png  | (-3.6, -3.0, 50.1, 50.8) | Dorset              | ...       |
+-----------+--------------------------+---------------------+-----------+
| ...       | ...                      | ...                 | ...       |
+-----------+--------------------------+---------------------+-----------+

Your file can contain as many columns/rows as you like, so long as it contains the ``image_id`` column.

.. note:: If the contents of your file contains commas, you should choose a different delimiter when saving your ``csv`` file. We reccomend using the pipe: ``|``.

Option 2 - Loading metadata from other file formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As `Pandas is able to read a number of different file formats <https://pandas.pydata.org/docs/user_guide/io.html>`_, you may still be able to use your metadata even if it is saved in a different file format.

To do this, you will need to use python to:

1. Read your file using one of pandas ``read_xxx`` methods and create a dataframe from it.
2. Ensure there is an ``image_ID`` column to your dataframe (and add one if there is not).
3. Pass your dataframe to MapReader.

Depending on the structure/format of your metadata, this may end up being a fairly complex task and so is not reccomended.
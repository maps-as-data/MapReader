Input guidance
===============

.. contents:: Table of Contents
    :depth: 2
    :local:

Input options
--------------

The MapReader pipeline is explained in detail `here <https://mapreader.readthedocs.io/en/latest/About.html>`__.
The inputs you will need for MapReader will depend on where you begin within the pipeline.

Option 1 - If the map(s) you want have been georeferenced and made available via a Tile Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some GLAM institutions or other services make digitized, georeferenced maps available via tile servers, for example as raster (XYZ, or 'slippy map') tiles.

Instructions for accessing tile layers from one example collection is below:

- National Library of Scotland tile layers

If you want to download maps from a TileServer using MapReader's ``Download`` subpackage, you will need to begin with the 'Download' task. 
For this, you will need:

* A ``json`` file containing metadata for each map sheet you would like to query/download. 
* The URL of the XYZ tile layer which you would like to access.

At a minimum, for each map sheet, your ``json`` file should contain information on:

- the name and URL of an individual sheet that is contained in the composite layer
- the geometry of the sheet (i.e. its coordinates), so that, where applicable, individual sheets can be isolated from the whole layer
- the coordinate reference system (CRS) used

These should be saved in a format that looks something like this:

.. code-block:: javascript

    {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {
                "geometry_name": "the_geom",
                "coordinates": [...]
            },
            "properties": {
                "IMAGE": "...",
                "WFS_TITLE": "..."
                "IMAGEURL": "..."
            },
        }],
        "crs": {
            "name": "EPSG:4326"
            },
    }

.. Check these links are still valid

Some example metadata files, corresponding to the `OS one-inch 2nd edition maps <https://mapseries-tilesets.s3.amazonaws.com/1inch_2nd_ed/index.html>`_ and `OS six-inch 1st edition maps for Scotland <https://mapseries-tilesets.s3.amazonaws.com/os/6inchfirst/index.html>`_, are provided in ``MapReader/worked_examples/persistent_data``.

Option 2 - If your files are already saved locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you already have your maps saved locally, you can skip the 'Download' task and move straight to 'Load'.

If you would like to work with georeferenced maps, you will need either:

* Your map images saved as standard, non-georeferenced, image files (e.g. JPEG, PNG or TIFF) along with a separate file containing georeferencing metadata you wish to associate to these map images **OR**
* You map images saved as georeferenced image files (e.g. geoTIFF).

Alternatively, if you would like to work with non-georeferenced maps/images, you will need:

* Your images saved as standard image files (e.g. JPEG, PNG or TIFF).

.. note:: It is possible to use non-georeferenced maps in MapReader, however none of the functionality around plotting patches based on geospatial coordinates will be possible. In this case, patches can be analyzed as regions within a map sheet, where the sheet itself may have some geospatial information associated with it (e.g. the geospatial coordinates for its center point, or the place name in its title).

Recommended directory structure
--------------------------------

If you are using non-georeferenced image files (e.g. PNG files) plus a separate metadata file, we recommend setting these up in the following directory structure:

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

.. note:: Additionally, map images should be available locally or you should set up access via cloud storage. If you are working with a very large corpus of maps, you should consider running MapReader in a Virtual Machine with adequate storage.

Preparing your metadata
------------------------

MapReader uses the file names of your map images as unique identifiers (``image_id`` s).
Therefore, if you would like to associate metadata to your map images, then, **at minimum**, your metadata must contain a column/header named ``image_id`` or ``name`` whose content is the file name of each map image.

To load metadata (e.g. georeferencing information, publication dates or any other information about your images) into MapReader, your metadata must be in a `pandas readable file format <https://pandas.pydata.org/>`_.

.. note:: Many map collections do not have item-level metadata, however even the minimal requirements here (a filename, geospatial coordinates, and CRS) will suffice for using MapReader. It is always a good idea to talk to the curators of the map collections you wish to use with MapReader to see if there are metadata files that can be shared for research purposes.


Option 1 - Using a ``csv``, ``xls`` or ``xlsx`` file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest option is to save your metadata as a ``csv``, ``xls`` or ``xlsx`` file and load it directly into MapReader.

.. note:: If you are using a ``csv`` file but the contents of you metadata contains commas, you will need to use another delimiter. We recommend using a pipe (``|``).

If you are loading metadata from a ``csv``, ``xls`` or ``xlsx`` file, your file should be structures as follows:

+-----------+-----------------------------+------------------------+--------------+
| image_id  | column1 (e.g. coords)       | column2 (e.g. region)  | column3      |
+===========+=============================+========================+==============+
| map1.png  | (-4.8, 55.8, -4.2, 56.4)    | Glasgow                | ...          |
+-----------+-----------------------------+------------------------+--------------+
| map2.png  | (-2.2, 53.2, -1.6, 53.8)    | Manchester             | ...          |
+-----------+-----------------------------+------------------------+--------------+
| map3.png  | (-3.6, 50.1, -3.0, 50.8)    | Dorset                 | ...          |
+-----------+-----------------------------+------------------------+--------------+
| ...       | ...                         | ...                    | ...          |
+-----------+-----------------------------+------------------------+--------------+

Your file can contain as many columns/rows as you like, so long as it contains at least one named ``image_id`` or ``name``.

.. Add comment about nature of coordinates as supplied by NLS vs what they might be for other collections

Option 2 - Loading metadata from other file formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As pandas is able to read `a number of different file formats <https://pandas.pydata.org/docs/user_guide/io.html>`_, you may still be able to use your metadata even if it is saved in a different file format.

To do this, you will need to use python to:

1. Read your file using one of pandas ``read_xxx`` methods and create a dataframe from it.
2. Ensure there is an ``image_ID`` column to your dataframe (and add one if there is not).
3. Pass your dataframe to MapReader.

Depending on the structure/format of your metadata, this may end up being a fairly complex task and so is not recommended unless absolutely necessary. 
A conversation with the collection curator is always a good idea to check what formats metadata may already be available in/or easily made available in using existing workflows.

Accessing Maps via TileServers
------------------------------

National Library of Scotland
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is possible to bring in any other georeferenced layers from the National Library of Scotland into MapReader. 
To do this, you would need to create a TileServer object and specify the metadata_path (the path to your metadata.json file) and the download_url (the WMTS or XYZ URL for your tileset) for your chosen tilelayer.

`This page <https://maps.nls.uk/guides/georeferencing/layers-list/>`__ lists some of the NLS's most popular georeferenced layers and provides links to their WMTS and XYZ URLs.
If, for example, you wanted to use the "Ordnance Survey - 10 mile, General, 1955 - 1:633,600" in MapReader, you would need to look up its XYZ URL (https://mapseries-tilesets.s3.amazonaws.com/ten_mile/general/{z}/{x}/{y}.png) and insert it your MapReader code as shown below:

.. code-block:: python
    
    from mapreader import TileServer

    my_ts = TileServer(
        metadata_path="path/to/metadata.json",
        download_url="https://mapseries-tilesets.s3.amazonaws.com/ten_mile/general/{z}/{x}/{y}.png",
    )

.. note:: You would need to generate the corresponding `metadata.json` before running this code.

More information about using NLS georeferenced layers is available `here <https://maps.nls.uk/guides/georeferencing/layers-urls/>`__, including details about accessing metadata for each layer. 
Please note the Re-use terms for each layer, as these vary.

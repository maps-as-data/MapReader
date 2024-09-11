Preparing your metadata
=======================

MapReader uses the file names of your map images as unique identifiers.
Therefore, if you would like to associate metadata (e.g. georeferencing information, publication dates or any other information about your images) to your map images, then your metadata must contain a column/header named ``image_id`` or ``name`` that matches the file names of your map images and columns for the metadata you'd like to add.

To load metadata from a file into MapReader, your metadata file should be in a CSV (or TSV/etc), Excel or GeoJSON file format.

e.g. If you are loading metadata from a CSV/TSV/etc or Excel file, your file could be structured as follows:

+-----------+-----------------------------+------------------------+--------------+
| image_id  | coordinates                 | region                 | column3      |
+===========+=============================+========================+==============+
| map1.png  | (-4.8, 55.8, -4.2, 56.4)    | Glasgow                | ...          |
+-----------+-----------------------------+------------------------+--------------+
| map2.png  | (-2.2, 53.2, -1.6, 53.8)    | Manchester             | ...          |
+-----------+-----------------------------+------------------------+--------------+
| map3.png  | (-3.6, 50.1, -3.0, 50.8)    | Dorset                 | ...          |
+-----------+-----------------------------+------------------------+--------------+
| ...       | ...                         | ...                    | ...          |
+-----------+-----------------------------+------------------------+--------------+

This file can contain as many columns as you like, but the ``image_id`` column is required to ensure the metadata is matched to the correct map image.

.. note:: Many map collections do not have item-level metadata, however even the minimal requirements here (a filename, geospatial coordinates, and CRS) will suffice for using MapReader. It is always a good idea to talk to the curators of the map collections you wish to use with MapReader to see if there are metadata files that can be shared for research purposes.

.. Add comment about nature of coordinates as supplied by NLS vs what they might be for other collections

Using metadata in other formats
--------------------------------

So long as your file is in a format readable by `Pandas <https://pandas.pydata.org/docs/user_guide/io.html>`_ or `GeoPandas <https://geopandas.org/en/stable/docs/user_guide/io.html>`_, you may still be able to use your metadata even if it is saved in a file format not supported by MapReader.

To do this, you will need to use Python to:

1. Read your file using one of Pandas ``read_xxx`` methods and create a dataframe from it.
2. Ensure there is an ``image_ID`` column to your dataframe (and add one if there is not).
3. Pass your dataframe to MapReader.

Depending on the structure/format of your metadata, this may end up being a fairly complex task and so is not recommended unless absolutely necessary.
A conversation with the collection curator is always a good idea to check what formats metadata may already be available in/or easily made available in using existing workflows.

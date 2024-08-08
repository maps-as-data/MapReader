Preparing your metadata
=======================

MapReader uses the file names of your map images as unique identifiers (``image_id`` s).
Therefore, if you would like to associate metadata to your map images, then, **at minimum**, your metadata must contain a column/header named ``image_id`` or ``name`` whose content is the file name of each map image.

To load metadata (e.g. georeferencing information, publication dates or any other information about your images) into MapReader, your metadata must be in a `file format readable by Pandas <https://pandas.pydata.org/>`_.

.. note:: Many map collections do not have item-level metadata, however even the minimal requirements here (a filename, geospatial coordinates, and CRS) will suffice for using MapReader. It is always a good idea to talk to the curators of the map collections you wish to use with MapReader to see if there are metadata files that can be shared for research purposes.


Option 1 - Using a ``csv``, ``xls`` or ``xlsx`` file
-----------------------------------------------------

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
---------------------------------------------------

As Pandas is able to read `a number of different file formats <https://pandas.pydata.org/docs/user_guide/io.html>`_, you may still be able to use your metadata even if it is saved in a different file format.

To do this, you will need to use Python to:

1. Read your file using one of Pandas ``read_xxx`` methods and create a dataframe from it.
2. Ensure there is an ``image_ID`` column to your dataframe (and add one if there is not).
3. Pass your dataframe to MapReader.

Depending on the structure/format of your metadata, this may end up being a fairly complex task and so is not recommended unless absolutely necessary.
A conversation with the collection curator is always a good idea to check what formats metadata may already be available in/or easily made available in using existing workflows.

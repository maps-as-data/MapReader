Recommended directory structure
===============================

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

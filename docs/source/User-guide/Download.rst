.. reminder: add note about running commands in IDE or jupyter notebook etc.

Download
=========

MapReader's ``download`` subpackage is primarily used to download files (e.g. map images and metadata) stored remotely and contains two methods of downloading these files:

- :ref:`Via TileServer_` - an open-source map server
- :ref:`Via Azure-Blob-Storage_` - Microsoft's cloud storage

Via TileServer_
----------------

To download maps from TileServer_, you will need to tell MapReader which maps to download.This is done by providing MapReader with a metadata file (usually a ``.json``), which contains information about the maps/map series you would like to download and a download URL.

Some example metadata files, corresponding to the `OS one-inch 2nd edition maps <https://mapseries-tilesets.s3.amazonaws.com/1inch_2nd_ed/index.html>`__ and `OS six-inch 1st edition maps for Scotland <https://mapseries-tilesets.s3.amazonaws.com/os/6inchfirst/index.html>`__, are provided in ``MapReader/mapreader/persistent_data``.

To set up your download, create a ``TileServer`` object and specify ``metadata_path`` (the path to your metadata file) and ``download_url`` (the base URL for the maps/map series): 

.. code :: python

     from mapreader import TileServer
     my_ts = TileServer(metadata_path="path/to/metadata.json", download_url="mapseries-tilesets.your_URL_here/{z}/{x}/{y}.png")

.. TODO: add link to info about OS 1-inch maps in statement below/edit statement to clarify what these maps are for as examples.

e.g. for the OS one-inch maps (detailed above):

.. code :: python

     my_ts = TileServer(metadata_path="~/MapReader/mapreader/persistent_data/metadata_OS_One_Inch_GB_WFS_light.json", download_url="https://mapseries-tilesets.s3.amazonaws.com/1inch_2nd_ed/{z}/{x}/{y}.png")


This creates a TileServer object (``my_ts``) which contains information about the maps/map series you'd like to download. Its ``.metadata`` attribute is a dictionary containing the information loaded from your metadata file (geometry, coordinates, etc.). This metadata can be used to understand your maps/map series and decide which to download.

For example, to plot a histogram of the publication dates of all maps included in your metadata, use: 

.. code :: python

     my_ts.hist_published_dates()

.. image:: ../figures/hist_published_dates.png
     :width: 400px
     :align: center


Or, to visualise the boundaries of all maps included in your metadata, use: 

.. code :: python

     my_ts.plot_metadata_on_map(add_text=True)

.. image:: ../figures/plot_metadata_on_map.png
     :width: 400px
     :align: center

To find valid ranges of latitudes and longitudes to use for querying, you can find the minimum and maximum of latitudes and longitudes of all maps included in your metadata using:

.. code :: python

     my_ts.minmax_latlon()

And finally, to query maps using latitudes and longitudes, use: 

.. code :: python

     my_ts.query_point([lat,lon])
     my_ts.print_found_queries()

or: 

.. code :: python

     my_ts.query_point([[lat1,lon1],[lat2,lon2],...])
     my_ts.print_found_queries()

By default, only the most recent query will be stored in memory. 
This can be changed, by pecifying ``append = True``, thereby allowing multiple query results to be stored and accessed.

e.g.: 

.. code :: python

     my_ts.query_point([55.9,-4.2])
     my_ts.query_point([57.1,-2.5], append=True)
     my_ts.query_point([56.4,-3.5], append=True)
     my_ts.print_found_queries()

Finally, to download maps from TileServer_, use: 

.. code :: python
  
    my_ts.download_tileserver()

By default, this downloads only queried maps (i.e. those returned by ``ts.print_found_queries()``), but can be set to download all maps from the metadata using ``mode = "all"``: 

.. code :: python

     my_ts.download_tileserver(mode="all")

Running the ``download_tileserver`` command downloads maps as ``.png`` files to a newly created ``./maps`` directory.
Metadata is also stored there as a ``.csv`` file named ``metadata.csv``.
Both the default output directory name and metadata file name can be changed by specifying ``output_maps_dirname`` and ``output_metadata_filename`` respectively: 

.. code :: python
  
     my_ts.download_tileserver(output_maps_dirname="./path/to/directory", output_metadata_filename="filename.csv")


Via Azure-Blob-Storage_
-------------------------

TBC


.. _TileServer: http://tileserver.org/
.. _Azure-Blob-Storage: https://azure.microsoft.com/en-gb/products/storage/blobs/ 

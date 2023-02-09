Download
=========

TileServer_ provides an easy way to download maps, e.g. 

    - `OS one-inch 2nd edition layer <https://mapseries-tilesets.s3.amazonaws.com/1inch_2nd_ed/index.html>`__
    - `OS six-inch 1st edition layer for Scotland <https://mapseries-tilesets.s3.amazonaws.com/os/6inchfirst/index.html>`__

MapReader's TileServer class is used to read in and query map metadata and download maps from TileServer_.

The user needs to provide metadata, containing information about maps stored on TileServer to initialise this class. This metadata is then stored in ``TileServer.metadata``. 
Some example metadata files, corresponding to the one-inch and six-inch OS maps detailed above, are provided in ``MapReader/mapreader/persistent_data``.

To load in metadata using MapReader, use: 

.. code :: python

     from mapreader import TileServer
     ts = TileServer(metadata_path="path/to/metadata.json")

e.g. to download metadata corresponding to the one-inch OS maps: 

.. code :: python

     ts = TileServer(metadata_path="~/MapReader/mapreader/persistent_data/metadata_OS_One_Inch_GB_WFS_light.json", download_url="https://mapseries-tilesets.s3.amazonaws.com/1inch_2nd_ed/{z}/{x}/{y}.png")

By default, MapReader's TileServer class uses the ``download_url`` of OS one-inch 2nd edition layer (shown above), but this can easily be changed by specifying ``download_url``. 
**To access other map series, please contact NLS**.

To visualise the boundaries of your maps from metadata, use: 

.. code :: python

     ts.plot_metadata_on_map(add_text=True)

.. image:: ../figures/plot_metadata_on_map.png
     :width: 400px
     :align: center

And, to plot a histogram of the publication dates of your maps from metadata, use: 

.. code :: python

     ts.hist_published_dates()

.. image:: ../figures/hist_published_dates.png
     :width: 400px
     :align: center

It can also be useful to find the minimum and maximum of latitudes and longitudes of your maps for querying. 
This can be done using: 

.. code :: python

     ts.minmax_latlon()

Then, to query maps from metadata, use: 

.. code :: python

     ts.query_point([lat,lon])
     ts.print_found_queries()

or: 

.. code :: python

     ts.query_point([[lat1,lon1],[lat2,lon2],...])
     ts.print_found_queries()

By default, only the most recent query will be stored in memory. 
However, specifying ``append = True`` allows multiple query results to be stored and accessed: 

e.g.: 

.. code :: python

     ts.query_point([55.9,-4.2])
     ts.query_point([57.1,-2.5], append=True)
     ts.query_point([56.4,-3.5], append=True)
     ts.print_found_queries()

Finally, to download maps from TileServer_, use: 

.. code :: python
  
    ts.download_tileserver()

By default, this downloads only queried maps (i.e. those returned by ``ts.print_found_queries()``), but can be set to download all maps from the metadata using ``mode = "all"``: 

.. code :: python

     ts.download_tileserver(mode="all")

Running the ``download_tileserver`` command downloads maps as ``.png`` files to a newly created ``./maps`` directory.
Metadata is also stored here as a ``.csv`` file named ``metadata.csv``.
Both the default output directory name and metadata file name can be changed by specifying ``output_maps_dirname`` and ``output_metadata_filename`` respectively: 

.. code :: python
  
     ts.download_tileserver(output_maps_dirname="./path/to/directory", output_metadata_filename="filename.csv")

Other important arguments for the ``download_tileserver`` function which you may want to specify include:

    - zoom_level: 
    - pixel closest: 

.. unsure exactly what these two arguments do. Maybe someone more familiar with MR can add - RW

.. _TileServer: http://tileserver.org/

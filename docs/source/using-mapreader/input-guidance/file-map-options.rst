File/Map Options
================

The MapReader pipeline is explained in detail :doc:`here </introduction-to-mapreader/what-is-mapreader>`.
The inputs you will need for MapReader will depend on where you begin within the pipeline.

Option 1 - If the map(s) you want have been georeferenced and made available via a Tile Server
------------------------------------------------------------------------------------------------

Some GLAM institutions or other services make digitized, georeferenced maps available via tile servers, for example as raster (XYZ, or 'slippy map') tiles.

Instructions for accessing tile layers from one example collection is below:

- National Library of Scotland tile layers

If you want to download maps from a TileServer using MapReader's ``Download`` subpackage, you will need to begin with the 'Download' task.
For this, you will need:

* A `geojson <https://geojson.org/>`__ file containing metadata for each map sheet you would like to query/download.
* The URL of the XYZ tile layer which you would like to access.

At a minimum, for each map sheet, your geojson file should contain information on:

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
--------------------------------------------------

If you already have your maps saved locally, you can skip the 'Download' task and move straight to 'Load'.

If you would like to work with georeferenced maps, you will need either:

* Your map images saved as standard, non-georeferenced, image files (e.g. JPEG, PNG or TIFF) along with a separate file containing georeferencing metadata you wish to associate to these map images **OR**
* You map images saved as georeferenced image files (e.g. geoTIFF).

Alternatively, if you would like to work with non-georeferenced maps/images, you will need:

* Your images saved as standard image files (e.g. JPEG, PNG or TIFF).

.. note:: It is possible to use non-georeferenced maps in MapReader, however none of the functionality around plotting patches based on geospatial coordinates will be possible. In this case, patches can be analyzed as regions within a map sheet, where the sheet itself may have some geospatial information associated with it (e.g. the geospatial coordinates for its center point, or the place name in its title).

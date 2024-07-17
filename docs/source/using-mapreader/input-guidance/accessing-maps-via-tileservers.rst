Accessing maps via TileServers
==============================

National Library of Scotland
----------------------------

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

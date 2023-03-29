:py:mod:`mapreader.download.tileserver_scraper`
===============================================

.. py:module:: mapreader.download.tileserver_scraper

.. autoapi-nested-parse::

   Scraper for tileserver

   The main code for the scraper was sourced from a repository located at
   https://github.com/stamen/the-ultimate-tile-stitcher, which is licensed under
   the MIT license. The adapted functions were then used to run the scraper via
   Python modules.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   mapreader.download.tileserver_scraper.tile_idxs_in_poly
   mapreader.download.tileserver_scraper.fetch_and_save
   mapreader.download.tileserver_scraper.runner
   mapreader.download.tileserver_scraper.scraper



.. py:function:: tile_idxs_in_poly(poly, zoom)

   Given a ``shapely.geometry.Polygon`` and a ``zoom`` (zoom level), generate
   a sequence of ``(x, y)`` tile indices that intersect with the Polygon.

   Parameters
   ----------
   poly : shapely.geometry.Polygon
       The Polygon object to intersect with tile indices.
   zoom : int
       The zoom level of the map.

   Raises
   ------
   TypeError
       If the ``poly`` parameter is not a ``shapely.geometry.Polygon`` object.

   Yields
   ------
   Tuple[int, int]
       A tuple of tile indices ``(x, y)`` that intersect with the Polygon
       object.


.. py:function:: fetch_and_save(session, url, retries, filepath, **kwargs)
   :async:

   Fetch an image from the specified URL using the specified ``aiohttp``
   session, and save it to a file (``filepath``). The function retries
   fetching the image for the specified number of times (``retries``).
   If the image is fetched successfully, the function returns ``True``;
   otherwise, it returns ``False``.

   Parameters
   ----------
   session : aiohttp.ClientSession
       The ``aiohttp`` session used to fetch the image.
   url : str
       The URL of the image to fetch.
   retries : int
       The number of times to retry fetching the image.
   filepath : str
       The file path where the image will be saved.
   **kwargs
       Optional keyword arguments that will be passed to the ``session.get()``
       method.

   Raises
   ------
   aiohttp.ClientError
       If any ``aiohttp`` client error occurs during the image fetching
       process.

   Returns
   -------
   bool
       ``True`` if the image is fetched successfully and saved to the
       specified file path, ``False`` otherwise.


.. py:function:: runner(opts)
   :async:

   Downloads tiles from a specified URL and saves them to disk within a
   specified polygon. Returns a list of URLs that failed to download.

   Parameters
   ----------
   opts : input_class
       The options for downloading the tiles, of the ``input_class`` type
       that contains the following attributes:
       
       - ``poly`` (shapely.geometry.Polygon): The polygon (in GeoJSON format).
       - ``zoom`` (int): The zoom level.
       - ``url`` (str): The URL string (formatted with ``"{x}"``, ``"{y}"`` and ``"{z}"``)
       - ``out_dir`` (str): The output file directory for resulting files.
       - ``retries`` (int): The number of retries to attempt to download the image.
       - ``max_connections`` (int): The number of maximum connections to pass onto Semaphore.

   Returns
   -------
   List[str]
       A list of URLs that failed to download.

   Notes
   -----
   This function is usually called through the
   :func:`mapreader.download.tileserver_scraper.scraper` function. Refer to
   the documentation of that method for a simpler implementation.


.. py:function:: scraper(poly, zoom, url, out_dir, max_connections = 20, retries = 10)

   Downloads tiles from the specified URL and saves them to disk within the
   specified polygon. If any tiles fail to download, writes a list of the
   failed URLs to a file.

   Parameters
   ----------
   poly : str
       The path to the GeoJSON file defining the polygon.
   zoom : int
       The zoom level at which to download tiles.
   url : str
       The URL pattern for the tiles to download.
   out_dir : str
       The directory in which to save the downloaded tiles.
   max_connections : int, optional
       The maximum number of simultaneous connections to use when
       downloading, by default ``20``.
   retries : int, optional
       The maximum number of times to retry a failed download, by default
       ``10``.

   Returns
   -------
   None



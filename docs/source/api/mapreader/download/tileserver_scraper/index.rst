:py:mod:`mapreader.download.tileserver_scraper`
===============================================

.. py:module:: mapreader.download.tileserver_scraper

.. autoapi-nested-parse::

   Scraper for tileserver

   The main-part/most of these codes are from the following repo:

   https://github.com/stamen/the-ultimate-tile-stitcher

   (released under MIT license)

   Here, we adapted the functions to run them via Python modules



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   mapreader.download.tileserver_scraper.tile_idxs_in_poly
   mapreader.download.tileserver_scraper.fetch_and_save
   mapreader.download.tileserver_scraper.runner
   mapreader.download.tileserver_scraper.scraper



Attributes
~~~~~~~~~~

.. autoapisummary::

   mapreader.download.tileserver_scraper.BASE_WAIT


.. py:data:: BASE_WAIT
   :value: 0.5

   

.. py:function:: tile_idxs_in_poly(poly: shapely.geometry.Polygon, zoom: int)


.. py:function:: fetch_and_save(session: aiohttp.ClientSession, url: str, retries: int, filepath: str, **kwargs)
   :async:


.. py:function:: runner(opts)
   :async:


.. py:function:: scraper(poly, zoom, url, out_dir, max_connections=20, retries=10)



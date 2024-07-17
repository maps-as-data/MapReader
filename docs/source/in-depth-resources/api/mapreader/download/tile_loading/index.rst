mapreader.download.tile_loading
===============================

.. py:module:: mapreader.download.tile_loading


Attributes
----------

.. autoapisummary::

   mapreader.download.tile_loading.logger
   mapreader.download.tile_loading.DEFAULT_TEMP_FOLDER
   mapreader.download.tile_loading.DEFAULT_IMG_DOWNLOAD_FORMAT


Classes
-------

.. autoapisummary::

   mapreader.download.tile_loading.TileDownloader


Module Contents
---------------

.. py:data:: logger

.. py:data:: DEFAULT_TEMP_FOLDER
   :value: '_tile_cache/'


.. py:data:: DEFAULT_IMG_DOWNLOAD_FORMAT
   :value: 'png'


.. py:class:: TileDownloader(tile_servers = None, img_format = None, show_progress = False)

   .. py:method:: generate_tile_name(index)

      Generates tile file names from GridIndex.

      :param index:
      :type index: GridIndex

      :returns: Tile file name
      :rtype: str



   .. py:method:: generate_tile_url(index, subserver_index)

      Generates tile download urls from GridIndex.

      :param index:
      :type index: GridIndex
      :param subserver_index: Index no. of subserver to use for download
      :type subserver_index: int

      :returns: Tile download url
      :rtype: str



   .. py:method:: download_tiles(grid_bb, download_in_parallel = True)

      Downloads tiles contained within GridBoundingBox.

      :param grid_bb: GridBoundingBox containing tiles to download
      :type grid_bb: GridBoundingBox
      :param download_in_parallel: Whether or not to download tiles in parallel, by default True
      :type download_in_parallel: bool, optional

      :rtype: xxxx

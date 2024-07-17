mapreader.download.tile_merging
===============================

.. py:module:: mapreader.download.tile_merging


Attributes
----------

.. autoapisummary::

   mapreader.download.tile_merging.logger
   mapreader.download.tile_merging.DEFAULT_OUT_FOLDER
   mapreader.download.tile_merging.DEFAULT_IMG_STORE_FORMAT


Classes
-------

.. autoapisummary::

   mapreader.download.tile_merging.TileMerger


Module Contents
---------------

.. py:data:: logger

.. py:data:: DEFAULT_OUT_FOLDER
   :value: './'


.. py:data:: DEFAULT_IMG_STORE_FORMAT
   :value: ('png', 'PNG')


.. py:class:: TileMerger(output_folder = None, img_input_format = None, img_output_format = None, show_progress=False)

   .. py:method:: merge(grid_bb, file_name = None, overwrite = False)

      Merges cells contained within GridBoundingBox.

      :param grid_bb: GridBoundingBox containing tiles to merge
      :type grid_bb: GridBoundingBox
      :param file_name: Name to use when saving map
                        If None, default name will be used, by default None
      :type file_name: Union[str, None], optional
      :param overwrite: Whether or not to overwrite existing files, by default False
      :type overwrite: bool, optional

      :returns: out path if file has successfully downloaded, False if not.
      :rtype: str or bool

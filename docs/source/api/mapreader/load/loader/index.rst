:py:mod:`mapreader.load.loader`
===============================

.. py:module:: mapreader.load.loader


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   mapreader.load.loader.loader
   mapreader.load.loader.load_patches



.. py:function:: loader(path_images=False, tree_level='parent', parent_path=None, **kwds)

   Construct mapImages object by passing the image path,

   Keyword Arguments:
       path_images {str or False} -- path to one or many images

   Returns:
       [mapImages object] -- mapImages object contains various methods to work with images


.. py:function:: load_patches(patch_paths, parent_paths=False, add_geo_par=False, clear_images=False)



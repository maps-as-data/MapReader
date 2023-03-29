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



.. py:function:: loader(path_images = None, tree_level = 'parent', parent_path = None, **kwds)

   Creates a ``mapImages`` class to manage a collection of image paths and
   construct image objects.

   Parameters
   ----------
   path_images : str or None, optional
       Path to the directory containing images (accepts wildcards). By
       default, ``None``
   tree_level : str, optional
       Level of the image hierarchy to construct. The value can be
       ``"parent"`` (default) and ``"patch"``.
   parent_path : str, optional
       Path to parent images (if applicable), by default ``None``.
   **kwds : dict, optional
       Additional keyword arguments to be passed to the ``imagesConstructor``
       method.

   Returns
   -------
   mapImages
       The ``mapImages`` class which can manage a collection of image paths
       and construct image objects.

   Notes
   -----
   This is a wrapper method. See the documentation of the
   :class:`mapreader.load.images.mapImages` class for more detail.


.. py:function:: load_patches(patch_paths, parent_paths = False, add_geo_par = False, clear_images = False)

   Creates a ``mapImages`` class to manage a collection of image paths and
   construct image objects. Then loads patch images from the given paths and
   adds them to the ``images`` dictionary in the ``mapImages`` instance.

   Parameters
   ----------
   patch_paths : str
       The file path of the patches to be loaded.

       *Note: The ``patch_paths`` parameter accepts wildcards.*
   parent_paths : str or bool, optional
       The file path of the parent images to be loaded. If set to
       ``False``, no parents are loaded. Default is ``False``.

       *Note: The ``parent_paths`` parameter accepts wildcards.*
   add_geo_par : bool, optional
       If ``True``, adds geographic information to the parent image.
       Default is ``False``.
   clear_images : bool, optional
       If ``True``, clears the images from the ``images`` dictionary
       before loading. Default is ``False``.

   Returns
   -------
   mapImages
       The ``mapImages`` class which can manage a collection of image paths
       and construct image objects.

   Notes
   -----
   This is a wrapper method. See the documentation of the
   :class:`mapreader.load.images.mapImages` class for more detail.

   This function in particular, also calls the
   :meth:`mapreader.load.images.mapImages.loadPatches` method. Please see
   the documentation for that method for more information as well.



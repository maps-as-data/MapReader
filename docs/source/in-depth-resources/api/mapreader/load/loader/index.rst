mapreader.load.loader
=====================

.. py:module:: mapreader.load.loader


Functions
---------

.. autoapisummary::

   mapreader.load.loader.loader
   mapreader.load.loader.load_patches


Module Contents
---------------

.. py:function:: loader(path_images = None, tree_level = 'parent', parent_path = None, **kwargs)

   Creates a ``MapImages`` class to manage a collection of image paths and
   construct image objects.

   :param path_images: Path to the directory containing images (accepts wildcards). By
                       default, ``None``
   :type path_images: str or None, optional
   :param tree_level: Level of the image hierarchy to construct. The value can be
                      ``"parent"`` (default) and ``"patch"``.
   :type tree_level: str, optional
   :param parent_path: Path to parent images (if applicable), by default ``None``.
   :type parent_path: str, optional
   :param \*\*kwargs: Additional keyword arguments to be passed to the ``_images_constructor()``
                      method.
   :type \*\*kwargs: dict, optional

   :returns: The ``MapImages`` class which can manage a collection of image paths
             and construct image objects.
   :rtype: MapImages

   .. rubric:: Notes

   This is a wrapper method. See the documentation of the
   :class:`mapreader.load.images.MapImages` class for more detail.


.. py:function:: load_patches(patch_paths, patch_file_ext = False, parent_paths = False, parent_file_ext = False, add_geo_info = False, clear_images = False)

   Creates a ``MapImages`` class to manage a collection of image paths and
   construct image objects. Then loads patch images from the given paths and
   adds them to the ``images`` dictionary in the ``MapImages`` instance.

   :param patch_paths: The file path of the patches to be loaded.

                       *Note: The ``patch_paths`` parameter accepts wildcards.*
   :type patch_paths: str
   :param patch_file_ext: The file extension of the patches, ignored if file extensions are specified in ``patch_paths`` (e.g. with ``"./path/to/dir/*png"``)
                          By default ``False``.
   :type patch_file_ext: str or bool, optional
   :param parent_paths: The file path of the parent images to be loaded. If set to
                        ``False``, no parents are loaded. Default is ``False``.

                        *Note: The ``parent_paths`` parameter accepts wildcards.*
   :type parent_paths: str or bool, optional
   :param parent_file_ext: The file extension of the parent images, ignored if file extensions are specified in ``parent_paths`` (e.g. with ``"./path/to/dir/*png"``)
                           By default ``False``.
   :type parent_file_ext: str or bool, optional
   :param add_geo_info: If ``True``, adds geographic information to the parent image.
                        Default is ``False``.
   :type add_geo_info: bool, optional
   :param clear_images: If ``True``, clears the images from the ``images`` dictionary
                        before loading. Default is ``False``.
   :type clear_images: bool, optional

   :returns: The ``MapImages`` class which can manage a collection of image paths
             and construct image objects.
   :rtype: MapImages

   .. rubric:: Notes

   This is a wrapper method. See the documentation of the
   :class:`mapreader.load.images.MapImages` class for more detail.

   This function in particular, also calls the
   :meth:`mapreader.load.images.MapImages.loadPatches` method. Please see
   the documentation for that method for more information as well.

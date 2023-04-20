#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mapreader.load.images import mapImages
from typing import Optional, Union, Dict


def loader(
    path_images: Optional[str] = None,
    tree_level: Optional[str] = "parent",
    parent_path: Optional[str] = None,
    **kwds: Dict
) -> mapImages:
    """
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
    """
    img = mapImages(
        path_images=path_images, tree_level=tree_level, parent_path=parent_path, **kwds
    )
    return img


def load_patches(
    patch_paths: str,
    parent_paths: Optional[Union[str, bool]] = False,
    add_geo_par: Optional[bool] = False,
    clear_images: Optional[bool] = False,
) -> mapImages:
    """
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
    """
    img = mapImages()
    img.loadPatches(
        patch_paths=patch_paths,
        parent_paths=parent_paths,
        add_geo_par=add_geo_par,
        clear_images=clear_images,
    )
    return img


'''
def read_patches(patch_paths, parent_paths, metadata=None,
               metadata_fmt="dataframe", metadata_cols2add=[], metadata_index_column="image_id",  # noqa
               clear_images=False):
    """Construct mapImages object by calling readPatches method
       This method reads patches from files (patch_paths) and add parents if parent_paths is provided  # noqa

    Arguments:
        patch_paths {str, wildcard accepted} -- path to patches
        parent_paths {False or str, wildcard accepted} -- path to parents
        metadata_path -- path to metadata
        metadata_fmt -- format of the metadata file (default: csv)
        metadata_index_column -- column to be used as index
        clear_images {bool} -- clear images variable before reading patches (default: {False})  # noqa
    """
    img = mapImages()
    img.readPatches(patch_paths, parent_paths, metadata, metadata_fmt, metadata_cols2add, metadata_index_column, clear_images)  # noqa
    return img
'''

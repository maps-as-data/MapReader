#!/usr/bin/env python
from __future__ import annotations

from mapreader.load.images import MapImages


def loader(
    path_images: str | None = None,
    tree_level: str | None = "parent",
    parent_path: str | None = None,
    **kwargs: dict,
) -> MapImages:
    """
    Creates a :class:`~.load.images.MapImages` class to manage a collection of image paths and
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
    **kwargs : dict, optional
        Additional keyword arguments to be passed to the ``_images_constructor()``
        method.

    Returns
    -------
    MapImages
        The :class:`~.load.images.MapImages` class which can manage a
        collection of image paths and construct image objects.

    Notes
    -----
    This is a wrapper method. See the documentation of the
    :class:`~.load.images.MapImages` class for more detail.
    """
    img = MapImages(
        path_images=path_images,
        tree_level=tree_level,
        parent_path=parent_path,
        **kwargs,
    )
    return img


def load_patches(
    patch_paths: str,
    patch_file_ext: str | bool | None = False,
    parent_paths: str | bool | None = False,
    parent_file_ext: str | bool | None = False,
    add_geo_info: bool | None = False,
    clear_images: bool | None = False,
) -> MapImages:
    """
    Creates a :class:`~.load.images.MapImages` class to manage a collection of
    image paths and construct image objects. Then loads patch images from the
    given paths and adds them to the ``images`` dictionary in the
    :class:`~.load.images.MapImages` instance.

    Parameters
    ----------
    patch_paths : str
        The file path of the patches to be loaded.

        *Note: The ``patch_paths`` parameter accepts wildcards.*
    patch_file_ext : str or bool, optional
        The file extension of the patches, ignored if file extensions are
        specified in ``patch_paths`` (e.g. with ``"./path/to/dir/*png"``)
        By default ``False``.
    parent_paths : str or bool, optional
        The file path of the parent images to be loaded. If set to
        ``False``, no parents are loaded. Default is ``False``.

        *Note: The ``parent_paths`` parameter accepts wildcards.*
    parent_file_ext : str or bool, optional
        The file extension of the parent images, ignored if file extensions
        are specified in ``parent_paths`` (e.g. with ``"./path/to/dir/*png"``)
        By default ``False``.
    add_geo_info : bool, optional
        If ``True``, adds geographic information to the parent image.
        Default is ``False``.
    clear_images : bool, optional
        If ``True``, clears the images from the ``images`` dictionary
        before loading. Default is ``False``.

    Returns
    -------
    MapImages
        The :class:`~.load.images.MapImages` class which can manage a
        collection of image paths and construct image objects.

    Notes
    -----
    This is a wrapper method. See the documentation of the
    :class:`~.load.images.MapImages` class for more detail.

    This function in particular, also calls the
    :meth:`~.load.images.MapImages.load_patches` method. Please see
    the documentation for that method for more information as well.
    """
    img = MapImages()
    img.load_patches(
        patch_paths=patch_paths,
        patch_file_ext=patch_file_ext,
        parent_paths=parent_paths,
        parent_file_ext=parent_file_ext,
        add_geo_info=add_geo_info,
        clear_images=clear_images,
    )
    return img


'''
def read_patches(patch_paths, parent_paths, metadata=None,
               metadata_fmt="dataframe", metadata_cols2add=[], metadata_index_column="image_id",  # noqa
               clear_images=False):
    """Construct MapImages object by calling readPatches method
       This method reads patches from files (patch_paths) and add parents if parent_paths is provided  # noqa

    Arguments:
        patch_paths {str, wildcard accepted} -- path to patches
        parent_paths {False or str, wildcard accepted} -- path to parents
        metadata_path -- path to metadata
        metadata_fmt -- format of the metadata file (default: csv)
        metadata_index_column -- column to be used as index
        clear_images {bool} -- clear images variable before reading patches (default: {False})  # noqa
    """
    img = MapImages()
    img.readPatches(patch_paths, parent_paths, metadata, metadata_fmt, metadata_cols2add, metadata_index_column, clear_images)  # noqa
    return img
'''

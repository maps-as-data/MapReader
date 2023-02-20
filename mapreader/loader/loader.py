#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .images import mapImages

def loader(path_images=False, tree_level="parent", parent_path=None, **kwds):
    """Construct mapImages object by passing image path

    Parameters
    ----------
    path_images : str or False, optional
        Path to images, by default False
    tree_level : str, optional
        Tree level, choices between "parent" or "child", by default "parent"
    parent_path : str or None, optional
        Path to parent images, by default None

    Returns
    -------
    [mapImages object]
        mapImages object containing various methods to work with images
    """

    img = mapImages(
        path_images=path_images, tree_level=tree_level, parent_path=parent_path, **kwds
    )
    return img


def load_patches(
    patch_paths, parent_paths=False, add_geo_par=False, clear_images=False
):
    """Load patches from path and, if parent_paths specified, add parents

    Parameters
    ----------
    patch_paths : str
        Path to patches, accepts wildcards
    parent_paths : str or False, optional
        Path to parents, accepts wildcards
        If False, no parents are loaded, by default False
    add_geo_par : bool, optional
        Add geographical info to parents, by default False
    clear_images : bool, optional
        Clear images variable before loading, by default False

    Returns
    -------
    [mapImages object]
        mapImages object containing various methods to work with images
    """

    img = mapImages()
    img.loadPatches(
        patch_paths=patch_paths,
        parent_paths=parent_paths,
        add_geo_par=add_geo_par,
        clear_images=clear_images,
    )
    return img


### def read_patches(patch_paths, parent_paths, metadata=None,
###                metadata_fmt="dataframe", metadata_cols2add=[], metadata_index_column="image_id",
###                clear_images=False):
###     """Construct mapImages object by calling readPatches method
###        This method reads patches from files (patch_paths) and add parents if parent_paths is provided
###
###     Arguments:
###         patch_paths {str, wildcard accepted} -- path to patches
###         parent_paths {False or str, wildcard accepted} -- path to parents
###         metadata_path -- path to metadata
###         metadata_fmt -- format of the metadata file (default: csv)
###         metadata_index_column -- column to be used as index
###         clear_images {bool} -- clear images variable before reading patches (default: {False})
###     """
###     img = mapImages()
###     img.readPatches(patch_paths, parent_paths, metadata, metadata_fmt, metadata_cols2add, metadata_index_column, clear_images)
###     return img
###

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mapreader.loader.images import mapImages


def loader(path_images=False, tree_level="parent", parent_path=None, **kwds):
    """Construct mapImages object by passing the image path,

    Keyword Arguments:
        path_images {str or False} -- path to one or many images

    Returns:
        [mapImages object] -- mapImages object contains various methods to work with images
    """
    img = mapImages(
        path_images=path_images, tree_level=tree_level, parent_path=parent_path, **kwds
    )
    return img


def load_patches(
    patch_paths, patch_file_ext=False, parent_paths=False, parent_file_ext=False, add_geo_par=False, clear_images=False
):

    img = mapImages()
    img.loadPatches(
        patch_paths=patch_paths,
        patch_file_ext=patch_file_ext,
        parent_paths=parent_paths,
        parent_file_ext=parent_file_ext,
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

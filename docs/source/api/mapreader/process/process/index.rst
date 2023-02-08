:py:mod:`mapreader.process.process`
===================================

.. py:module:: mapreader.process.process


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   mapreader.process.process.preprocess_all
   mapreader.process.process.preprocess



.. py:function:: preprocess_all(image_paths, save_preproc_dir, **kwds)

   Preprocess a list of images

   Args:
       image_paths (list or path): a path (wildcard accepted) or a list of paths to images to be preprocessed
       save_preproc_dir (str, path): path to save preprocessed images


.. py:function:: preprocess(image_path, save_preproc_dir, dst_crs='EPSG:3857', crop_prefix='preproc_', reproj_prefix='preproc_tmp_', resample_prefix='preproc_resample_', resize_percent=40, remove_reproj_file=True)

   preprocess an image

   Preprocessing has three steps:
   - reproject maps to dst_crs
   - crop images by removing the white borders
   - resanme using resize_percent

   Args:
       image_path (str, path): path to an image to be preprocessed
       save_preproc_dir (str, path): path to save preprocessed image
       dst_crs (str, optional): target map projection. Defaults to 'EPSG:3857'.
       crop_prefix (str, optional): prefix to cropped image filename. Defaults to "preproc_".
       reproj_prefix (str, optional): prefix to reprojected image filename. Defaults to "preproc_tmp_".
       resample_prefix (str, optional): prefix to resnamed image filename. Defaults to "preproc_resample_".
       resize_percent (int, optional): resize images by to this. Defaults to 40.
       remove_reproj_file (bool, optional): after preprocessing is finished, cleanup the files. Defaults to True.



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

   Preprocess all images in a list of file paths or a directory using the
   ``preprocess`` function and save them to the specified directory.

   Parameters
   ----------
   image_paths : str or list of str
       Either a string representing the path to a directory containing
       images (wildcards accepted), or a list of file paths representing
       individual images to be preprocessed.
   save_preproc_dir : str
       The path to the directory where preprocessed images will be saved.
   **kwds : keyword arguments
       Additional keyword arguments to be passed to the ``preprocess``
       function.

   Returns
   -------
   saved_paths : list of str
       A list containing the file paths of the preprocessed images that were
       saved.


.. py:function:: preprocess(image_path, save_preproc_dir, dst_crs = 'EPSG:3857', crop_prefix = 'preproc_', reproj_prefix = 'preproc_tmp_', resample_prefix = 'preproc_resample_', resize_percent = 40, remove_reproj_file = True)

   Preprocesses an image file by reprojecting it to a new coordinate
   reference system, cropping (removing white borders) and resampling it to a
   given percentage size.

   Parameters
   ----------
   image_path : str
       The path to the input image file to be preprocessed.
   save_preproc_dir : str
       The directory to save the preprocessed image files.
   dst_crs : str, optional
       The coordinate reference system to reproject the image to, by default
       ``"EPSG:3857"``.
   crop_prefix : str, optional
       The prefix to use for the cropped image file, by default
       ``"preproc_"``.
   reproj_prefix : str, optional
       The prefix to use for the reprojected image file, by default
       ``"preproc_tmp_"``.
   resample_prefix : str, optional
       The prefix to use for the resampled image file, by default
       ``"preproc_resample_"``.
   resize_percent : int, optional
       The percentage to resize the cropped image by, by default ``40``.
   remove_reproj_file : bool, optional
       Whether to remove the reprojected image file after preprocessing, by
       default ``True``.

   Returns
   -------
   str
       The path to the resampled image file if preprocessing was successful,
       otherwise the path to the cropped image file, or ``"False"`` if
       preprocessing failed.



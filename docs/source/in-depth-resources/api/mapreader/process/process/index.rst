mapreader.process.process
=========================

.. py:module:: mapreader.process.process


Functions
---------

.. autoapisummary::

   mapreader.process.process.preprocess_all
   mapreader.process.process.preprocess


Module Contents
---------------

.. py:function:: preprocess_all(image_paths, save_preproc_dir, **kwds)

   Preprocess all images in a list of file paths or a directory using the
   ``preprocess`` function and save them to the specified directory.

   :param image_paths: Either a string representing the path to a directory containing
                       images (wildcards accepted), or a list of file paths representing
                       individual images to be preprocessed.
   :type image_paths: str or list of str
   :param save_preproc_dir: The path to the directory where preprocessed images will be saved.
   :type save_preproc_dir: str
   :param \*\*kwds: Additional keyword arguments to be passed to the ``preprocess``
                    function.
   :type \*\*kwds: keyword arguments

   :returns: **saved_paths** -- A list containing the file paths of the preprocessed images that were
             saved.
   :rtype: list of str


.. py:function:: preprocess(image_path, save_preproc_dir, dst_crs = 'EPSG:3857', crop_prefix = 'preproc_', reproj_prefix = 'preproc_tmp_', resample_prefix = 'preproc_resample_', resize_percent = 40, remove_reproj_file = True)

   Preprocesses an image file by reprojecting it to a new coordinate
   reference system, cropping (removing white borders) and resampling it to a
   given percentage size.

   :param image_path: The path to the input image file to be preprocessed.
   :type image_path: str
   :param save_preproc_dir: The directory to save the preprocessed image files.
   :type save_preproc_dir: str
   :param dst_crs: The coordinate reference system to reproject the image to, by default
                   ``"EPSG:3857"``.
   :type dst_crs: str, optional
   :param crop_prefix: The prefix to use for the cropped image file, by default
                       ``"preproc_"``.
   :type crop_prefix: str, optional
   :param reproj_prefix: The prefix to use for the reprojected image file, by default
                         ``"preproc_tmp_"``.
   :type reproj_prefix: str, optional
   :param resample_prefix: The prefix to use for the resampled image file, by default
                           ``"preproc_resample_"``.
   :type resample_prefix: str, optional
   :param resize_percent: The percentage to resize the cropped image by, by default ``40``.
   :type resize_percent: int, optional
   :param remove_reproj_file: Whether to remove the reprojected image file after preprocessing, by
                              default ``True``.
   :type remove_reproj_file: bool, optional

   :returns: The path to the resampled image file if preprocessing was successful,
             otherwise the path to the cropped image file, or ``"False"`` if
             preprocessing failed.
   :rtype: str

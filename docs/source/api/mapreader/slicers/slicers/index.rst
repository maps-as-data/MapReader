:py:mod:`mapreader.slicers.slicers`
===================================

.. py:module:: mapreader.slicers.slicers


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   mapreader.slicers.slicers.sliceByPixel



.. py:function:: sliceByPixel(image_path, slice_size, path_save='sliced_images', square_cuts=False, resize_factor=False, output_format='png', rewrite=False, verbose=False)

   Slice an image by pixels

   Parameters
   ----------
   image_path : str
       Path to image
   slice_size : int
       Number of pixels/meters in both x and y to use for slicing
   path_save : str, optional
       Directory to save the sliced image, by default "sliced_images"
   square_cuts : bool, optional
       If True, all sliced images will have the same number of pixels in x and y, by default False
   resize_factor : bool, optional
       If True, resize the images before slicing, by default False
   output_format : str, optional
       Format to use when writing image files, by default "png"
   rewrite : bool, optional
       If True, existing slices will be rewritten, by default False
   verbose : bool, optional
       If True, progress updates will be printed throughout, by default False

   Returns
   -------
   list
       sliced_images_info



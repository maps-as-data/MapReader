:py:mod:`mapreader.load.slicers`
================================

.. py:module:: mapreader.load.slicers


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   mapreader.load.slicers.patchifyByPixel



.. py:function:: patchifyByPixel(image_path, patch_size, path_save='patches', square_cuts=True, resize_factor=False, output_format='png', rewrite=False, verbose=True)

   Patchify an image by pixels

   Arguments:
       image_path {str} -- Path to the image to be sliced
       patch_size {int} -- Number of pixels in both x and y directions

   Keyword Arguments:
       path_save {str} -- Directory to save the patches (default: {"patches"})
       square_cuts {bool} -- All patches will have the same number of pixels in x and y (default: {True})
       resize_factor {bool} -- Resize image before slicing (default: {False})
       output_format {str} -- Output format (default: {"png"})
       verbose {bool} -- Print the progress (default: {True})



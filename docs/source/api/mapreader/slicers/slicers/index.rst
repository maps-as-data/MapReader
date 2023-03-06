:py:mod:`mapreader.slicers.slicers`
===================================

.. py:module:: mapreader.slicers.slicers


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   mapreader.slicers.slicers.sliceByPixel



.. py:function:: sliceByPixel(image_path, slice_size, path_save='test', square_cuts=True, resize_factor=False, output_format='PNG', rewrite=False, verbose=True)

   Slice an image by pixels

   Arguments:
       image_path {str} -- Path to the image to be sliced
       slice_size {int} -- Number of pixels in both x and y directions

   Keyword Arguments:
       path_save {str} -- Directory to save the sliced images (default: {"test"})
       square_cuts {bool} -- All sliced images will have the same number of pixels in x and y (default: {True})
       resize_factor {bool} -- Resize image before slicing (default: {False})
       output_format {str} -- Output format (default: {"PNG"})
       verbose {bool} -- Print the progress (default: {True})



:py:mod:`mapreader.download.tileserver_stitcher`
================================================

.. py:module:: mapreader.download.tileserver_stitcher

.. autoapi-nested-parse::

   Stitcher for tileserver

   The main code for the stitcher was sourced from a repository located at
   https://github.com/stamen/the-ultimate-tile-stitcher, which is licensed under
   the MIT license. The adapted functions were then used to run the scraper via
   Python modules.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   mapreader.download.tileserver_stitcher.runner
   mapreader.download.tileserver_stitcher.myround
   mapreader.download.tileserver_stitcher.stitcher



.. py:function:: runner(opts)

   Stitch together a series of images into a larger image.

   Parameters
   -----------
   opts : input_class
       The options for the runner, of the ``input_class`` type that contains
       the following attributes:
       
       - ``dir`` (str): The directory containing the input images.
       - ``out_file`` (str): The output file path for the stitched image.
       - ``pixel_closest`` (int, optional): The closest pixel value to round the image size to.

   Raises
   ------
   SystemExit
       If no input files are found in the specified directory.

   Returns
   -------
   None
       The function saves the stitched image to the specified output file
       path.

   Notes
   -----
   This function is usually called through the
   :func:`mapreader.download.tileserver_stitcher.stitcher` function. Refer to
   the documentation of that method for a simpler implementation.


.. py:function:: myround(x, base = 100)

   Round a number to the nearest multiple of the given base.

   Parameters
   ----------
   x : float or int
       The number to be rounded.
   base : int, optional
       The base to which ``x`` will be rounded. Default is ``100``.

   Returns
   -------
   int
       The rounded number.

   ..
       TODO: Could we make this function private? It's only used by the
       runner above.


.. py:function:: stitcher(dir_name, out_file, pixel_closest = None)

   Stitch together multiple images from a directory and save the result to a
   file.

   Parameters
   ----------
   dir_name : str
       The directory containing the images to be stitched.
   out_file : str
       The name of the file to which the stitched image will be saved.
   pixel_closest : int or None
       The distance between the closest neighboring pixels. If ``None``, the
       optimal value will be determined automatically.

   Returns
   -------
   None



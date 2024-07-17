mapreader.utils.slice_parallel
==============================

.. py:module:: mapreader.utils.slice_parallel


Attributes
----------

.. autoapisummary::

   mapreader.utils.slice_parallel.parser


Functions
---------

.. autoapisummary::

   mapreader.utils.slice_parallel.slice_serial


Module Contents
---------------

.. py:function:: slice_serial(path2images_dir, slice_size=100, slice_method='pixel', output_dirname='slice_100_100')

   Slice images stored in path2images_dir
   This function is the serial version and will be run in parallel using parhugin


.. py:data:: parser

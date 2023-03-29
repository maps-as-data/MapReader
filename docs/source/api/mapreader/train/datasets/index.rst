:py:mod:`mapreader.train.datasets`
==================================

.. py:module:: mapreader.train.datasets


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   mapreader.train.datasets.patchTorchDataset
   mapreader.train.datasets.patchContextDataset




Attributes
~~~~~~~~~~

.. autoapisummary::

   mapreader.train.datasets.parhugin_installed


.. py:data:: parhugin_installed
   :value: True

   

.. py:class:: patchTorchDataset(patchframe, transform = None, label_col = 'label', convert2 = 'RGB', input_col = 0)

   Bases: :py:obj:`torch.utils.data.Dataset`

   An abstract class representing a :class:`Dataset`.

   All datasets that represent a map from keys to data samples should subclass
   it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
   data sample for a given key. Subclasses could also optionally overwrite
   :meth:`__len__`, which is expected to return the size of the dataset by many
   :class:`~torch.utils.data.Sampler` implementations and the default options
   of :class:`~torch.utils.data.DataLoader`.

   .. note::
     :class:`~torch.utils.data.DataLoader` by default constructs a index
     sampler that yields integral indices.  To make it work with a map-style
     dataset with non-integral indices/keys, a custom sampler must be provided.

   .. py:method:: return_orig_image(idx)

      Return the original image associated with the given index.

      Parameters
      ----------
      idx : int or Tensor
          The index of the desired image, or a Tensor containing the index.

      Returns
      -------
      PIL.Image.Image
          The original image associated with the given index.

      Notes
      -----
      This method returns the original image associated with the given index
      by loading the image file using the file path stored in the
      ``input_col`` column of the ``patchframe`` DataFrame at the given
      index. The loaded image is then converted to the format specified by
      the ``convert2`` attribute of the object. The resulting
      ``PIL.Image.Image`` object is returned.



.. py:class:: patchContextDataset(patchframe, transform1 = None, transform2 = None, label_col = 'label', convert2 = 'RGB', input_col = 0, context_save_path = './maps/maps_context', create_context = False, par_path = './maps', x_offset = 1.0, y_offset = 1.0, slice_method = 'scale')

   Bases: :py:obj:`torch.utils.data.Dataset`

   An abstract class representing a :class:`Dataset`.

   All datasets that represent a map from keys to data samples should subclass
   it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
   data sample for a given key. Subclasses could also optionally overwrite
   :meth:`__len__`, which is expected to return the size of the dataset by many
   :class:`~torch.utils.data.Sampler` implementations and the default options
   of :class:`~torch.utils.data.DataLoader`.

   .. note::
     :class:`~torch.utils.data.DataLoader` by default constructs a index
     sampler that yields integral indices.  To make it work with a map-style
     dataset with non-integral indices/keys, a custom sampler must be provided.

   .. py:method:: save_parents(num_req_p = 10, sleep_time = 0.001, use_parhugin = True, par_split = '#', loc_split = '-', overwrite = False)

      Save parent patches for all patches in the patchframe.

      Parameters
      ----------
      num_req_p : int, optional
          The number of required processors for the job, by default 10.
      sleep_time : float, optional
          The time to wait between jobs, by default 0.001.
      use_parhugin : bool, optional
          Flag indicating whether to use Parhugin to parallelize the job, by
          default True.
      par_split : str, optional
          The string used to separate parent IDs in the patch filename, by
          default "#".
      loc_split : str, optional
          The string used to separate patch location and level in the patch
          filename, by default "-".
      overwrite : bool, optional
          Flag indicating whether to overwrite existing parent files, by
          default False.

      Returns
      -------
      None

      Notes
      -----
      Parhugin is a Python package for parallelizing computations across
      multiple CPU cores. The method uses Parhugin to parallelize the
      computation of saving parent patches to disk. When Parhugin is
      installed and ``use_parhugin`` is set to True, the method parallelizes
      the calling of the ``save_parents_idx`` method and its corresponding
      arguments. If Parhugin is not installed or ``use_parhugin`` is set to
      False, the method executes the loop over patch indices sequentially
      instead.


   .. py:method:: save_parents_idx(idx, par_split = '#', loc_split = '-', overwrite = False, return_image = False)

      Save the parents of a specific patch to the specified location.

      Parameters
      ----------
          idx : int
              Index of the patch in the dataset.
          par_split : str, optional
              Delimiter to split the parent names in the file path. Default
              is "#".
          loc_split : str, optional
              Delimiter to split the location of the patch in the file path.
              Default is "-".
          overwrite : bool, optional
              Whether to overwrite the existing parent files. Default is
              False.

      Raises
      ------
      ValueError
          If the patch is not found in the dataset.

      Returns
      -------
      None


   .. py:method:: return_orig_image(idx)

      Return the original image associated with the given index.

      Parameters
      ----------
      idx : int or Tensor
          The index of the desired image, or a Tensor containing the index.

      Returns
      -------
      PIL.Image.Image
          The original image associated with the given index.

      Notes
      -----
      This method returns the original image associated with the given index
      by loading the image file using the file path stored in the
      ``input_col`` column of the ``patchframe`` DataFrame at the given
      index. The loaded image is then converted to the format specified by
      the ``convert2`` attribute of the object. The resulting
      ``PIL.Image.Image`` object is returned.


   .. py:method:: plot_sample(indx)

      Plot a sample patch and its corresponding context from the dataset.

      Parameters
      ----------
      indx : int
          The index of the sample to plot.

      Returns
      -------
      None
          Displays the plot of the sample patch and its corresponding
          context.

      Notes
      -----
      This method plots a sample patch and its corresponding context side-by-
      side in a single figure with two subplots. The figure size is set to
      10in x 5in, and the titles of the subplots are set to "Patch" and
      "Context", respectively. The resulting figure is displayed using
      the ``matplotlib`` library (required).




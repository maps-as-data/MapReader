mapreader.classify.datasets
===========================

.. py:module:: mapreader.classify.datasets


Attributes
----------

.. autoapisummary::

   mapreader.classify.datasets.parhugin_installed


Classes
-------

.. autoapisummary::

   mapreader.classify.datasets.PatchDataset
   mapreader.classify.datasets.PatchContextDataset


Module Contents
---------------

.. py:data:: parhugin_installed
   :value: True


.. py:class:: PatchDataset(patch_df, transform, delimiter = ',', patch_paths_col = 'image_path', label_col = None, label_index_col = None, image_mode = 'RGB')

   Bases: :py:obj:`torch.utils.data.Dataset`


   An abstract class representing a :class:`Dataset`.

   All datasets that represent a map from keys to data samples should subclass
   it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
   data sample for a given key. Subclasses could also optionally overwrite
   :meth:`__len__`, which is expected to return the size of the dataset by many
   :class:`~torch.utils.data.Sampler` implementations and the default options
   of :class:`~torch.utils.data.DataLoader`. Subclasses could also
   optionally implement :meth:`__getitems__`, for speedup batched samples
   loading. This method accepts list of indices of samples of batch and returns
   list of samples.

   .. note::
     :class:`~torch.utils.data.DataLoader` by default constructs an index
     sampler that yields integral indices.  To make it work with a map-style
     dataset with non-integral indices/keys, a custom sampler must be provided.


   .. py:method:: return_orig_image(idx)

      Return the original image associated with the given index.

      :param idx: The index of the desired image, or a Tensor containing the index.
      :type idx: int or Tensor

      :returns: The original image associated with the given index.
      :rtype: PIL.Image.Image

      .. rubric:: Notes

      This method returns the original image associated with the given index
      by loading the image file using the file path stored in the
      ``patch_paths_col`` column of the ``patch_df`` DataFrame at the given
      index. The loaded image is then converted to the format specified by
      the ``image_mode`` attribute of the object. The resulting
      ``PIL.Image.Image`` object is returned.



   .. py:method:: create_dataloaders(set_name = 'infer', batch_size = 16, shuffle = False, num_workers = 0, **kwargs)

      Creates a dictionary containing a PyTorch dataloader.

      :param set_name: The name to use for the dataloader.
      :type set_name: str, optional
      :param batch_size: The batch size to use for the dataloader. By default ``16``.
      :type batch_size: int, optional
      :param shuffle: Whether to shuffle the PatchDataset, by default False
      :type shuffle: bool, optional
      :param num_workers: The number of worker threads to use for loading data. By default ``0``.
      :type num_workers: int, optional
      :param \*\*kwargs: Additional keyword arguments to pass to PyTorch's ``DataLoader`` constructor.

      :returns: Dictionary containing dataloaders.
      :rtype: Dict



.. py:class:: PatchContextDataset(patch_df, total_df, transform, delimiter = ',', patch_paths_col = 'image_path', label_col = None, label_index_col = None, image_mode = 'RGB', context_dir = './maps/maps_context', create_context = False, parent_path = './maps')

   Bases: :py:obj:`PatchDataset`


   An abstract class representing a :class:`Dataset`.

   All datasets that represent a map from keys to data samples should subclass
   it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
   data sample for a given key. Subclasses could also optionally overwrite
   :meth:`__len__`, which is expected to return the size of the dataset by many
   :class:`~torch.utils.data.Sampler` implementations and the default options
   of :class:`~torch.utils.data.DataLoader`. Subclasses could also
   optionally implement :meth:`__getitems__`, for speedup batched samples
   loading. This method accepts list of indices of samples of batch and returns
   list of samples.

   .. note::
     :class:`~torch.utils.data.DataLoader` by default constructs an index
     sampler that yields integral indices.  To make it work with a map-style
     dataset with non-integral indices/keys, a custom sampler must be provided.


   .. py:method:: save_context(processors = 10, sleep_time = 0.001, use_parhugin = True, overwrite = False)

      Save context images for all patches in the patch_df.

      :param processors: The number of required processors for the job, by default 10.
      :type processors: int, optional
      :param sleep_time: The time to wait between jobs, by default 0.001.
      :type sleep_time: float, optional
      :param use_parhugin: Whether to use Parhugin to parallelize the job, by default True.
      :type use_parhugin: bool, optional
      :param overwrite: Whether to overwrite existing parent files, by default False.
      :type overwrite: bool, optional

      :rtype: None

      .. rubric:: Notes

      Parhugin is a Python package for parallelizing computations across
      multiple CPU cores. The method uses Parhugin to parallelize the
      computation of saving parent patches to disk. When Parhugin is
      installed and ``use_parhugin`` is set to True, the method parallelizes
      the calling of the ``get_context_id`` method and its corresponding
      arguments. If Parhugin is not installed or ``use_parhugin`` is set to
      False, the method executes the loop over patch indices sequentially
      instead.



   .. py:method:: get_context_id(id, overwrite = False, save_context = False, return_image = True)

      Save the parents of a specific patch to the specified location.

      :param id: Index of the patch in the dataset.
      :param overwrite: Whether to overwrite the existing parent files. Default is
                        False.
      :type overwrite: bool, optional
      :param save_context: Whether to save the context image. Default is False.
      :type save_context: bool, optional
      :param return_image: Whether to return the context image. Default is True.
      :type return_image: bool, optional

      :raises ValueError: If the patch is not found in the dataset.

      :rtype: None



   .. py:method:: plot_sample(idx)

      Plot a sample patch and its corresponding context from the dataset.

      :param idx: The index of the sample to plot.
      :type idx: int

      :returns: Displays the plot of the sample patch and its corresponding
                context.
      :rtype: None

      .. rubric:: Notes

      This method plots a sample patch and its corresponding context side-by-
      side in a single figure with two subplots. The figure size is set to
      10in x 5in, and the titles of the subplots are set to "Patch" and
      "Context", respectively. The resulting figure is displayed using
      the ``matplotlib`` library (required).

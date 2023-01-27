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

   

.. py:class:: patchTorchDataset(patchframe, transform=None, label_col='label', convert2='RGB', input_col=0)

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



.. py:class:: patchContextDataset(patchframe, transform1=None, transform2=None, label_col='label', convert2='RGB', input_col=0, context_save_path='./maps/maps_context', create_context=False, par_path='./maps', x_offset=1.0, y_offset=1.0, slice_method='scale')

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

   .. py:method:: save_parents(num_req_p=10, sleep_time=0.001, use_parhugin=True, par_split='#', loc_split='-', overwrite=False)


   .. py:method:: save_parents_idx(idx, par_split='#', loc_split='-', overwrite=False, return_image=False)


   .. py:method:: return_orig_image(idx)


   .. py:method:: plot_sample(indx)




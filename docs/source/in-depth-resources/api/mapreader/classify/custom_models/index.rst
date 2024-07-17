mapreader.classify.custom_models
================================

.. py:module:: mapreader.classify.custom_models


Classes
-------

.. autoapisummary::

   mapreader.classify.custom_models.twoParallelModels


Module Contents
---------------

.. py:class:: twoParallelModels(patch_model, context_model, fc_layer)

   Bases: :py:obj:`torch.nn.Module`


   A class for building a model that contains two parallel branches, with
   separate input pipelines, but shares a fully connected layer at the end.
   This class inherits from PyTorch's nn.Module.


   .. py:method:: forward(x1, x2)

      Defines the computation performed at every forward pass. Receives two
      inputs, x1 and x2, and feeds them through the respective feature
      extractor modules, then concatenates the output and passes it through
      the fully connected layer.

      Parameters:
      -----------
      x1 : torch.Tensor
          The input tensor for the patch only pipeline.
      x2 : torch.Tensor
          The input tensor for the context pipeline.

      Returns:
      --------
      torch.Tensor
          The output tensor of the model.

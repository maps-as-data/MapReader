:py:mod:`mapreader.train.custom_models`
=======================================

.. py:module:: mapreader.train.custom_models


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   mapreader.train.custom_models.twoParallelModels




.. py:class:: twoParallelModels(feature1, feature2, fc_layer)

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
          The input tensor for the first input pipeline.
      x2 : torch.Tensor
          The input tensor for the second input pipeline.

      Returns:
      --------
      torch.Tensor
          The output tensor of the model.




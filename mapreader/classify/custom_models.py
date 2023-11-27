#!/usr/bin/env python
from __future__ import annotations

import torch


class twoParallelModels(torch.nn.Module):
    """
    A class for building a model that contains two parallel branches, with
    separate input pipelines, but shares a fully connected layer at the end.
    This class inherits from PyTorch's nn.Module.
    """

    def __init__(
        self,
        patch_model: torch.nn.Module,
        context_model: torch.nn.Module,
        fc_layer: torch.nn.Linear,
    ):
        """
        Initializes a new instance of the twoParallelModels class.

        Parameters:
        -----------
        patch_model : nn.Module
            The feature extractor module for the first patch only pipeline.
        context_model : nn.Module
            The feature extractor module for the second context pipeline.
        fc_layer : nn.Linear
            The fully connected layer at the end of the model.
            Input size should be output size of patch_model + output size of context_model.
            Output size should be number of classes (labels).
        """
        super().__init__()
        self.patch_model = patch_model
        self.context_model = context_model
        self.fc_layer = fc_layer

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
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
        """

        x1 = self.patch_model(x1)
        x1 = x1.view(x1.size(0), -1)

        x2 = self.context_model(x2)
        x2 = x2.view(x2.size(0), -1)

        # Concatenate in dim1 (feature dimension)
        x = torch.cat((x1, x2), 1)
        x = self.fc_layer(x)
        return x

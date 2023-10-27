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
        feature1: torch.nn.Module,
        feature2: torch.nn.Module,
        fc_layer: torch.nn.Linear,
    ):
        """
        Initializes a new instance of the twoParallelModels class.

        Parameters:
        -----------
        feature1 : nn.Module
            The feature extractor module for the first input pipeline.
        feature2 : nn.Module
            The feature extractor module for the second input pipeline.
        fc_layer : nn.Linear
            The fully connected layer at the end of the model.
        """
        super().__init__()
        self.features1 = feature1
        self.features2 = feature2
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
            The input tensor for the first input pipeline.
        x2 : torch.Tensor
            The input tensor for the second input pipeline.

        Returns:
        --------
        torch.Tensor
            The output tensor of the model.
        """

        x1 = self.features1(x1)
        x1 = x1.view(x1.size(0), -1)

        x2 = self.features2(x2)
        x2 = x2.view(x2.size(0), -1)

        # Concatenate in dim1 (feature dimension)
        x = torch.cat((x1, x2), 1)
        x = self.fc_layer(x)
        return x

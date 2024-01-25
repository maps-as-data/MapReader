#!/usr/bin/env python
from __future__ import annotations

import copy

import torch


class PatchContextModel(torch.nn.Module):
    """
    Model that contains two parallel branches, with separate input pipelines, but one shared fully connected layer at the end.
    This class inherits from PyTorch's nn.Module.
    """

    def __init__(
        self,
        patch_model: torch.nn.Module,
        context_model: torch.nn.Module,
        fc_layer: torch.nn.Linear,
    ):
        """
        Initializes a new instance of the PatchContextModel class.

        Parameters:
        -----------
        patch_model : nn.Module
            The feature extractor module for the first patch only pipeline.
        context_model : nn.Module
            The feature extractor module for the second context pipeline.
        fc_layer : nn.Linear
            The fully connected layer at the end of the model.
            Input size should be output size of patch_model + output size of context_model.
            Output size should be number of classes (labels) at the patch level.
        """
        super().__init__()

        if patch_model is context_model:
            context_model = copy.deepcopy(context_model)

        self.patch_model = patch_model
        self.context_model = context_model
        self.fc_layer = fc_layer

    def forward(self, patch: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Defines the computation performed at every forward pass.
        Receives two inputs, patch and context, and feeds them through the respective feature extractor modules, then concatenates the output and passes it through
        the fully connected layer.

        Parameters:
        -----------
        patch : torch.Tensor
            The input tensor for the patch pipeline.
        context : torch.Tensor
            The input tensor for the context pipeline.

        Returns:
        --------
        torch.Tensor
            The output tensor of the model.
        """

        patch_output = self.patch_model(patch)
        context_output = self.context_model(context)

        # Concatenate in dim1 (feature dimension)
        out = torch.cat((patch_output, context_output), 1)
        out = self.fc_layer(out)
        return (patch_output, context_output), out

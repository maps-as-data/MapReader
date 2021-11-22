#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn
import torch


class twoParallelModels(nn.Module):
    def __init__(self, feature1, feature2, fc_layer):
        super(twoParallelModels, self).__init__()
        self.features1 = feature1
        self.features2 = feature2
        self.fc_layer = fc_layer

    def forward(self, x1, x2):
        x1 = self.features1(x1)
        x1 = x1.view(x1.size(0), -1)

        x2 = self.features2(x2)
        x2 = x2.view(x2.size(0), -1)

        # Concatenate in dim1 (feature dimension)
        x = torch.cat((x1, x2), 1)
        x = self.fc_layer(x)
        return x

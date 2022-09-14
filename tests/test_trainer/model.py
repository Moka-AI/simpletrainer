from __future__ import annotations

import torch
import torch.nn as nn


class ForTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.criterion = nn.MSELoss()

    def forward(self, x, y=None):
        pred = x * self.scale
        output = {'pred': pred}
        if y is not None:
            loss = self.criterion(pred, y)
            output['loss'] = loss
        return output

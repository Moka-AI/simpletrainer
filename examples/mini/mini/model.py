import time
from typing import Optional

import torch
import torch.nn as nn
from attrs import define


@define
class MiniModelOutput:
    predictions: torch.Tensor
    loss: Optional[torch.Tensor] = None


class MiniModel(nn.Module):
    def __init__(self, num_args: int):
        super().__init__()
        self.linear = nn.Linear(num_args, 1)
        self.criterion = nn.MSELoss()

    def forward(self, x, y) -> MiniModelOutput:
        y_hat = self.linear(x).squeeze()
        output = MiniModelOutput(predictions=y_hat)
        time.sleep(0.002)
        if y is not None:
            loss = self.criterion(y_hat, y)
            output.loss = loss
        return output

import torch
from torch.utils.data import IterableDataset


class MiniDataset(IterableDataset):
    def __init__(self, n=200000) -> None:
        self.n = n
        self.x = torch.randn(n, 3)
        self.y = torch.mm(self.x, torch.tensor([1.0, 2.0, 3.0]).view(-1, 1)).squeeze() + torch.randn(1)

    def __iter__(self):
        for i in range(self.n):
            yield {'x': self.x[i, :], 'y': self.y[i]}

    def __len__(self):
        return self.n

from torch.utils.data import Dataset


class ForTestDataset(Dataset):
    def __init__(self) -> None:
        self.x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    def __getitem__(self, index: int):
        return {
            'x': float(self.x[index]),
            'y': float(self.y[index]),
        }

    def __len__(self):
        return 10

import pytest
import torch
from accelerate import Accelerator
from accelerate.data_loader import prepare_data_loader
from torch.utils.data import DataLoader, IterableDataset

from simpletrainer.utils.torch import get_data_info


class NoSizedIterableDataset(IterableDataset):
    def __init__(self, max_number: int) -> None:
        self.max_number = max_number

    def __iter__(self):
        for i in range(self.max_number):
            yield i


@pytest.mark.parametrize(
    ('dataset', 'drop_last', 'num_samples', 'num_batches_per_epoch'),
    [
        (NoSizedIterableDataset(10), False, None, None),
        (NoSizedIterableDataset(10), True, None, None),
        (range(10), True, 10, 3),
        (range(10), False, 10, 4),
        (list(range(10)), True, 10, 3),
        (list(range(10)), False, 10, 4),
    ],
)
def test_single_process_get_data_info(dataset, drop_last, num_samples, num_batches_per_epoch):
    Accelerator()
    dataloader = DataLoader(dataset, batch_size=3, drop_last=drop_last)

    single_process_dataloader = prepare_data_loader(
        dataloader,
        device=torch.device('cpu'),
        num_processes=1,
        process_index=0,
    )
    data_info = get_data_info(single_process_dataloader)
    assert data_info.batch_size == 3
    assert data_info.batch_size_per_device == 3
    assert data_info.num_sampels == num_samples
    assert data_info.num_batches_per_epoch == num_batches_per_epoch


@pytest.mark.parametrize(
    ('dataset', 'drop_last', 'num_samples', 'num_batches_per_epoch'),
    [
        (NoSizedIterableDataset(10), False, None, None),
        (NoSizedIterableDataset(10), True, None, None),
        (range(10), True, 10, 1),
        (range(10), False, 10, 2),
        (list(range(10)), True, 10, 1),
        (list(range(10)), False, 10, 2),
    ],
)
def test_multi_process_get_data_info(dataset, drop_last, num_samples, num_batches_per_epoch):
    Accelerator()
    dataloader = DataLoader(dataset, batch_size=3, drop_last=drop_last)

    multi_process_dataloader = prepare_data_loader(
        dataloader,
        device=torch.device('cpu'),
        num_processes=2,
        process_index=0,
    )
    data_info = get_data_info(multi_process_dataloader, world_size=2)
    assert data_info.batch_size == 6
    assert data_info.batch_size_per_device == 3
    assert data_info.num_sampels == num_samples
    assert data_info.num_batches_per_epoch == num_batches_per_epoch

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from simpletrainer import Trainer, TrainerConfig
from simpletrainer.utils.torch import set_seed

from .dataset import ForTestDataset
from .model import ForTestModel


@pytest.mark.parametrize(
    ('batch_size', 'epochs', 'accumulate_grad_batches'),
    [
        (2, 1, 1),
        (1, 1, 2),
        (2, 2, 1),
    ],
)
def test_train(tmpdir, batch_size, epochs, accumulate_grad_batches):
    set_seed(0)
    scale, loss = calulate_scale_and_loss(batch_size * accumulate_grad_batches, epochs)
    dataset = ForTestDataset()
    train_dataloader = DataLoader(dataset, batch_size=batch_size)
    valid_dataloader = DataLoader(dataset, batch_size=2)
    model = ForTestModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    config = TrainerConfig(
        epochs=epochs,
        experiment_name='for_test',
        accumulate_grad_batches=accumulate_grad_batches,
        output_dir=tmpdir,
    )
    config.accelerator.cpu = True
    trainer = Trainer.from_config(config)
    trainer.train(
        model,
        optimizer,
        train_dataloader,
        valid_dataloader,
    )
    assert round(model.scale.item(), 4) == round(scale, 4)
    assert round(trainer.latest_valid_metrics['loss'], 4) == round(loss, 4)  # type: ignore


def calulate_scale_and_loss(batch_size: int, epochs: int) -> tuple[float, float]:
    set_seed(0)
    dataset = ForTestDataset()
    train_dataloader = DataLoader(dataset, batch_size=batch_size)
    valid_dataloader = DataLoader(dataset, batch_size=batch_size)
    model = ForTestModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(epochs):
        for batch in train_dataloader:
            loss = model(**batch)['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    loss_list = []
    for batch in valid_dataloader:
        loss = model(**batch)['loss']
        loss_list.append(loss.item())
    return model.scale.item(), sum(loss_list) / len(loss_list)

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Optional, Sized, TypeVar

import torch
import torchinfo
from accelerate.utils.random import set_seed
from torch.utils.data import DataLoader

from simpletrainer.utils.common import pretty_str

T = TypeVar('T')
set_seed = set_seed


@dataclass
class DataInfo:
    batch_size: int
    batch_size_per_device: int
    num_sampels: Optional[int]
    num_batches_per_epoch: Optional[int]

    def __repr__(self) -> str:
        return pretty_str(asdict(self), self.__class__.__name__)


def get_batch_size_from_dataloader(dataloader: DataLoader) -> int:
    if dataloader.batch_size is None:
        try:
            return dataloader.batch_sampler.batch_size  # type: ignore
        except AttributeError:
            raise ValueError(
                'Can not get batch size from dataloader, does not support `BatchSampler` with varying batch size yet.'
            )
    else:
        return dataloader.batch_size


def get_num_samples_from_dataloader(dataloader: DataLoader) -> Optional[int]:
    if isinstance(dataloader.dataset, Sized):
        return len(dataloader.dataset)
    elif isinstance(dataloader.sampler, Sized):
        return len(dataloader.sampler)
    else:
        sampler = getattr(dataloader.batch_sampler, 'sampler')
        if isinstance(sampler, Sized):
            return len(sampler)
        else:
            return


def get_data_info(dataloader: DataLoader, world_size: int = 1) -> DataInfo:
    num_samples = get_num_samples_from_dataloader(dataloader)

    try:
        num_batches_per_epoch = len(dataloader)
    except:
        num_batches_per_epoch = None

    batch_size_per_device = get_batch_size_from_dataloader(dataloader)
    batch_size = batch_size_per_device * world_size

    return DataInfo(
        batch_size=batch_size,
        batch_size_per_device=batch_size_per_device,
        num_sampels=num_samples,
        num_batches_per_epoch=num_batches_per_epoch,
    )


def get_model_info(
    model: torch.nn.Module,
    input_data: Optional[torchinfo.torchinfo.INPUT_DATA_TYPE] = None,
    device: Optional[torch.device] = None,
) -> torchinfo.ModelStatistics:
    try:
        model_statistics = torchinfo.summary(model, input_data=input_data, verbose=0, device=device)
    except Exception:
        model_statistics = torchinfo.summary(model, verbose=0, device=device)
    return model_statistics


def get_parameter_id_group_map(
    optimizer: torch.optim.Optimizer,
) -> dict[int, str]:
    parameter_id_group_map = {}
    for group, params in enumerate(optimizer.param_groups):
        for param in params['params']:
            parameter_id_group_map[id(param)] = str(group)
    return parameter_id_group_map


def get_params_with_pattern(model: torch.nn.Module, pattern: re.Pattern):
    params = []
    for name, param in model.named_parameters():
        if pattern.search(name):
            params.append(param)
    return params


def get_module_learning_rate_summary(module: torch.nn.Module, optimizer: torch.optim.Optimizer):
    lr_dict: dict[str, float] = {}
    names = {param: name for name, param in module.named_parameters()}
    for group in optimizer.param_groups:
        if 'lr' not in group:
            continue
        lr = group['lr']
        for param in group['params']:
            if param.requires_grad:
                lr_dict[names[param]] = lr
            else:
                lr_dict[names[param]] = 0.0
    return lr_dict


def get_module_parameter_summary(model: torch.nn.Module):
    parameter_mean: dict[str, float] = {}
    parameter_std: dict[str, float] = {}

    for name, param in model.named_parameters():
        if param.data.numel() > 0:
            parameter_mean[name] = float(param.data.mean().item())
        if param.data.numel() > 1:
            parameter_std[name] = float(param.data.std().item())
    return parameter_mean, parameter_std


def get_module_gradient_summary(model: torch.nn.Module):
    gradient_mean: dict[str, float] = {}
    gradient_std: dict[str, float] = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            if param.grad.is_sparse:
                grad_data = param.grad.data._values()
            else:
                grad_data = param.grad.data

            # skip empty gradients
            if torch.prod(torch.tensor(grad_data.shape)).item() > 0:
                gradient_mean[name] = float(grad_data.mean().item())
                if grad_data.numel() > 1:
                    gradient_std[name] = float(grad_data.std().item())
    return gradient_mean, gradient_std

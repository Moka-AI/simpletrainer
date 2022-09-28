from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from attrs import asdict, define

from simpletrainer.common.types import PathOrStr

if TYPE_CHECKING:
    from simpletrainer.core.trainer import Trainer


@define
class TrainerInitAttributes:
    epochs: int
    do_eval: bool
    accumulate_grad_batches: int
    core_metric: str
    experiment_name: str
    run_name: str
    output_dir: str
    auto_restore: bool


@define
class AcceleratorInfo:
    mixed_precision: str
    num_processes: int
    device: str
    distributed_type: str


@define
class TrainerInfo:
    init_attributes: TrainerInitAttributes
    accelerator: AcceleratorInfo
    logger: str
    components: list[str]
    hyper_params: dict

    train_metrics: list[dict]
    valid_metrics: list[dict]

    model: str
    optimizer: str
    lr_scheduler: str

    @classmethod
    def from_trainer(cls, trainer: 'Trainer'):
        if hasattr(trainer, 'model'):
            model = trainer.raw_model.__class__.__name__
        else:
            model = ''

        if hasattr(trainer, 'optimizer'):
            optimizer = trainer.raw_optimizer.__class__.__name__
        else:
            optimizer = ''

        if hasattr(trainer, 'lr_scheduler'):
            lr_scheduler = trainer.lr_scheduler.__class__.__name__
        else:
            lr_scheduler = ''

        init_attributes = TrainerInitAttributes(
            epochs=trainer.epochs,
            do_eval=trainer.do_eval,
            accumulate_grad_batches=trainer.accumulate_grad_batches,
            core_metric=str(trainer.core_metric),
            experiment_name=trainer.experiment_name,
            run_name=trainer.run_name,
            output_dir=str(trainer.output_dir),
            auto_restore=trainer.auto_restore,
        )

        acclerator = AcceleratorInfo(
            mixed_precision=str(trainer.accelerator.state.mixed_precision),
            num_processes=trainer.accelerator.state.num_processes,
            device=str(trainer.accelerator.state.device),
            distributed_type=str(trainer.accelerator.state.distributed_type),
        )

        return cls(
            init_attributes=init_attributes,
            accelerator=acclerator,
            logger=trainer.logger.__class__.__name__,
            components=[i.__class__.__name__ for i in trainer._components],
            hyper_params=trainer.hyper_params,
            train_metrics=trainer.state.train_metrics_history,
            valid_metrics=trainer.state.valid_metrics_history,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

    def to_json(self, json_file: PathOrStr) -> None:
        Path(json_file).write_text(json.dumps(asdict(self)))

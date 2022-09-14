from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Union, cast

import torch
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from torch.utils.data import DataLoader

from simpletrainer.common.default_settings import DefaultSettings
from simpletrainer.common.types import (
    Batch,
    BatchOutput,
    HyperParams,
    MetricDict,
    Prime,
    SignMetric,
)
from simpletrainer.core.configs import TrainerConfig
from simpletrainer.core.hook import TrainerEvent, TrainerHookEngine, entrypoint
from simpletrainer.core.mixins import AcceleratorMixin, TrainerStateMixin
from simpletrainer.core.states import TrainerLoopState, TrainerStageState
from simpletrainer.loggers import BaseDeepLearningLogger, TensorboardLogger
from simpletrainer.utils.common import random_experiment_name, smartget
from simpletrainer.utils.torch import get_data_info

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from simpletrainer.components.base import Component


class Trainer(TrainerStateMixin, AcceleratorMixin):
    EVENT = TrainerEvent

    def __init__(
        self,
        epochs: int = DefaultSettings.epochs,
        do_eval: bool = True,
        accumulate_grad_batches: int = 1,
        core_metric: SignMetric = SignMetric('-', 'loss'),
        experiment_name: str = DefaultSettings.experiment_name,
        run_name: Optional[str] = None,
        output_dir: Optional[Path] = None,
        accelerator: Optional[Accelerator] = None,
        logger: Optional[BaseDeepLearningLogger] = None,
        components: Sequence['Component'] = tuple(),
    ) -> None:
        self.epochs = epochs
        self.do_eval = do_eval
        self.accumulate_grad_batches = accumulate_grad_batches
        self.core_metric = core_metric
        self.experiment_name = experiment_name
        self.run_name = run_name or random_experiment_name()
        self.output_dir = output_dir or (Path(self.experiment_name) / self.run_name)
        self.accelerator = accelerator or Accelerator()
        self.logger = logger or TensorboardLogger()
        self.components = []
        self.exports: dict[str, Any] = {}

        self.hook_engine = TrainerHookEngine(self)
        components = list(components) + self._get_default_components()
        for component in components:
            self.hook_engine.add(component)
        self.hook_engine.on(TrainerEvent.INIT)

    @classmethod
    def from_config(cls, config: TrainerConfig) -> 'Trainer':
        from simpletrainer.common.registry import DeepLearningLoggerRegistry
        from simpletrainer.components import (
            RichInspect,
            RichProgressBar,
            TqdmProgressBar,
        )

        accelerator = config.accelerator.build()
        logger = DeepLearningLoggerRegistry[config.logger]()

        components = []
        if config.inspect:
            components.append(RichInspect())
        if config.progress_bar == 'rich':
            components.append(RichProgressBar())
        elif config.progress_bar == 'tqdm':
            components.append(TqdmProgressBar())

        return cls(
            accelerator=accelerator,
            logger=logger,
            epochs=config.epochs,
            do_eval=config.do_eval,
            accumulate_grad_batches=config.accumulate_grad_batches,
            core_metric=config.core_metric,
            experiment_name=config.experiment_name,
            run_name=config.run_name,
            output_dir=config.output_dir,
            components=components,
        )

    def train(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        valid_dataloader: Optional[DataLoader] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        try:
            self.prepare()
            self.loop()
            self.hook_engine.on(TrainerEvent.FINISH)
        except (Exception, KeyboardInterrupt) as e:
            self.exception = e
            self.hook_engine.on(TrainerEvent.CRASH)
            raise
        finally:
            self.teardown()

    def prepare(self) -> None:
        if self.valid_dataloader is None:
            self.do_eval = False

        self.train_data_info = get_data_info(self.train_dataloader, self.accelerator.num_processes)
        if self.valid_dataloader is None:
            self.valid_data_info = None
        else:
            self.valid_data_info = get_data_info(self.valid_dataloader, self.accelerator.num_processes)

        self._accelerate_prepare()
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.logger.with_trainer(self)
        self._loop_state = TrainerLoopState()
        self.hook_engine.on(TrainerEvent.PREPARE)

    def teardown(self) -> None:
        self.logger.teardown()
        self.hook_engine.on(TrainerEvent.TEARDOWN)

    @entrypoint
    def loop(self) -> None:
        while not self.should_stop_loop():
            self.run_epoch()
            self.advance_epoch()

    @entrypoint
    def should_stop_loop(self) -> bool:
        return self.current_epoch > self.epochs

    @entrypoint
    def run_epoch(self) -> None:
        logger.info(f'[epoch {self.current_epoch}] train stage start')
        self.prepare_for_stage(train=True)
        self.run_stage(train=True)

        if self.do_eval:
            logger.info(f'[epoch {self.current_epoch}] valid stage start')
            self.prepare_for_stage(train=False)
            self.run_stage(train=False)

    @entrypoint
    def run_stage(self, train: bool) -> None:
        if train:
            self.run_train_stage()
        else:
            self.run_valid_stage()

    @entrypoint
    def prepare_for_stage(self, train: bool) -> None:
        self.model.train(train)
        self.valid_dataloader = cast(DataLoader, self.valid_dataloader)
        data_iterator = iter(self.train_dataloader) if train else iter(self.valid_dataloader)
        self._stage_state = TrainerStageState(
            data_iterator,
            train=train,
            current_epoch=self.current_epoch,
            loss_metric_scale=self.accumulate_grad_batches,
        )

    @entrypoint
    def run_train_stage(self) -> None:
        while batch := self.generate_batch():
            self.set_batch(batch)
            self.run_batch()

            if self.should_step():
                self.step()

        stage_metrics = self.collect_stage_metrics()
        stage_metrics = {k: round(v, DefaultSettings.metric_round) for k, v in stage_metrics.items()}
        logger.info(f'[epoch {self.current_epoch}] train stage metrics: {stage_metrics}')
        self.train_metrics_history.append(stage_metrics)

    @entrypoint
    def generate_batch(self) -> Optional[Batch]:
        if self.stage_state.batch_iterator is None:
            raise ValueError('Batch iterator is not initialized')
        try:
            return next(self.stage_state.batch_iterator)
        except StopIteration:
            return

    @entrypoint
    def set_batch(self, batch: Batch) -> None:
        self.stage_state.batch = batch

    @entrypoint
    def run_valid_stage(self) -> None:
        while batch := self.generate_batch():
            self.set_batch(batch)
            self.run_batch()

        stage_metrics = self.collect_stage_metrics()
        stage_metrics = {k: round(v, DefaultSettings.metric_round) for k, v in stage_metrics.items()}
        logger.info(f'[epoch {self.current_epoch}] valid stage metrics: {stage_metrics}')
        self.valid_metrics_history.append(stage_metrics)

    @entrypoint
    def collect_stage_metrics(self) -> MetricDict:
        return self.stage_state.metrics

    @entrypoint
    def should_step(self) -> bool:
        return (self.stage_state.num_batches % self.accumulate_grad_batches) == 0

    @entrypoint
    def run_batch(self) -> None:
        self.advance_batch()
        if self.in_train_stage:
            self.run_train_batch()
        else:
            self.run_valid_batch()

    @entrypoint
    def run_train_batch(self) -> None:
        self.forward()
        self.backward()

    @entrypoint
    def run_valid_batch(self) -> None:
        with torch.no_grad():
            self.forward()

    @entrypoint
    def forward(self) -> None:
        batch_output: BatchOutput = self.model(**self.stage_state.batch)
        self.set_batch_output(batch_output)
        self.set_loss()

    @entrypoint
    def backward(self) -> None:
        self.accelerator.backward(self.stage_state.loss)

    @entrypoint
    def set_batch_output(self, batch_output: BatchOutput) -> None:
        self.stage_state.batch_output = batch_output

    @entrypoint
    def set_loss(self) -> None:
        loss = smartget(self.stage_state.batch_output, 'loss') / self.accumulate_grad_batches
        self.stage_state.loss = loss

    @entrypoint
    def step(self) -> None:
        self.advance_step()
        self.optimizer.step()
        self.optimizer.zero_grad()

    @property
    def hyper_params(self) -> HyperParams:
        hyper_params: HyperParams = {
            'epochs': self.epochs,
            'mixed_precision': self.mixed_precision,
            'accumulate_grad_batches': self.accumulate_grad_batches,
        }
        for component in self.components:
            hyper_params.update(**component.hyper_params)
        return hyper_params

    @property
    def num_steps_per_epoch(self) -> Optional[int]:
        if self.train_data_info.num_batches_per_epoch is None:
            return
        return self.train_data_info.num_batches_per_epoch // self.accumulate_grad_batches

    @property
    def total_steps(self):
        if self.num_steps_per_epoch is None:
            return
        return self.num_steps_per_epoch * self.epochs

    @property
    def raw_model(self) -> torch.nn.Module:
        if isinstance(self.model, torch.nn.parallel.distributed.DistributedDataParallel):
            return self.model.module
        elif isinstance(self.model, torch.nn.parallel.DataParallel):
            return self.model.module
        else:
            return self.model

    @property
    def raw_optimizer(self) -> torch.optim.Optimizer:
        if isinstance(self.optimizer, AcceleratedOptimizer):
            return self.optimizer.optimizer  # type: ignore
        else:
            return self.optimizer

    @property
    def attributes(self) -> dict[str, Prime]:
        return {
            'epochs': self.epochs,
            'do_eval': self.do_eval,
            'accumulate_grad_batches': self.accumulate_grad_batches,
            'core_metric': str(self.core_metric),
            'experiment_name': self.experiment_name,
            'run_name': self.run_name,
            'output_dir': str(self.output_dir),
        }

    def log(
        self, data: Mapping[str, Union[float, torch.Tensor]], step: Optional[int] = None, prefix: Optional[str] = None
    ) -> None:
        if prefix is None:
            prefix = 'train' if self.stage_state.train else 'validation'

        if step is None:
            step = self.cum_steps

        scalars = {k: v for k, v in data.items() if isinstance(v, (int, float))}
        if scalars:
            scalars = self.logger.add_prefix(prefix, scalars)
            self.logger.log_scalars(scalars, step=step)

        tensors = {k: v for k, v in data.items() if isinstance(v, torch.Tensor)}
        if tensors:
            tensors = self.logger.add_prefix(prefix, tensors)
            self.logger.log_tensors(tensors, step=step)

    def _get_default_components(self):
        from simpletrainer.components import FileHandler, SaveTrainerInfo, Timer

        return [Timer(), SaveTrainerInfo(), FileHandler()]

    def _accelerate_prepare(self) -> None:
        self.model: torch.nn.Module = self.accelerator.prepare(self.model)  # type: ignore
        self.optimizer: torch.optim.Optimizer = self.accelerator.prepare(self.optimizer)  # type: ignore
        self.train_dataloader: DataLoader = self.accelerator.prepare(self.train_dataloader)  # type: ignore
        if self.valid_dataloader is not None:
            self.valid_dataloader: Optional[DataLoader] = self.accelerator.prepare(self.valid_dataloader)  # type: ignore

    def __getitem__(self, key: str):
        return self.exports[key]

    def __setitem__(self, key: str, item: Any):
        self.exports[key] = item

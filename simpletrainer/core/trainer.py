from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Union

import torch
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from torch.utils.data import DataLoader

from simpletrainer.common.default_settings import DefaultSettings
from simpletrainer.common.protocols import Stateful
from simpletrainer.common.sign_metric import SignMetric
from simpletrainer.common.types import (
    Batch,
    BatchOutput,
    HyperParams,
    LRScheduler,
    MetricDict,
    PathOrStr,
)
from simpletrainer.core.configs import TrainerConfig
from simpletrainer.core.hook import TrainerEvent, TrainerHookEngine, entrypoint
from simpletrainer.core.info import TrainerInfo
from simpletrainer.core.mixins import AcceleratorMixin, TrainerStateMixin
from simpletrainer.core.state import TrainerState
from simpletrainer.loggers import DeepLearningLogger, DeepLearningLoggerRegistry
from simpletrainer.utils.common import random_experiment_name, smartget
from simpletrainer.utils.torch import get_data_info

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from simpletrainer.core.component import BaseComponent


class Trainer(TrainerStateMixin, AcceleratorMixin):
    EVENT = TrainerEvent

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    train_dataloader: DataLoader
    valid_dataloader: Optional[DataLoader]
    lr_scheduler: Optional[LRScheduler]

    def __init__(
        self,
        epochs: int = DefaultSettings.epochs,
        do_eval: bool = True,
        accumulate_grad_batches: int = 1,
        core_metric: SignMetric = SignMetric.from_str(DefaultSettings.core_metric),
        experiment_name: str = DefaultSettings.experiment_name,
        run_name: Optional[str] = None,
        output_dir: Optional[Path] = None,
        auto_restore: bool = True,
        checkpoint_for_restore: Optional[PathOrStr] = None,
        accelerator: Optional[Accelerator] = None,
        logger: Optional[DeepLearningLogger] = None,
        components: Iterable['BaseComponent'] = tuple(),
    ) -> None:
        self.epochs = epochs
        self.do_eval = do_eval
        self.accumulate_grad_batches = accumulate_grad_batches
        self.core_metric = core_metric
        self.experiment_name = experiment_name
        self.run_name = run_name or random_experiment_name()
        self.output_dir = output_dir or (Path(self.experiment_name) / self.run_name)
        self.auto_restore = auto_restore
        self.checkpoint_for_restore = checkpoint_for_restore
        self.accelerator = accelerator or Accelerator()
        self.logger = logger or DeepLearningLoggerRegistry[DefaultSettings.logger]()
        self._components: list['BaseComponent'] = []
        self.exports: dict[str, Any] = {}
        self.state = TrainerState()

        self.hook_engine = TrainerHookEngine(self)
        self.register_components(components)

    @classmethod
    def from_config(cls, config: TrainerConfig) -> 'Trainer':
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
            auto_restore=config.auto_restore,
            checkpoint_for_restore=config.checkpoint_for_restore,
            components=components,
        )

    def train(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        valid_dataloader: Optional[DataLoader] = None,
        lr_scheduler: Optional[LRScheduler] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.lr_scheduler = lr_scheduler

        self.prepare()
        self.try_restore_checkpoint(self.checkpoint_for_restore)

        try:
            self.hook_engine.on(TrainerEvent.START)
            self.start_loop()
            self.hook_engine.on(TrainerEvent.FINISH)
        except (Exception, KeyboardInterrupt) as e:
            self.exception = e
            self.crash()
            raise
        finally:
            self.teardown()

    def prepare(self) -> None:
        self._accelerate_prepare()

        if self.valid_dataloader is None:
            self.do_eval = False
        self.train_data_info = get_data_info(self.train_dataloader, self.accelerator.num_processes)
        if self.valid_dataloader is None:
            self.valid_data_info = None
        else:
            self.valid_data_info = get_data_info(self.valid_dataloader, self.accelerator.num_processes)

        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.logger.post_init(self)
        for component in self._components:
            component.post_init_with_trainer(self)
        self.info.to_json(self.output_dir / DefaultSettings.trainer_info_json_file_name)

    def try_restore_checkpoint(self, checkpoint: Optional[PathOrStr] = None) -> None:
        if checkpoint is not None:
            self.load(checkpoint)
        elif self.auto_restore:
            if not self.checkpoint_dir.exists():
                return
            latest_checkpoint = max((d for d in self.checkpoint_dir.iterdir() if d.is_dir()), key=lambda x: x.stat().st_mtime)
            self.load(latest_checkpoint)

    def teardown(self) -> None:
        self.logger.teardown()
        self.info.to_json(self.output_dir / DefaultSettings.trainer_info_json_file_name)
        self.hook_engine.on(TrainerEvent.TEARDOWN)

    def crash(self) -> None:
        if isinstance(self.exception, KeyboardInterrupt):
            self.save('on-keyboard-interrupt')
        self.hook_engine.on(TrainerEvent.CRASH)

    @entrypoint
    def start_loop(self) -> None:
        while not self.should_stop_loop():
            self.run_epoch()
            self.advance_epoch()

    @entrypoint
    def should_stop_loop(self) -> bool:
        return self.current_epoch > self.epochs

    @entrypoint
    def run_epoch(self) -> None:
        if self.state.should_restore:
            self.run_restore_epoch()
        else:
            self.prepare_for_stage(train=True)
            self.run_stage(train=True)
            if self.do_eval:
                self.prepare_for_stage(train=False)
                self.run_stage(train=False)

    @entrypoint
    def run_restore_epoch(self) -> None:
        tn = self.run_train_stage.__name__
        vn = self.run_valid_stage.__name__

        if tn not in self.state.checkpoint_entrypoint_stack and vn not in self.state.checkpoint_entrypoint_stack:
            self.state.checkpoint_entrypoint_stack = []
            return

        if tn in self.state.checkpoint_entrypoint_stack:
            self.prepare_for_stage(train=True)
            self.run_stage(train=True)

        if self.do_eval:
            logger.debug(f'[epoch {self.current_epoch}] valid stage start')
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
        batch_iterator = iter(self.train_dataloader) if train else iter(self.valid_dataloader)   # type: ignore
        if self.state.should_restore:
            self.state.restore(train, batch_iterator)
        else:
            self.state.prepare(train, batch_iterator)

    @entrypoint
    def run_train_stage(self) -> None:
        while batch := self.generate_batch():
            self.set_batch(batch)
            self.run_batch()

            if self.should_step():
                self.step()

        stage_metrics = self.collect_stage_metrics()
        stage_metrics = {k: round(v, DefaultSettings.metric_round) for k, v in stage_metrics.items()}
        logger.debug(f'[epoch {self.current_epoch}] train stage metrics: {stage_metrics}')
        self.train_metrics_history.append(stage_metrics)

    @entrypoint
    def generate_batch(self) -> Optional[Batch]:
        if self.state.batch_iterator is None:
            raise ValueError('Batch iterator is not initialized')
        try:
            return next(self.state.batch_iterator)
        except StopIteration:
            return

    @entrypoint
    def set_batch(self, batch: Batch) -> None:
        self.state.batch = batch

    @entrypoint
    def run_valid_stage(self) -> None:
        while batch := self.generate_batch():
            self.set_batch(batch)
            self.run_batch()

        stage_metrics = self.collect_stage_metrics()
        stage_metrics = {k: round(v, DefaultSettings.metric_round) for k, v in stage_metrics.items()}
        logger.debug(f'[epoch {self.current_epoch}] valid stage metrics: {stage_metrics}')
        self.valid_metrics_history.append(stage_metrics)

    @entrypoint
    def collect_stage_metrics(self) -> MetricDict:
        return self.state.stage_metrics

    @entrypoint
    def should_step(self) -> bool:
        return (self.state.num_batches % self.accumulate_grad_batches) == 0

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
        batch_output: BatchOutput = self.model(**self.state.batch)
        self.set_batch_output(batch_output)
        self.set_loss()

    @entrypoint
    def backward(self) -> None:
        self.accelerator.backward(self.state.loss)

    @entrypoint
    def set_batch_output(self, batch_output: BatchOutput) -> None:
        self.state.batch_output = batch_output

    @entrypoint
    def set_loss(self) -> None:
        loss = smartget(self.state.batch_output, 'loss')
        self.state.set_loss(loss, self.accumulate_grad_batches)

    @entrypoint
    def step(self) -> None:
        self.advance_step()
        self.optimizer.step()
        self.optimizer.zero_grad()

    @property
    def components(self):
        return self._components

    @property
    def hyper_params(self) -> HyperParams:
        hyper_params: HyperParams = {
            'epochs': self.epochs,
            'mixed_precision': self.mixed_precision,
            'accumulate_grad_batches': self.accumulate_grad_batches,
        }
        for component in self._components:
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
        return self.accelerator.unwrap_model(self.model)

    @property
    def raw_optimizer(self) -> torch.optim.Optimizer:
        if isinstance(self.optimizer, AcceleratedOptimizer):
            return self.optimizer.optimizer  # type: ignore
        else:
            return self.optimizer

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / DefaultSettings.checkpoints_root_dir_name

    @property
    def info(self) -> TrainerInfo:
        return TrainerInfo.from_trainer(self)

    def register_components(self, components: Iterable['BaseComponent']) -> None:
        for component in components:
            self.hook_engine.register(component)

    def log(
        self, data: Mapping[str, Union[float, torch.Tensor]], step: Optional[int] = None, prefix: Optional[str] = None
    ) -> None:
        if prefix is None:
            prefix = 'train' if self.in_train_stage else 'validation'

        if step is None:
            step = self.cum_steps

        scalars = {k: v for k, v in data.items() if isinstance(v, (int, float))}
        if scalars:
            scalars = self.logger.add_prefix(prefix, scalars)
            self.logger.log_metrics(scalars, step=step)

        tensors = {k: v for k, v in data.items() if isinstance(v, torch.Tensor)}
        if tensors:
            tensors = self.logger.add_prefix(prefix, tensors)
            self.logger.log_tensors(tensors, step=step)

    def save(self, checkpoint_name_or_dir: PathOrStr) -> None:
        is_checkpoint_name = isinstance(checkpoint_name_or_dir, str)
        if is_checkpoint_name:
            save_dir = self.output_dir / DefaultSettings.checkpoints_root_dir_name / checkpoint_name_or_dir
        else:
            if checkpoint_name_or_dir.is_dir():
                save_dir = checkpoint_name_or_dir
            else:
                raise ValueError(f'checkpoint_name_or_dir must be a directory, got {checkpoint_name_or_dir}')

        self.accelerator.save_state(str(save_dir))
        self.state.to_json(save_dir / DefaultSettings.trainer_state_json_file_name)

    def load(self, checkpoint_name_or_dir: PathOrStr) -> None:
        if isinstance(checkpoint_name_or_dir, Path):
            if checkpoint_name_or_dir.is_dir():
                load_dir = checkpoint_name_or_dir
            else:
                raise ValueError(f'checkpoint_name_or_dir must be a directory, got {checkpoint_name_or_dir}')
        else:
            load_dir = self.output_dir / DefaultSettings.checkpoints_root_dir_name / checkpoint_name_or_dir

        self.accelerator.load_state(str(load_dir))
        self.state = self.state.from_json(load_dir / DefaultSettings.trainer_state_json_file_name)
        self.state.checkpoint_entrypoint_stack = self.state.entrypoint_stack
        self.state.entrypoint_stack = []
        self.accelerator.wait_for_everyone()

    def _accelerate_prepare(self) -> None:
        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(  # type: ignore
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
        )  # type: ignore
        if self.valid_dataloader is not None:
            self.valid_dataloader = self.accelerator.prepare(self.valid_dataloader)  # type: ignore
        for component in self._components:
            if isinstance(component, Stateful):
                self.accelerator.register_for_checkpointing(component)

    def __getitem__(self, key: str):
        return self.exports[key]

    def __setitem__(self, key: str, item: Any):
        self.exports[key] = item

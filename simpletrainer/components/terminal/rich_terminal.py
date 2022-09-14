# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.pretty import Pretty
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule
from rich.table import Column, Table
from rich.text import Text
from torch.optim import Optimizer
from torchinfo import ModelStatistics

import simpletrainer.utils.torch as torch_utils
from simpletrainer import BaseComponent, Trainer, after, before, on
from simpletrainer.common.types import MetricDict
from simpletrainer.core.hook import HookCollection, Preposition, TrainerEvent
from simpletrainer.utils.common import pretty_str

console = Console()
logger = logging.getLogger(__name__)


def convert_dict_to_table(data: dict, title: str):
    column_names = [
        'name',
        'value',
    ]
    table = Table(
        *[Column('[bold]' + col) for col in column_names],
        title=title,
        title_style='i b magenta',
    )
    for k, v in data.items():
        table.add_row(
            Text(k, style='b'),
            Pretty(v),
        )
    return table


class RichLiveObj:
    def __init__(self):
        self.live_objs = {}

    def get_metrics_section(self) -> list:
        stage_metrics = self.live_objs.get('stage_metrics')
        if stage_metrics is None:
            return []
        else:
            panels = [Panel.fit(f'[b]{k}[/b]:\n[green]{v:.3f}[/green]') for k, v in stage_metrics.items()]
            return panels

    def __rich_console__(self, console, options):
        objs = [
            Rule('Trainer Progress'),
            Text('\n'),
            Columns([self.live_objs['progress']] + self.get_metrics_section()),
        ]
        if self.live_objs.get('epoch_metrics'):
            objs += [
                Text('\n'),
                self.live_objs.get('epoch_metrics'),
            ]
        yield from objs


class RichProgressBar(BaseComponent):
    only_main_process = True
    tags = ['progress_bar']
    train_task_id: TaskID
    valid_task_id: TaskID

    def __init__(self):
        self.progress = Progress(
            '{task.description}',
            BarColumn(),
            TextColumn('{task.fields[steps]}{task.fields[total_steps]}'),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn('[progress.percentage]{task.percentage:>3.0f}%'),
            console=console,
        )
        self.epoch_metrics_table = Table(title='[magenta]Epoch Metrics[/magenta]')
        self.rich_live_obj = RichLiveObj()
        self._live = Live(console=console)
        self._rich_handler_added = False

    # def with_trainer(self, trainer: Trainer) -> None:
    #     self.train_task_id = self.add_task('train', trainer.train_data_info.num_batches_per_epoch)
    #     if trainer.do_eval:
    #         self.valid_task_id = self.add_task('valid', trainer.valid_data_info.num_batches_per_epoch)  # type: ignore

    def _prepare_rich_handler(self):
        self.rich_handler = RichHandler(console=console)
        logging.root.addHandler(self.rich_handler)
        self.removed_handlers = []
        for handler in logging.root.handlers:
            if isinstance(handler, logging.StreamHandler):
                self.removed_handlers.append(handler)
                logging.root.removeHandler(handler)
        logger.debug(f'Remove handlers: {self.removed_handlers}')

    @staticmethod
    def get_total_steps_str(num_batches_per_epoch: Optional[int] = None):
        if num_batches_per_epoch is None:
            total_steps = ''
        else:
            total_steps = f'/{num_batches_per_epoch}'
        return total_steps

    def add_task(self, task_name: str, total_steps: Optional[int] = None):
        task_id = self.progress.add_task(
            f'[green]{task_name.capitalize()}[/green]',
            total=total_steps or 0,
            steps=0,
            start=False,
            total_steps=self.get_total_steps_str(total_steps),
        )
        return task_id

    @on(Trainer.EVENT.PREPARE)
    def fit_progress(self, trainer: Trainer):
        self.train_task_id = self.add_task('train', trainer.train_data_info.num_batches_per_epoch)
        if trainer.do_eval:
            self.valid_task_id = self.add_task('valid', trainer.valid_data_info.num_batches_per_epoch)  # type: ignore
        self._prepare_rich_handler()
        self._live.start()
        self._live.update(self.rich_live_obj)

    @after(Trainer.run_batch)
    def advance_batch(self, trainer: Trainer):
        task_id = self.train_task_id if trainer.in_train_stage else self.valid_task_id
        self.progress.update(task_id, steps=trainer.stage_state.num_batches, advance=1)

    @before(Trainer.run_stage)
    def start_train_stage(self, trainer: Trainer):
        if trainer.in_train_stage:
            self.progress.start_task(self.train_task_id)
        else:
            train_steps = trainer.train_data_info.num_batches_per_epoch
            if train_steps is not None:
                self.progress.update(self.train_task_id, steps=train_steps)
            self.progress.start_task(self.valid_task_id)

        self.rich_live_obj.live_objs['stage_metrics'] = trainer.stage_state.metrics

    @before(Trainer.run_epoch)
    def set_progress(self, trainer: Trainer):
        self.rich_live_obj.live_objs['progress'] = Panel.fit(
            self.progress,  # type: ignore
            title=f'Epoch {trainer.current_epoch}/{trainer.epochs}',
        )

    @after(Trainer.run_epoch)
    def refresh_progress(self, trainer: Trainer):

        self.progress.reset(
            self.train_task_id,
            start=False,
            steps=0,
            total_steps=self.get_total_steps_str(trainer.train_data_info.num_batches_per_epoch),
        )
        if trainer.do_eval:
            self.progress.reset(
                self.valid_task_id,
                start=False,
                steps=0,
                total_steps=self.get_total_steps_str(trainer.valid_data_info.num_batches_per_epoch),  # type: ignore
            )
        self.rich_live_obj.live_objs['progress'].title = f'Epoch {trainer.current_epoch}/{trainer.epochs}'
        self.rich_live_obj.live_objs['epoch_metrics'] = self.get_epoch_metrics_tabel(
            trainer.train_metrics_history,
            trainer.valid_metrics_history,
        )

    @on(Trainer.EVENT.TEARDOWN)
    def stop(self, trainer: Trainer):
        self.progress.stop()
        self._live.stop()
        for handler in logging.root.handlers:
            if handler is self.rich_handler:
                logging.root.removeHandler(handler)
        for handler in self.removed_handlers:
            logging.root.addHandler(handler)

    @staticmethod
    def get_epoch_metrics_tabel(
        train_metrics_list: list[MetricDict],
        valid_metrics_list: list[MetricDict],
    ):
        metric_set = set()
        for metrics in train_metrics_list:
            metric_set |= set(f'train-{i}' for i in metrics)
        for metrics in valid_metrics_list:
            metric_set |= set(f'valid-{i}' for i in metrics)

        all_metrics = sorted(list(metric_set))
        cols = ['epoch'] + all_metrics

        epoch_metrics_table = Table(*cols, title='[magenta]Epoch Metrics[/magenta]')
        for idx in range(len(train_metrics_list)):
            row_values = [Text(str(idx + 1), style='bold')]
            for metric in all_metrics:
                if metric.startswith('train'):
                    value = train_metrics_list[idx][metric[6:]]
                else:
                    value = valid_metrics_list[idx][metric[6:]]
                row_values.append(Text(str(round(value, 3))))
            epoch_metrics_table.add_row(*row_values)
        return epoch_metrics_table

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class RichInspect(BaseComponent):
    only_main_process = True
    tags = ['inspect']

    def __init__(self, model_summary_depth: int = 3):
        self.model_summary_depth = model_summary_depth

    @on(Trainer.EVENT.PREPARE)
    def show_trainer_info(self, trainer: Trainer):
        named_panels = [
            (
                'Model Info',
                self.get_model_panel(
                    trainer.raw_model,
                    None,
                    trainer.raw_optimizer,
                    trainer.device,
                ),
            ),
            (
                'Optimizer Info',
                self.get_optimizer_panel(trainer.raw_optimizer),
            ),
            (
                'Data Info',
                self.get_data_panel(trainer.train_data_info, trainer.valid_data_info),
            ),
            ('Trainer Info', self.get_trainer_panel(trainer)),
            (
                'Trainer Hook',
                self.get_hook_tabel_panel(trainer.hook_engine.entrypoint_hooks),
            ),
            (
                'Trainer Components',
                self.get_components_panel(trainer.components),
            ),
        ]
        for name, panels in named_panels:
            self.print_section(name, panels)

    @staticmethod
    def get_module_group_set(module, id_group_map: Dict[int, str]):
        group_set = set()
        for _, p in module.named_parameters():
            group = id_group_map.get(id(p))
            if group is not None:
                group_set.add(group)
        return group_set

    @staticmethod
    def get_model_param_panels(model_statistics: ModelStatistics):
        param_info = {
            'Total params': f'[green]{model_statistics.total_params}[/green]',
            'Trainable params': f'[green]{model_statistics.trainable_params}[/green]',
            'Non-trainable params': f'\
            [green]{model_statistics.total_params - model_statistics.trainable_params}[/green]',
        }
        panels = [Panel(f'[b]{k}:\n[/b]{v}') for k, v in param_info.items()]
        return panels

    def get_model_table(
        self,
        model_statistics: ModelStatistics,
        model: nn.Module,
        optimizer: Optimizer,
    ):
        column_names = [
            'Module',
            'Params',
            'Optimizer Group',
            'Output Shape',
            'Trainable',
        ]
        model_table = Table(
            *[Column('[bold]' + col) for col in column_names],
            title=f'{model.__class__.__name__} Architecture',
            title_style='i b magenta',
        )

        parameter_id_group_map = torch_utils.get_parameter_id_group_map(optimizer)
        id_layer_map = {layer.layer_id: layer for layer in model_statistics.summary_list}
        for module in model.modules():
            layer = id_layer_map.get(id(module))
            if layer is None or layer.depth > self.model_summary_depth:
                continue

            if layer.depth == 0:
                module_name = f'[bold green]{layer.class_name}'
            else:
                module_name = '[blue]|  [/blue]' * (layer.depth - 1) + '[blue]└─ [/blue]' + f'[bold green]{layer.class_name}'
            num_params = Pretty(layer.num_params) if layer.num_params else '--'
            _group_set = self.get_module_group_set(module, parameter_id_group_map)
            group = Pretty(_group_set) if _group_set else '--'
            output_size = Pretty(layer.output_size) if layer.output_size else '--'
            trainable = Pretty(layer.trainable_params != 0)
            model_table.add_row(module_name, num_params, group, output_size, trainable)
        return model_table

    @staticmethod
    def get_optimizer_panel(optimizer: Optimizer):
        optimizer_panel = Panel.fit(Pretty(optimizer), title='[b i magenta]Optimizer')
        return [Align.center(Columns([optimizer_panel], padding=(0, 5), equal=True))]

    @staticmethod
    def get_trainer_panel(trainer: Trainer):
        return [
            Align.center(
                Columns(
                    [
                        convert_dict_to_table(trainer.accelerator.state.__dict__, title='Accelerator'),
                        convert_dict_to_table(trainer.hyper_params, title='HyperParams'),
                        convert_dict_to_table(trainer.attributes, title='Attributes'),
                    ],
                    padding=(0, 5),
                    equal=True,
                )
            )
        ]

    @staticmethod
    def get_data_panel(train_data_info, valid_data_info):
        panels = [Panel.fit(Pretty(train_data_info), title='Train Data Info')]
        if valid_data_info is not None:
            panels.append(Panel.fit(Pretty(valid_data_info), title='Valid Data Info'))
        return [
            Align.center(
                Columns(
                    panels,
                    padding=(0, 5),
                    equal=True,
                )
            )
        ]

    @staticmethod
    def print_section(section_name: str, section_elements: list) -> None:
        section_elements.insert(0, Rule(f'[b]{section_name}'))
        for paragraph in section_elements:
            if isinstance(paragraph, list):
                for ele in paragraph:
                    console.print(ele)
            else:
                console.print(paragraph)
            console.print('\n')
        console.print('\n')

    def get_model_panel(
        self,
        model: nn.Module,
        input_data,
        optimizer: Optimizer,
        device: torch.device,
    ):
        model_statistics = torch_utils.get_model_info(model, input_data, device=device)
        model_table = self.get_model_table(model_statistics, model, optimizer)
        params_panels = self.get_model_param_panels(model_statistics)
        model_class_name = model.__class__.__name__

        return [
            Align.center(model_table),
            [
                Align.center(
                    Text(
                        f'{model_class_name} Params Statistics',
                        style='b i magenta',
                    )
                ),
                Align.center(Columns(params_panels, equal=True)),
            ],
        ]

    @staticmethod
    def get_components_panel(components):
        return [Align.center(Panel.fit(Group(*(Pretty(component) for component in components))))]

    @staticmethod
    def get_hook_tabel_panel(hooks: dict[str, HookCollection]):
        column_names = [
            'hook',
            'prep',
            'name',
        ]
        hook_table = Table(
            *[Column('[bold]' + col) for col in column_names],
            title='Hooks',
            title_style='i b magenta',
        )
        hook_event_set = set(i.value for i in list(TrainerEvent))
        for entrypoint_name, hook_collection in hooks.items():
            if entrypoint_name in hook_event_set:
                entrypoint_name = f'Event({entrypoint_name})'
            for preposition in Preposition:
                for hook in hook_collection.data[preposition]:
                    hook_table.add_row(
                        Text(entrypoint_name, style='b'),
                        Text(preposition, style='green'),
                        Text(hook.__qualname__),
                    )
        return [Align.center(hook_table)]

    @staticmethod
    def get_hyper_params_panel(hyper_params):
        return [Align.center(Panel(pretty_str(hyper_params, name='HyperParams')), width=100)]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

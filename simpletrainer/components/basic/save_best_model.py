from pathlib import Path
from typing import Any, Callable

import torch

from simpletrainer import AttrsComponent, DefaultSettings, Trainer, define, field, on
from simpletrainer.common.types import SignMetric
from simpletrainer.components.basic.best_model_state_tracker import (
    BestModelStateTracker,
)

SaveFunction = Callable[[Any, Path], None]


def torch_save(obj: Any, path: Path) -> None:
    torch.save(obj, path)


@define(only_main_process=True)
class SaveBestModel(AttrsComponent):
    sign_metric: SignMetric = field(default=None, converter=SignMetric.from_str)
    save_state_dict: bool = True
    save_model: bool = False
    save_function: SaveFunction = field(default=torch_save)

    def __attrs_post_init__(self) -> None:
        if (not self.save_state_dict) and (not self.save_model):
            raise ValueError('Either save_state_dict or save_model should be True')

    def with_trainer(self, trainer: Trainer) -> None:
        self.sign_metric = self.sign_metric or trainer.core_metric
        self.best_model_tracker = trainer.hook_engine.setdefault(BestModelStateTracker(sign_metric=self.sign_metric))

    @on(Trainer.EVENT.FINISH)
    def track_best_model(self, trainer: Trainer) -> None:
        if self.save_state_dict:
            self.save_function(
                self.best_model_tracker.best_model_state, trainer.output_dir / DefaultSettings.best_model_state_file_name
            )

        if self.save_model:
            trainer.raw_model.load_state_dict(self.best_model_tracker.best_model_state)
            self.save_function(trainer.raw_model, trainer.output_dir / DefaultSettings.best_model_file_name)

from pathlib import Path
from typing import Literal, Optional

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, PrecisionType
from pydantic import BaseModel, Field, validator

from simpletrainer.common.default_settings import DefaultSettings
from simpletrainer.common.types.sign_metric import SignMetric
from simpletrainer.utils.common import pretty_str, random_experiment_name


class AcceleratorConfig(BaseModel):
    cpu: bool = Field(default=False, description='Whether or not to force the script to execute on CPU.')
    mixed_precision: PrecisionType = PrecisionType.NO
    ddp_kwargs: dict = Field(default_factory=dict)

    def build(self):
        args = []
        if self.ddp_kwargs:
            args.append(DistributedDataParallelKwargs(**self.ddp_kwargs))

        if self.mixed_precision is None:
            _mixed_precision = PrecisionType.NO
        else:
            _mixed_precision = self.mixed_precision

        accelerator = Accelerator(
            mixed_precision=_mixed_precision,
            cpu=self.cpu,
            kwargs_handlers=args,
        )
        return accelerator


class TrainerConfig(BaseModel):
    epochs: int = DefaultSettings.epochs
    do_eval: bool = True
    accumulate_grad_batches: int = 1
    core_metric: SignMetric = SignMetric('-', 'loss')
    experiment_name: str = DefaultSettings.experiment_name
    run_name: str = Field(default_factory=lambda: random_experiment_name())
    output_dir: Optional[Path] = None
    inspect: bool = False
    progress_bar: Literal['rich', 'tqdm'] = 'rich'
    accelerator: AcceleratorConfig = Field(default_factory=AcceleratorConfig)
    logger: str = 'tensorboard'

    @validator('core_metric')
    def convert_string_to_sign_metric(cls, v):
        if isinstance(v, str):
            return SignMetric.from_str(v)
        elif isinstance(v, SignMetric):
            return v
        raise ValueError(f'Invalid core_metric: {v}')

    def __repr__(self) -> str:
        return pretty_str(self.dict(), self.__class__.__name__)

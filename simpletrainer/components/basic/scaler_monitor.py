from typing import Callable, Union, cast

from simpletrainer import AttrsComponent, Trainer, after, define

ScalerGetter = Callable[[Trainer], Union[int, float]]


@define
class ScalerMonitor(AttrsComponent):
    name: str
    getter: ScalerGetter
    intervel: int

    @after(Trainer.run_batch)
    def collect(self, trainer: Trainer):
        self.intervel = cast(int, self.intervel)

        if trainer.stage_state.num_batches % self.intervel == 0:
            trainer.update_metrics({self.name: self.getter(trainer)})

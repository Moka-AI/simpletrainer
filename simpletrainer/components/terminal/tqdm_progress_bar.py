import tqdm

from simpletrainer import BaseComponent, Trainer, after, before
from simpletrainer.common.types import MetricDict


class TqdmProgressBar(BaseComponent):
    only_main_process = True
    tags = ['progress_bar']

    def __init__(self, update_intervel: int = 10) -> None:
        self.update_intervel = update_intervel

    @before(Trainer.run_stage)
    def start_train(self, trainer: Trainer):
        if trainer.in_train_stage:
            total = trainer.train_data_info.num_batches_per_epoch
        else:
            total = trainer.valid_data_info.num_batches_per_epoch  # type: ignore

        self.progress_bar = tqdm.tqdm(
            total=total,
            desc=self.get_trainer_description(trainer),
            unit='bat',
        )

    @after(Trainer.run_stage)
    def close(self, trainer: Trainer):
        self.progress_bar.close()

    @after(Trainer.run_batch)
    def advance_batch(self, trainer: Trainer):
        self.progress_bar.update(1)
        if trainer.stage_state.num_batches % self.update_intervel == 1:
            description = (
                self.get_trainer_description(trainer) + ' | ' + self.get_metrics_description(trainer.stage_state.metrics)
            )
            self.progress_bar.set_description(description)

    @staticmethod
    def get_trainer_description(trainer: Trainer) -> str:
        stage_name = 'Trian' if trainer.in_train_stage else 'Valid'
        description = f'Epoch {trainer.current_epoch}/{trainer.epochs} | {stage_name}'
        return description

    @staticmethod
    def get_metrics_description(metrics: MetricDict) -> str:
        processed_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, float):
                processed_metrics[k] = f'{v:.3f}'
            else:
                processed_metrics[k] = v
        description = ' | '.join(f'{k}: {v}' for k, v in processed_metrics.items())
        return description

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

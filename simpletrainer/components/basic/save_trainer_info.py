import json

from simpletrainer import AttrsComponent, DefaultSettings, Trainer, define, on


@define(only_main_process=True)
class SaveTrainerInfo(AttrsComponent):
    @on(Trainer.EVENT.PREPARE)
    def save_config(self, trainer: Trainer):
        with open(trainer.output_dir / DefaultSettings.attributes_file_name, 'w') as f:
            json.dump(trainer.attributes, f, ensure_ascii=False)

        with open(trainer.output_dir / DefaultSettings.hyper_params_file_name, 'w') as f:
            json.dump(trainer.hyper_params, f, ensure_ascii=False)

    @on(Trainer.EVENT.FINISH)
    def save_on_finish(self, trainer: Trainer):
        with open(trainer.output_dir / DefaultSettings.hyper_params_file_name, 'w') as f:
            json.dump(trainer.hyper_params, f, ensure_ascii=False)

        metrics = {
            'train': trainer.train_metrics_history,
            'valid': trainer.valid_metrics_history,
        }
        with open(trainer.output_dir / DefaultSettings.metrics_file_name, 'w') as f:
            json.dump(metrics, f, ensure_ascii=False)

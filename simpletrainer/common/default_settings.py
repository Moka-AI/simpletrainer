from pydantic import BaseSettings


class SimpleTrainergDefaultSettings(BaseSettings):
    experiment_name: str = 'untitled_experiment'
    epochs: int = 3
    seed: int = 42
    core_metric: str = '-loss'
    logger: str = 'tensorboard'
    log_file_name: str = 'trainer.log'
    best_model_file_name: str = 'best_model.pt'
    best_model_state_file_name: str = 'best_model_state.pt'
    metrics_file_name = 'metrics.json'
    realtime_metric_prefix: str = 'rt_'
    model_state_file_name: str = 'pytorch_model.bin'
    trainer_info_json_file_name: str = 'trainer_info.json'
    trainer_state_json_file_name: str = 'trainer_state.json'
    checkpoints_root_dir_name: str = 'checkpoints'
    checkpoints_register_book_file_name: str = 'checkpoints_register_book.txt'
    pred_key: str = 'logits'
    target_key: str = 'labels'
    metric_round: int = 4
    log_format: str = '[%(levelname)s] %(asctime)s %(module)s: %(message)s'
    date_format = '%Y-%m-%d %H:%M'

    class Config:
        env_prefix = 'SIMPLETRAINER_'


DefaultSettings = SimpleTrainergDefaultSettings()

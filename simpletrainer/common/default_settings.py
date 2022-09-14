from pydantic import BaseSettings


class SimpleTrainergDefaultSettings(BaseSettings):
    experiment_name: str = 'experiments'
    epochs: int = 3
    seed: int = 42
    sign_metric: str = '-loss'
    log_file_name: str = 'trainer.log'
    best_model_file_name: str = 'best_model.pt'
    best_model_state_file_name: str = 'best_model_state.pt'
    attributes_file_name: str = 'attributes.json'
    hyper_params_file_name: str = 'hyper_params.json'
    metrics_file_name = 'metrics.json'
    realtime_metric_prefix: str = 'rt_'
    model_state_file_name: str = 'pytorch_model.bin'
    trainer_state_file_name: str = 'trainer_state.bin'
    checkpoints_root_dir_name: str = 'checkpoints'
    pred_key: str = 'logits'
    target_key: str = 'labels'
    metric_round: int = 4
    log_format: str = '[%(levelname)s] %(asctime)s %(module)s: %(message)s'
    date_format = '%Y-%m-%d %H:%M'


DefaultSettings = SimpleTrainergDefaultSettings()

import socket
import traceback
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from simpletrainer import Trainer

DATE_FORMAT = '%Y-%m-%d %H:%M'


class NotificationType(str, Enum):
    START = 'start'
    FINISH = 'finish'
    CRASH = 'crash'


@dataclass
class NotificationInfo:
    content: str
    type_: NotificationType


def start_info(trainer: Trainer) -> NotificationInfo:
    host_name = socket.gethostname()
    date = datetime.now().strftime(DATE_FORMAT)
    contents = [
        f'Machine: {host_name}',
        f'Model: {trainer.model.__class__.__name__}',
        f'Experiment: {trainer.experiment_name}',
        f'Start: {date}',
    ]
    return NotificationInfo(
        content='\n'.join(contents),
        type_=NotificationType.START,
    )


def finish_info(trainer: Trainer) -> NotificationInfo:
    host_name = socket.gethostname()
    date = datetime.now().strftime(DATE_FORMAT)

    contents = [
        f'Machine: {host_name}',
        f'Model: {trainer.model.__class__.__name__}',
        f'Experiment: {trainer.experiment_name}',
        f'End: {date}',
        'Train Metrics:',
        str(trainer.train_metrics_history),
    ]
    if trainer.do_eval:
        contents.append('Validation Metrics:')
        contents.append(str(trainer.valid_metrics_history))
    return NotificationInfo(
        content='\n'.join(contents),
        type_=NotificationType.FINISH,
    )


def crash_info(trainer: Trainer) -> NotificationInfo:
    host_name = socket.gethostname()
    date = datetime.now().strftime(DATE_FORMAT)

    contents = [
        f'Machine: {host_name}',
        f'Model: {trainer.model.__class__.__name__}',
        f'Experiment: {trainer.experiment_name}',
        f'Crash: {date}',
        "Here's the error:",
        f'trainer.exception\n\n',
        'Traceback:',
        str(traceback.format_exc()),
    ]
    return NotificationInfo(
        content='\n'.join(contents),
        type_=NotificationType.CRASH,
    )

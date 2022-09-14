# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from datetime import datetime

from simpletrainer import AttrsComponent, DefaultSettings, Trainer, define, field, on

logger = logging.getLogger(__name__)


@define(only_main_process=True)
class Timer(AttrsComponent):
    start_time: datetime = field(init=False, export=True)
    end_time: datetime = field(init=False, export=True)

    @on(Trainer.EVENT.PREPARE, try_first=True)
    def start(self, trainer: Trainer):
        self.start_time = datetime.now()

    @on(Trainer.EVENT.FINISH, try_first=True)
    def stop(self, trainer: Trainer):
        self.end_time = datetime.now()
        logger.info(f'🎉  {trainer.experiment_name}/{trainer.run_name} Finish, Elapsed Time: {self.elapsed}')

    @property
    def start_string(self) -> str:
        return self.start_time.strftime(DefaultSettings.date_format)

    @property
    def end_string(self) -> str:
        return self.end_time.strftime(DefaultSettings.date_format)

    @property
    def elapsed(self) -> str:
        time_delta = self.end_time - self.start_time
        mins = time_delta.total_seconds() // 60
        elapsed = f'{int(mins // 60)}h,{int(mins % 60)}m'
        return elapsed

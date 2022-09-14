# -*- coding: utf-8 -*-
from simpletrainer import BaseComponent, Trainer, on
from simpletrainer.components.notification.common import (
    NotificationInfo,
    crash_info,
    finish_info,
    start_info,
)


class BaseNotification(BaseComponent):
    @on(Trainer.EVENT.PREPARE)
    def send_start_info(self, trainer: Trainer) -> None:
        self.send(start_info(trainer))

    @on(Trainer.EVENT.FINISH)
    def send_finish_info(self, trainer: Trainer) -> None:
        self.send(finish_info(trainer))

    @on(Trainer.EVENT.CRASH)
    def send_crash_info(self, trainer: Trainer) -> None:
        self.send(crash_info(trainer))

    def send(self, info: NotificationInfo) -> None:
        raise NotImplementedError

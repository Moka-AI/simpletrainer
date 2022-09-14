# -*- coding: utf-8 -*-
import os
import warnings
from typing import Optional

import requests

from simpletrainer import define
from simpletrainer.components.notification.base import BaseNotification
from simpletrainer.components.notification.common import (
    NotificationInfo,
    NotificationType,
)

_notification_type_title_map = {
    NotificationType.START: ' 🎬 Train Started ',
    NotificationType.FINISH: '🎉 Train Completed',
    NotificationType.CRASH: '💥 Train Crashed',
}


@define(only_main_process=True, tags=('notification',))
class BarkNotification(BaseNotification):
    token: Optional[str] = None

    def __attrs_post_init__(self) -> None:
        self.token = self.token or os.getenv('BARK_TOKEN')
        if self.token is None:
            warnings.warn('Can not get bark token, BarkNotificationCallback will disabled!')
            self.bark_api = None
        else:
            self.bark_api = f'https://api.day.app/{self.token}/'

    def send(self, info: NotificationInfo) -> None:
        if self.bark_api is not None:
            requests.post(
                self.bark_api,
                json={
                    'title': _notification_type_title_map[info.type_],
                    'body': info.content,
                },
            )

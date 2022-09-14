# -*- coding: utf-8 -*-
import os
import warnings

import requests

from simpletrainer import define, field
from simpletrainer.components.notification.base import BaseNotification
from simpletrainer.components.notification.common import (
    NotificationInfo,
    NotificationType,
)

_notification_type_title_map = {
    NotificationType.START: 'Your train has started 🎬',
    NotificationType.FINISH: 'Your train is complete 🎉',
    NotificationType.CRASH: 'Your train is crashed 💥',
}


@define(only_main_process=True, tags=('notification',))
class WechatNotification(BaseNotification):
    webhook_url: str = field(default=None)

    def __attrs_post_init__(self):
        self.webhook_url = self.webhook_url or os.getenv('WECHAT_WEBHOOK_URL', '')

        if not self.webhook_url:
            warnings.warn('Can not get notification webhook url, WechatNotificationCallback will disabled!')

    def send(self, info: NotificationInfo) -> None:
        title = _notification_type_title_map[info.type_]
        msg = {
            'msgtype': 'text',
            'text': {'content': ''},
        }
        msg['text']['content'] = title + '\n\n' + info.content
        if self.webhook_url:
            requests.post(self.webhook_url, json=msg)

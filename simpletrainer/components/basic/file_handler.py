# -*- coding: utf-8 -*-
import logging

from simpletrainer import AttrsComponent, DefaultSettings, Trainer, define, on


@define(only_main_process=True)
class FileHandler(AttrsComponent):
    log_file_name: str = DefaultSettings.log_file_name
    level: int = logging.INFO

    @on(Trainer.EVENT.PREPARE)
    def prepare_handler(self, trainer: Trainer):
        log_file = trainer.output_dir / self.log_file_name
        self.file_handler = logging.FileHandler(str(log_file))
        self.file_handler.setLevel(self.level)
        logging.root.addHandler(self.file_handler)

    @on(Trainer.EVENT.TEARDOWN)
    def remove_handler(self, trainer: Trainer):
        logging.root.removeHandler(self.file_handler)

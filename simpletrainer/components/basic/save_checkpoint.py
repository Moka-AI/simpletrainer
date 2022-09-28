from simpletrainer import BaseComponent, Trainer, after


class SaveCheckpoint(BaseComponent):
    def __init__(self, save_every_steps: int = 100) -> None:
        super().__init__()
        self.save_every_steps = save_every_steps

    @after(Trainer.run_epoch)
    def save_checkpoint(self, trainer: Trainer) -> None:
        trainer.save(f'epoch-{trainer.current_epoch}')

    def __repr__(self) -> str:
        return 'SaveCheckpoint()'

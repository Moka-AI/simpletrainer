import torch.nn as nn

from simpletrainer.components import LabelSmoothing


def test_label_smoothing():
    label_smoothing = LabelSmoothing(rate=0.1)

    class FakeTrainer:
        def __init__(self, model) -> None:
            self.model = model

    class SimpleModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.criterion = nn.CrossEntropyLoss()

    model = SimpleModel()
    trainer = FakeTrainer(model)
    label_smoothing.set_label_smoothing(trainer)  # type: ignore
    assert model.criterion.label_smoothing == 0.1

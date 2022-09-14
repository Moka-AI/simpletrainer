from dataclasses import dataclass
from typing import Literal

Sign = Literal['+', '-']


@dataclass
class SignMetric:
    sign: Sign
    name: str

    @classmethod
    def from_str(cls, sign_metric_str: str) -> 'SignMetric':
        if isinstance(sign_metric_str, SignMetric):
            return sign_metric_str

        sign = sign_metric_str[0]
        name = sign_metric_str[1:]
        if sign != '+' and sign != '-':
            raise ValueError(f'Invalid sign: {sign_metric_str[0]}')
        if len(name.strip()) == 0:
            raise ValueError(f'Empty metric, {sign_metric_str}')
        return cls(sign, name)

    def is_better(self, score_a: float, score_b: float) -> bool:
        if self.sign == '+':
            return score_a > score_b
        else:
            return score_a < score_b

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        elif isinstance(other, SignMetric):
            return self.name == other.name
        else:
            return False

    def __repr__(self):
        return f'{self.sign}{self.name}'

    def __str__(self) -> str:
        return self.__repr__()

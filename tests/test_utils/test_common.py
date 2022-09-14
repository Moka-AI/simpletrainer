import random

import pytest

from simpletrainer.utils.common import *


def test_temp_random_seed():
    random.seed(0)
    old_state = random.getstate()
    with temp_random_seed(1):
        new_state = random.getstate()
        assert old_state != new_state
    recovered_state = random.getstate()
    assert old_state == recovered_state


def test_random_experiment_name():
    random.seed(0)
    experiment_name1 = random_experiment_name()
    assert isinstance(experiment_name1, str)
    time.sleep(0.1)
    random.seed(0)
    experiment_name2 = random_experiment_name()
    assert experiment_name1 != experiment_name2


def test_get_init_params():
    class Algorithm:
        def __init__(self, name, is_component=True, *args, **kwargs):
            self.name = name
            self.is_component = is_component

    a1 = Algorithm('AMP')
    assert get_init_params(a1) == {'is_component': True, 'name': 'AMP'}
    a2 = Algorithm('AWP', False)
    assert get_init_params(a2) == {'is_component': False, 'name': 'AWP'}

    class BadAlgorithm:
        def __init__(self, name, is_component=True, *args, **kwargs):
            self.name = name
            self.in_trainer = is_component

    b1 = BadAlgorithm('AMP')
    with pytest.raises(AttributeError, match='maybe attribute name'):
        get_init_params(b1)

import numpy as np


def test_threshold_uses_oos_budget_and_maximizes_correct_routes():
    from intent_gate.policy import tune_threshold
    result = tune_threshold(np.array([.9, .8, .6]), np.array([True, True, False]), np.array([.7, .5]), max_far=0.0)
    assert .7 < result['threshold'] <= .8
    assert result['oos_false_accept_rate'] == 0
    assert result['correct_route_rate'] == 2 / 3

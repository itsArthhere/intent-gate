"""Small explicit examples guard against misleading metric denominators."""
import pytest
from intent_gate.modeling import evaluate


def test_selective_accuracy_distinguishes_unsupported_acceptances():
    report = evaluate(
        ['a', 'b', 'c', 'd'], ['a', 'wrong', 'c', 'wrong'],
        [.9, .8, .7, .2], [.95, .1], threshold=.7,
    )
    assert report['intent_accuracy'] == .5
    assert report['in_scope_coverage'] == .75
    assert report['selective_accuracy'] == pytest.approx(2 / 3)
    assert report['oos_false_accept_rate'] == .5
    assert report['mixed_coverage'] == pytest.approx(4 / 6)
    assert report['mixed_selective_accuracy'] == .5
    assert report['correct_route_rate'] == .5


def test_no_accepted_requests_is_not_perfect_accuracy():
    report = evaluate(['a', 'b'], ['a', 'b'], [.2, .3], [.1], threshold=.99)
    assert report['selective_accuracy'] is None
    assert report['mixed_selective_accuracy'] is None
    assert report['in_scope_coverage'] == 0
    assert report['oos_false_accept_rate'] == 0


def test_oos_detection_auroc_uses_low_confidence_as_positive_signal():
    report = evaluate(['a', 'b'], ['a', 'b'], [.8, .9], [.1, .2], threshold=.5)
    assert report['oos_auroc'] == 1

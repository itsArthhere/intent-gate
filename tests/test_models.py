import numpy as np
import pytest


def test_models_fit_real_text_and_return_probabilities():
    from intent_gate.modeling import build_models, evaluate
    texts = ['send money now', 'transfer dollars', 'pay my friend', 'send cash please', 'weather today', 'rain tomorrow', 'sunny forecast', 'weather outside']
    labels = ['transfer']*4 + ['weather']*4
    for model in build_models().values():
        model.fit(texts, labels)
        p = model.predict_proba(texts)
        assert p.shape == (8,2)
        np.testing.assert_allclose(p.sum(axis=1), 1)
    metrics = evaluate(np.array(['a','b']), np.array(['a','a']), np.array([.9,.7]), np.array([.8,.2]), .8)
    assert metrics['intent_accuracy'] == .5
    assert metrics['in_scope_coverage'] == .5
    assert metrics['selective_accuracy'] == 1
    assert metrics['oos_false_accept_rate'] == .5
    assert metrics['mixed_selective_accuracy'] == .5


def test_no_accepted_inputs_is_not_reported_as_perfect_accuracy():
    from intent_gate.modeling import evaluate
    m = evaluate(np.array(['a']),np.array(['a']),np.array([.1]),np.array([.2]),.9)
    assert m['selective_accuracy'] is None
    assert m['mixed_selective_accuracy'] is None

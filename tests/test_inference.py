import pytest


def test_tied_scores_match_evaluation_first_maximum():
    from intent_gate.inference import Router
    from intent_gate.modeling import score
    from sklearn.dummy import DummyClassifier

    model = DummyClassifier(strategy='prior').fit(['alpha', 'beta'], ['a', 'b'])
    expected, _ = score(model, ['request'])
    router = Router({'model': model, 'threshold': .5, 'name': 'tie-test'})
    result = router.route('request')
    assert result['intent'] == expected[0] == 'a'
    assert [c['intent'] for c in result['candidates']] == ['a', 'b']


def test_inference_abstains_for_unseen_vocabulary_and_rejects_empty():
    from intent_gate.inference import Router
    from intent_gate.modeling import build_models
    model=build_models()['word_lr']
    model.fit(['send money', 'transfer cash', 'sunny weather', 'rain forecast'],['pay','pay','weather','weather'])
    router=Router({'model':model,'threshold':.8,'name':'unit'})
    result=router.route('zzzzxxxx qqqqvvvv')
    assert result['decision']=='abstain'
    assert result['intent'] is None
    assert len(result['candidates'])==2
    with pytest.raises(ValueError): router.route('  ')
    with pytest.raises(ValueError): router.route('x'*2001)

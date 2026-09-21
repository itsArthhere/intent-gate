import pytest


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

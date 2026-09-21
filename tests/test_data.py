def test_download_official_splits_and_audit():
    from intent_gate.data import load_data, audit
    data = load_data()
    assert {k: len(v) for k, v in data.items()} == {'train':15000,'val':3000,'test':4500,'oos_train':100,'oos_val':100,'oos_test':1000}
    assert len({y for _, y in data['train']}) == 150
    report = audit({'train': [[' Hello  WORLD ', 'a']], 'test': [['hello world', 'b']]})
    assert report['cross_split_normalized_overlap']['train:test'] == 1
    assert report['conflicting_normalized_texts'] == 1

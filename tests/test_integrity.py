import pytest
from intent_gate import data


def test_modified_cached_dataset_is_rejected_before_parsing(tmp_path, monkeypatch):
    raw = tmp_path / 'data' / 'raw'
    raw.mkdir(parents=True)
    (raw / 'data_full.json').write_text('{}', encoding='utf-8')
    (raw / 'LICENSE').write_text('Test-only fixture', encoding='utf-8')
    monkeypatch.setattr(data, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='checksum mismatch'):
        data.load_data()

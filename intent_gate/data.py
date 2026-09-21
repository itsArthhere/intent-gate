"""Pinned official CLINC150 download and text-overlap audit."""
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '828f8093932c8fe6ca7936c3d2e52903b1c523de'
BASE = f'https://raw.githubusercontent.com/clinc/oos-eval/{COMMIT}'
DATA_SHA256 = '36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0'


def load_data():
    directory = ROOT / 'data/raw'
    directory.mkdir(parents=True, exist_ok=True)
    for name, remote in [('data_full.json', 'data/data_full.json'), ('LICENSE', 'LICENSE')]:
        path = directory / name
        if not path.exists():
            response = requests.get(f'{BASE}/{remote}', timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
    raw = (directory / 'data_full.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError('Dataset checksum mismatch; remove data/raw/data_full.json and download the pinned source again.')
    data = json.loads(raw)
    expected = {'train': 15000, 'val': 3000, 'test': 4500, 'oos_train':100, 'oos_val':100, 'oos_test':1000}
    if {k:len(v) for k,v in data.items()} != expected:
        raise ValueError('Unexpected official split sizes')
    for rows in data.values():
        if any(not isinstance(t, str) or not t.strip() or not isinstance(y,str) for t,y in rows):
            raise ValueError('Malformed or empty dataset row')
    provenance = {'repository':'https://github.com/clinc/oos-eval', 'commit':COMMIT,
                  'data_url':f'{BASE}/data/data_full.json', 'license_url':f'{BASE}/LICENSE',
                  'license':'CC BY 3.0', 'sha256':hashlib.sha256((directory/'data_full.json').read_bytes()).hexdigest()}
    (ROOT/'outputs').mkdir(exist_ok=True)
    (ROOT/'outputs/provenance.json').write_text(json.dumps(provenance,indent=2), encoding='utf-8')
    return data


def normalize(text):
    return ' '.join(text.casefold().split())


def audit(data):
    sets, labels, splits = {}, defaultdict(set), {}
    for split, rows in data.items():
        texts = [t for t,_ in rows]
        normalized = [normalize(t) for t in texts]
        counts = Counter(y for _,y in rows)
        sets[split] = set(normalized)
        splits[split] = {'rows':len(rows), 'classes':len(counts), 'class_min':min(counts.values()),
                         'class_max':max(counts.values()), 'exact_duplicate_rows':len(texts)-len(set(texts)),
                         'normalized_duplicate_rows':len(texts)-len(set(normalized)),
                         'empty_texts':sum(not t.strip() for t in texts),
                         'word_count_mean':sum(len(t.split()) for t in texts)/len(texts)}
        for t,y in rows:
            labels[normalize(t)].add(y)
    return {'splits':splits, 'cross_split_normalized_overlap':
            {f'{a}:{b}':len(sets[a]&sets[b]) for a,b in itertools.combinations(data,2)},
            'conflicting_normalized_texts':sum(len(v)>1 for v in labels.values()),
            'normalization':'Unicode casefold and collapse whitespace; punctuation retained',
            'protocol':'Official splits retained, overlaps reported, no test-based training filtering.'}


if __name__ == '__main__':
    print(json.dumps(audit(load_data()), indent=2))

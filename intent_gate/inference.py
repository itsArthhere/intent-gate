"""Local inference. Load joblib artifacts only from a trusted training run."""
import argparse
import json
import joblib
import numpy as np
from intent_gate.data import ROOT


class Router:
    def __init__(self, artifact=None):
        self.artifact = artifact if artifact is not None else joblib.load(ROOT/'models/router.joblib')

    def route(self, text):
        if not isinstance(text,str) or not text.strip() or len(text)>2000:
            raise ValueError('Enter 1–2000 characters of nonempty text.')
        model=self.artifact['model']
        p=model.predict_proba([text])[0]
        ranked=np.argsort(p)[::-1][:3]
        confidence=float(p[ranked[0]])
        accepted=confidence>=self.artifact['threshold']
        return {'decision':'route' if accepted else 'abstain',
                'intent':str(model.classes_[ranked[0]]) if accepted else None,
                'confidence':confidence,'threshold':self.artifact['threshold'],'model':self.artifact['name'],
                'candidates':[{'intent':str(model.classes_[i]),'score':float(p[i])} for i in ranked]}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Route an utterance or abstain')
    parser.add_argument('text')
    print(json.dumps(Router().route(parser.parse_args().text),indent=2))

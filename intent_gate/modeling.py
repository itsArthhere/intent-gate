"""CPU text baselines. Every vectorizer is fitted only on training data."""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def word_features():
    return TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=30000, sublinear_tf=True)


def build_models():
    word_lr = Pipeline([('tfidf',word_features()), ('classifier',LogisticRegression(C=10, max_iter=300, solver='lbfgs', random_state=42))])
    hybrid = Pipeline([('features',FeatureUnion([
        ('word',word_features()),
        ('char',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,max_features=40000,sublinear_tf=True))])),
        ('classifier',LogisticRegression(C=10,max_iter=300,solver='lbfgs',random_state=42))])
    # Wrapping the WHOLE pipeline avoids fitting vocabulary/IDF on calibration folds.
    svm = CalibratedClassifierCV(Pipeline([('tfidf',word_features()),
        ('classifier',LinearSVC(C=1,random_state=42,dual='auto'))]),method='sigmoid',cv=3,n_jobs=1)
    return {'word_lr':word_lr, 'word_char_lr':hybrid, 'calibrated_svm':svm}


def score(model, texts):
    probabilities = model.predict_proba(texts)
    index = probabilities.argmax(axis=1)
    return model.classes_[index], probabilities[np.arange(len(index)),index]


def evaluate(labels, predictions, in_scores, oos_scores, threshold):
    correct = np.asarray(labels) == np.asarray(predictions)
    accepted = np.asarray(in_scores) >= threshold
    false_accept = np.asarray(oos_scores) >= threshold
    total_accepted = int(accepted.sum()+false_accept.sum())
    return {'intent_accuracy':float(accuracy_score(labels,predictions)),
        'intent_macro_f1':float(f1_score(labels,predictions,average='macro',zero_division=0)),
        'oos_auroc':float(roc_auc_score(np.r_[np.zeros(len(in_scores)),np.ones(len(oos_scores))],1-np.r_[in_scores,oos_scores])),
        'in_scope_coverage':float(accepted.mean()),
        'selective_accuracy':float(correct[accepted].mean()) if accepted.any() else None,
        'correct_route_rate':float((correct & accepted).mean()),
        'oos_false_accept_rate':float(false_accept.mean()),
        'mixed_coverage':total_accepted/(len(in_scores)+len(oos_scores)),
        'mixed_selective_accuracy':float((correct & accepted).sum()/total_accepted) if total_accepted else None}

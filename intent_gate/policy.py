"""Validation-only threshold selection; acceptance is score >= threshold."""
import numpy as np


def tune_threshold(in_scores, correct, oos_scores, max_far=0.05):
    in_scores, correct, oos_scores = map(np.asarray, (in_scores, correct, oos_scores))
    if not len(in_scores) or not len(oos_scores) or len(correct) != len(in_scores):
        raise ValueError('Nonempty aligned validation scores required')
    if not 0 <= max_far <= 1:
        raise ValueError('max_far must be between zero and one')
    if not np.isfinite(in_scores).all() or not np.isfinite(oos_scores).all():
        raise ValueError('Scores must be finite')
    candidates = np.unique(np.r_[0., in_scores, oos_scores, np.nextafter(oos_scores, np.inf), 1.0000001])
    best = None
    for threshold in candidates:
        accepted = in_scores >= threshold
        far = float(np.mean(oos_scores >= threshold))
        if far > max_far + 1e-12:
            continue
        row = {'threshold': float(threshold), 'oos_false_accept_rate': far,
               'correct_route_rate': float(np.mean(accepted & correct)),
               'in_scope_coverage': float(accepted.mean())}
        # Prefer more correct routes, then fewer mistakes, then stricter threshold.
        key = (row['correct_route_rate'], -float(np.mean(accepted & ~correct)), threshold)
        if best is None or key > best[0]:
            best = key, row
    return best[1]

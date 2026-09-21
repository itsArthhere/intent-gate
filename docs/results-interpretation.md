# Reading the results

The selected word + character TF-IDF logistic regression model was chosen using validation data. Its acceptance threshold was frozen before test reporting. The figures below refer to that saved run in `outputs/metrics.json`; rerunning training can change timings and may change numerical details across library versions.

## What the routing policy actually does

Of 4,500 supported test requests:

- 3,424 are accepted for routing.
- 3,380 of those accepted requests receive the right intent.
- 44 accepted requests receive the wrong intent.
- The rest are rejected for review, including some requests that the underlying classifier could have labeled correctly.

Of 1,000 unsupported test requests, 40 are accepted incorrectly. These are not included in the **in-scope selective accuracy** denominator. The report also includes **mixed selective accuracy**, which penalizes those false acceptances.

These are held-out benchmark results, not a customer-facing service guarantee. The source has two normalized train/test text overlaps; `nonoverlap_intent_accuracy` reports classification accuracy after excluding those test rows for sensitivity analysis. The original split is retained for the principal experiment.

## Sample uncertainty

A 4% unsupported false-acceptance rate means 40 observed errors, not that the next 1,000 requests will contain exactly 40 errors. For a fixed policy and independent draws from the same distribution, the exact binomial 95% interval for 40/1,000 is approximately **2.87%–5.41%**. The interval for supported coverage, 3,424/4,500, is approximately **74.81%–77.33%**.

These intervals quantify finite-sample uncertainty only. They do not cover domain drift, multi-intent messages, adversarial input or repeated tuning on the test set. The much smaller unsupported validation sample is another reason not to treat the chosen 5% validation constraint as a guarantee.

## Reproduce the intervals

Run after training:

```python
import json
from pathlib import Path
from scipy.stats import binomtest

report = json.loads(Path('outputs/metrics.json').read_text())
metrics = report['models'][report['selected_model']]['test']
for metric, total in [('in_scope_coverage', 4500), ('oos_false_accept_rate', 1000)]:
    count = round(metrics[metric] * total)
    print(metric, count, total, binomtest(count, total).proportion_ci(method='exact'))
```

SciPy is installed as a scikit-learn dependency. The denominators above are the original CLINC150 full test split sizes; update them if evaluating a different dataset.

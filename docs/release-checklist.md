# Release checklist

## Data and experiment

- [ ] Verify the downloaded dataset checksum against the recorded source.
- [ ] Inspect split sizes, label counts, empty inputs and normalized duplicates.
- [ ] Keep test examples out of fitting, model selection and threshold tuning.
- [ ] Fit feature vocabularies and supervised estimators on training data only.
- [ ] Use validation data to choose the model and acceptance policy.
- [ ] Freeze the selection before reading the final test report.
- [ ] Report supported-intent classification separately from unsupported-request detection.
- [ ] Include accepted-error examples and false-acceptance counts, not only aggregate accuracy.
- [ ] Distinguish benchmark results from live traffic expectations.

## Local artifact

- [ ] Regenerate the model and reports from the documented commands.
- [ ] Execute the explanatory notebook from its first cell.
- [ ] Exercise empty input, long input, missing artifacts and abstention behavior.
- [ ] Exercise the UI's inference path, not only its HTTP health endpoint.
- [ ] Verify that model artifacts and raw data are excluded from source control.
- [ ] Check that citations and dataset licensing remain visible in public reports.

## Before connecting real handlers

The repository does not execute user actions. A deployment that does needs additional controls:

- [ ] Authenticate the caller and authorize each proposed action independently of the model.
- [ ] Test model startup, artifact compatibility, timeouts and rollback.
- [ ] Set review-queue capacity and define behavior when it is unavailable.
- [ ] Define data retention and access policies before storing user messages.
- [ ] Measure latency under concurrent requests with the deployed artifact.
- [ ] Evaluate on domain-specific requests and a time-separated holdout.
- [ ] Get a domain owner to approve the cost of wrong routes versus manual review.

This is a reusable checklist, not a record that deployment checks have passed.

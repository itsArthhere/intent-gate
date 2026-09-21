# Dataset provenance

Source: [CLINC out-of-scope evaluation repository](https://github.com/clinc/oos-eval), accompanying Larson et al., *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction*, EMNLP-IJCNLP 2019 ([paper](https://aclanthology.org/D19-1131/)).

The official README describes crowdsourced English single-intent requests. These are real published benchmark data, not production support tickets. The benchmark spans 150 supported intents; using it for routing does not make the software a deployed support system.

Upstream [license](https://github.com/clinc/oos-eval/blob/master/LICENSE): [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). Attribution applies to dataset-derived examples as well as the original data. Raw data are downloaded separately, not committed here.

## Independent source check

Downloaded from `https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json`.

SHA-256 of the downloaded bytes:

```
36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0
```

| Split | In scope | Out of scope |
|---|---:|---:|
| Train | 15000 | 100 |
| Validation | 3000 | 100 |
| Test | 4500 | 1000 |

After lowercase and whitespace normalization, the original in-scope train/validation sets share 3 distinct texts; train/test share 2; validation/test share none. This is a property of the source splits. The experiment's data audit and preprocessing document how overlaps are handled. Do not assume official splits are automatically leakage-free.

The validation set has relatively few unsupported requests, and the test set has a different unsupported-request fraction. Overall acceptance rates therefore cannot be compared without considering that mixture. Model choice and rejection thresholds must use validation only; the test set is for the final report.

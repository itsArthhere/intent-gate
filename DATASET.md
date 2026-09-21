# Dataset attribution

CLINC150 / Out-of-Scope Prediction, released by Stefan Larson and collaborators in the official [clinc/oos-eval repository](https://github.com/clinc/oos-eval).

Paper: **An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction**, EMNLP-IJCNLP 2019. https://aclanthology.org/D19-1131/

The upstream repository LICENSE was checked directly and is **Creative Commons Attribution 3.0 Unported (CC BY 3.0)**: https://creativecommons.org/licenses/by/3.0/ . The download script saves an unmodified copy next to the raw dataset. Pinned source commit: `828f8093932c8fe6ca7936c3d2e52903b1c523de`.

The dataset is unmodified. Aggregate analyses, figures, and the attributed utterance excerpts in `outputs/high_confidence_errors.csv` are derived from it. Casefold/whitespace normalization is used only in the duplicate audit; TF-IDF performs its own preprocessing. These data derivatives retain the upstream CC BY 3.0 attribution. The MIT license in this repository applies only to project code, not the dataset.

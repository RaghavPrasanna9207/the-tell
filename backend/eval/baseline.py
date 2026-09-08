"""TF-IDF + Logistic Regression baseline classifier.

This is the point of comparison that gives an accuracy number meaning. A
strong classifier "94% accurate" on a skewed base rate isn't impressive on
its own — showing what a simple linear baseline gets you first is what
makes the harder models' numbers legible. See docs/DESIGN_RULES.md and the plan's
"Eval A" section.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from app.taxonomy import Technique


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            (
                "clf",
                OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced")),
            ),
        ]
    )


def load_gold_as_arrays(records: list[dict]) -> tuple[list[str], np.ndarray, list[str]]:
    """Convert [{"text", "labels"}] records into (texts, y_true, technique_names).

    technique_names is fixed to taxonomy order (not alphabetical) so output
    always lines up with `app.taxonomy.Technique`.
    """
    technique_names = [t.value for t in Technique]
    texts = [r["text"] for r in records]
    label_sets = [r["labels"] for r in records]

    mlb = MultiLabelBinarizer(classes=technique_names)
    y_true = mlb.fit_transform(label_sets)
    return texts, y_true, technique_names


def run_baseline_cv(texts: list[str], y_true: np.ndarray, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Out-of-fold predicted probabilities via cross-validation.

    With N this small (see eval/gold.py's DRAFT_GOLD_WARNING), a single
    train/test split would be too noisy to mean anything — every example
    gets evaluated only when held out, and trained on otherwise.
    """
    pipeline = build_pipeline()
    cv = KFold(n_splits=min(n_splits, len(texts)), shuffle=True, random_state=seed)
    y_proba = cross_val_predict(pipeline, texts, y_true, cv=cv, method="predict_proba")
    return y_proba

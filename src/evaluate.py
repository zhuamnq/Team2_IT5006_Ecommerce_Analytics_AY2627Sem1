import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score, roc_auc_score,
)


def classification_metrics(y_true, y_pred, y_score):
    """Precision, recall, F1, ROC-AUC and PR-AUC.

    `y_pred` are 0/1 labels at the chosen threshold; `y_score` the predicted
    probability of the positive (late) class. PR-AUC is the more informative
    of the two AUCs here because only about 7% of orders are late.
    """
    return {
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_score),
        "PR-AUC": average_precision_score(y_true, y_score),
    }


def regression_metrics(y_true, y_pred):
    """MAE, RMSE, R² and bias (mean of prediction - actual; > 0 predicts too slow)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true)),
    }


def cv_summary(cv_results):
    """Mean ± std of each CV metric, as a table for the report.

    Accepts the dict from sklearn's cross_validate (uses its "test_*" keys,
    negated "neg_*" scores are turned back into errors) or {metric: fold scores}.
    """
    rows = {}
    for key, scores in cv_results.items():
        if key in ("fit_time", "score_time") or key.startswith("train_"):
            continue
        name = key.removeprefix("test_")
        scores = np.asarray(scores, dtype=float)
        if name.startswith("neg_"):
            name, scores = name.removeprefix("neg_"), -scores
        rows[name] = {"mean": scores.mean(), "std": scores.std(), "folds": len(scores)}
    table = pd.DataFrame(rows).T
    table["mean ± std"] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(table["mean"], table["std"])]
    return table

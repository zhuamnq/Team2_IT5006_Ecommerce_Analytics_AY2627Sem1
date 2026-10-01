def classification_metrics(y_true, y_pred, y_score):
    """Precision, recall, F1, ROC-AUC and PR-AUC."""
    raise NotImplementedError


def regression_metrics(y_true, y_pred):
    """MAE, RMSE and R²."""
    raise NotImplementedError


def cv_summary(cv_results):
    """Mean ± std of each CV metric, as a table for the report."""
    raise NotImplementedError

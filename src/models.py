# Model families (2–3 total): Linear, Tree-based, optional Ensemble.
# Each family starts from its simplest variant as the baseline.
# Tune with TimeSeriesSplit on the time-sorted training set (src.data.split),
# preprocess with src.features.build_preprocessor inside each pipeline.


def classification_models():
    """Return {name: (estimator, param_grid)} for late-delivery classification."""
    raise NotImplementedError("TODO: LogisticRegression, DecisionTree, RandomForest")


def regression_models():
    """Return {name: (estimator, param_grid)} for delivery-time regression."""
    raise NotImplementedError("TODO: LinearRegression, Ridge, DecisionTree, RandomForest")

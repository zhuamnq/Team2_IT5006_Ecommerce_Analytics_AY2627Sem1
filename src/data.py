import pandas as pd

from src.config import PROCESSED

# Raw tables are joined, cleaned and split in notebooks/02_preprocessing.ipynb,
# which writes these files. data/processed/ is not committed.
PROCESSED_FILES = [
    "late_delivery_classification.csv.gz",
    "delivery_time_regression.csv.gz",
    "order_seller_bridge.csv.gz",
]
SPLITS = ("train", "validation", "test")


def read_processed(name, **read_csv_args):
    """Read one of PROCESSED_FILES, with a hint when it has not been built yet."""
    path = PROCESSED / name
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. Run notebooks/02_preprocessing.ipynb first, e.g.\n"
            "  jupyter nbconvert --to notebook --execute --inplace notebooks/02_preprocessing.ipynb"
        )
    return pd.read_csv(path, **read_csv_args)


def split(frame, target, features=None):
    """Split a feature table on its time-based `split` column.

    Returns {"train": (X, y), "validation": (X, y), "test": (X, y)}, each
    sorted by purchase time so TimeSeriesSplit folds train on earlier orders.
    `features` defaults to every column except IDs, timestamps, `split` and
    the target; pass a feature list (e.g. src.features.feature_columns(...))
    to select columns.
    """
    frame = frame.sort_values("order_purchase_timestamp", kind="stable")
    if features is None:
        drop = {"order_id", "order_purchase_timestamp", "purchase_month", "split", target}
        features = [c for c in frame.columns if c not in drop]
    if target in features:
        raise ValueError(f"The target {target!r} is in the feature list")
    return {
        name: (part[features], part[target])
        for name, part in ((name, frame[frame["split"] == name]) for name in SPLITS)
    }

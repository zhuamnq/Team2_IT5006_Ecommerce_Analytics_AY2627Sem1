import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.config import PROCESSED

# Raw tables are joined, cleaned and split in notebooks/02_preprocessing.ipynb,
# which writes these files. data/processed/ is not committed.
PROCESSED_FILES = [
    "late_delivery_classification.csv.gz",
    "delivery_time_regression.csv.gz",
    "order_seller_bridge.csv.gz",
]
SPLITS = ("train", "validation", "test")
VALIDATION_CUTOFF = pd.Timestamp("2018-07-01")
TEST_CUTOFF = pd.Timestamp("2018-08-01")
DELIVERY_TIME = "order_delivered_customer_date"
METADATA_COLUMNS = {"order_id", "order_purchase_timestamp", "purchase_month", "split", DELIVERY_TIME}


def mature_training_rows(frame, cutoff, splits=("train",)):
    """Eligible labels for one fit; keep the original cohort for later refits."""
    cutoff = pd.Timestamp(cutoff)
    return frame.loc[
        frame["split"].isin(splits)
        & pd.to_datetime(frame["order_purchase_timestamp"]).lt(cutoff)
        & pd.to_datetime(frame[DELIVERY_TIME]).lt(cutoff)
    ].copy()


def training_audit(candidates, training, evaluation, cutoff, stage):
    cutoff = pd.Timestamp(cutoff)
    assert training["order_purchase_timestamp"].lt(cutoff).all()
    assert training[DELIVERY_TIME].notna().all() and training[DELIVERY_TIME].lt(cutoff).all()
    assert set(training.order_id) <= set(candidates.order_id)
    assert set(training.order_id).isdisjoint(evaluation.order_id)
    return {
        "stage": stage, "cutoff": cutoff, "candidate_train_n": len(candidates),
        "immature_excluded": len(candidates) - len(training), "train_n": len(training),
        "score_n": len(evaluation), "latest_training_label": training[DELIVERY_TIME].max(),
    }


def temporal_folds(frame, n_splits=5, selection_cutoff=VALIDATION_CUTOFF):
    """Original purchase windows, purged fit labels and labels known at selection.

    Return positional indices into the supplied, chronologically sorted train cohort.
    Move a boundary to the beginning of any purchase-time tie rather than splitting it.
    Evaluation outcomes may arrive after prediction, but before selection_cutoff.
    """
    purchased = pd.to_datetime(frame["order_purchase_timestamp"])
    delivered = pd.to_datetime(frame[DELIVERY_TIME])
    assert purchased.is_monotonic_increasing
    assert frame["split"].eq("train").all()
    raw = list(TimeSeriesSplit(n_splits=n_splits).split(frame))
    starts = [int(purchased.searchsorted(purchased.iloc[ev[0]], side="left")) for _, ev in raw]
    assert len(set(starts)) == n_splits and starts[0] > 0
    folds, audit = [], []
    for k, (start, end) in enumerate(zip(starts, starts[1:] + [len(frame)]), 1):
        cutoff = purchased.iloc[start]
        assert cutoff < selection_cutoff
        fit = np.flatnonzero(purchased.lt(cutoff) & delivered.lt(cutoff))
        evaluation = np.arange(start, end)
        evaluation = evaluation[delivered.iloc[evaluation].lt(selection_cutoff).to_numpy()]
        assert len(fit) and len(evaluation)
        row = training_audit(frame.iloc[:start], frame.iloc[fit], frame.iloc[evaluation], cutoff, f"CV {k}")
        row["evaluation_labels_unavailable_at_selection"] = end - start - len(evaluation)
        row["selection_cutoff"] = selection_cutoff
        folds.append((fit, evaluation))
        audit.append(row)
    return folds, pd.DataFrame(audit)


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
        drop = METADATA_COLUMNS | {target}
        features = [c for c in frame.columns if c not in drop]
    forbidden = set(features) & (METADATA_COLUMNS | {target})
    if forbidden:
        raise ValueError(f"Target or metadata in the feature list: {sorted(forbidden)}")
    return {
        name: (part[features], part[target])
        for name, part in ((name, frame[frame["split"] == name]) for name in SPLITS)
    }

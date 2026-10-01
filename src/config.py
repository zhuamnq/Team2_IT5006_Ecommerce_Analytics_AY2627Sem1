import random
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parent / "IT5006_Project-Data" / "Olist_CSV"
FIGURES = ROOT / "report" / "figures"
TABLES = ROOT / "report" / "tables"
MODELS = ROOT / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class DeploymentSmokeTest(unittest.TestCase):
    def test_required_artifacts_exist(self):
        required = [
            ROOT / "deployment" / "app.py",
            ROOT / "deployment" / "pages" / "EDA_Explorer.py",
            ROOT / "deployment" / "pages" / "Delivery_Time_Predictor.py",
            ROOT / "deployment" / "models" / "regression_final_model.joblib",
            ROOT / "data" / "app" / "regression_holdout.csv.gz",
            ROOT / "docs" / "report" / "tables" / "regression_selected_features.json",
        ]
        self.assertEqual([], [str(path.relative_to(ROOT)) for path in required if not path.is_file()])

    def test_model_scores_holdout_rows(self):
        try:
            import joblib
            import numpy as np
            import pandas as pd
        except ImportError as error:
            self.skipTest(f"Install requirements.txt to run the model smoke test: {error}")

        features = json.loads(
            (ROOT / "docs" / "report" / "tables" / "regression_selected_features.json").read_text()
        )["tree"]
        columns = features["numeric"] + features["categorical"]
        frame = pd.read_csv(ROOT / "data" / "app" / "regression_holdout.csv.gz", nrows=3)
        for column in features["categorical"]:
            frame[column] = frame[column].astype(object)

        try:
            model = joblib.load(
                ROOT / "deployment" / "models" / "regression_final_model.joblib"
            )
        except Exception as error:
            if "libomp" in str(error):
                self.skipTest("Install libomp to run the XGBoost model smoke test on macOS")
            raise
        predicted_days = frame["promised_delivery_days"].to_numpy() * np.exp(
            model.predict(frame[columns])
        )
        self.assertTrue(np.isfinite(predicted_days).all())
        self.assertTrue((predicted_days > 0).all())


if __name__ == "__main__":
    unittest.main()

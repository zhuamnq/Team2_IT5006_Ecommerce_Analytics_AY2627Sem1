# Deployment

The Streamlit application lives in this directory.

Run the deployed application from the repository root:

```bash
streamlit run deployment/app.py
```

Required deployment artifacts are committed:

- `deployment/models/regression_final_model.joblib`
- `data/app/regression_holdout.csv.gz`
- `docs/report/tables/regression_selected_features.json`
- `docs/report/tables/regression_test_metrics.csv`

Streamlit Community Cloud should use Python 3.12 and `deployment/app.py` as its
entry point. No credentials or secrets are required.

For local macOS runs, install XGBoost's OpenMP runtime with
`brew install libomp` before starting the predictor.

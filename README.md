# Team 2 Olist E-commerce Analytics

End-to-end IT5006 analytics project using the Olist Brazilian e-commerce
dataset. The repository contains exploratory analysis, a three-class delivery
promise experiment, delivery-time regression, and a deployed Streamlit proof
of concept.

**Live application:** https://team2-it5006-dashboard.streamlit.app/

## Project overview

The project supports marketplace customer-success and operations teams through
two analytics problems:

1. **Promise-band classification:** predict whether an order will arrive in
   less than half of its promised window, between half and the full window, or
   late.
2. **Delivery-time regression:** estimate delivery duration relative to the
   promised window and convert it into an expected delivery date.

The tuned XGBoost regression model is deployed because it provides the stronger
proof of concept. On 6,351 held-out August 2018 orders it achieved MAE 2.883
days, RMSE 4.086, and R² 0.112. The revised classification experiment is kept
for analysis but did not meet its provisional business precision/recall
criteria. See [`docs/model_card.md`](docs/model_card.md) for limitations.

## Repository structure

```text
.
├── data/                  Data (see data/README.md)
│   ├── raw/               9 original Olist CSV files (committed, read-only)
│   ├── processed/         Modelling tables rebuilt by the notebooks (git-ignored)
│   ├── app/               Frozen held-out orders used by the deployed predictor
│   └── olist_dashboard.csv.gz   Dataset for the EDA Explorer page
├── notebooks/             EDA, problem scoping, preprocessing, features, modelling
├── src/                   Reusable code and data-preparation scripts
│   └── scripts/           prepare_data.py, export_holdout.py
├── deployment/            Streamlit app and deployment artifacts
│   ├── pages/             EDA Explorer and Delivery Time Predictor
│   ├── models/            Final fitted model loaded by the app
│   └── tests/             Deployment smoke tests
├── docs/                  Project documentation and report outputs
│   └── report/
│       ├── tables/        Result tables written by the notebooks
│       └── figures/       Exported figures
├── requirements.txt       Minimal dependencies for the Streamlit app
├── requirements-modelling.txt  Full dependencies for re-running the notebooks
├── environment.yml        Conda alternative to requirements-modelling.txt
└── README.md
```

## Environment

**Reference version: Python 3.12.** The deployed app and `.python-version` use
3.12, and the full notebook pipeline was re-run end to
end on Python 3.12 with `requirements-modelling.txt` before submission.

Team members executed the notebooks on their own machines, so the saved
notebook metadata records different interpreters (3.9–3.14). The code runs on
3.12; small numeric differences from the committed outputs are described in
[Reproducibility check](#reproducibility-check).

| File | Use it for |
| --- | --- |
| `requirements.txt` | Running the Streamlit app only (minimal; what Streamlit Cloud installs) |
| `requirements-modelling.txt` | Re-running the notebooks (includes `requirements.txt`) |
| `environment.yml` | Same as above, for conda users |

`scikit-learn==1.6.1` and `xgboost==3.2.0` are pinned because the committed
model was saved with these versions and may not load with others.

## Quick start: run the app

On macOS, install the OpenMP runtime required by XGBoost first:

```bash
brew install libomp
```

```bash
git clone https://github.com/zhuamnq/Team2_IT5006_Ecommerce_Analytics_AY2627Sem1.git
cd Team2_IT5006_Ecommerce_Analytics_AY2627Sem1
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run deployment/app.py
```

The app uses committed, frozen artifacts, so the notebooks do not need to be
rerun first. On Streamlit Community Cloud the entry point is
`deployment/app.py` with Python 3.12.

## Data and reproducibility

The nine CSV files in `data/raw/` are the unchanged Olist copy supplied through
the IT5006 Canvas module. Generated modelling tables in `data/processed/` are
ignored by Git and must be rebuilt. See [`data/README.md`](data/README.md) for
the table inventory and data policy.

Rebuild the EDA dashboard dataset:

```bash
python src/scripts/prepare_data.py
```

Install the modelling dependencies (or `conda env create -f environment.yml`):

```bash
python -m pip install -r requirements-modelling.txt
```

Run each problem's notebooks in order:

```text
Shared exploration
  00_eda
  01_problem_scoping

Problem A: promise-band classification
  02_preprocessing_problemA
  03_feature_engineering_problemA
  04_classification_promise_band_modeling

Problem B: delivery-time regression
  02_preprocessing
  03_feature_engineering
  05_regression_delivery_time
```

The two problem chains are independent of each other. On a 2-core machine the full run takes about
80 minutes, most of it in `03_feature_engineering` (~13 min) and
`05_regression_delivery_time` (~55 min, hyperparameter search).

For example:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/02_preprocessing.ipynb
```

All chronological splits, random seeds, label-maturity rules, and leakage
controls are documented in the notebooks and reusable code under `src/`.

### Reproducibility check

Before submission the whole pipeline was re-run from `data/raw/` on Linux with
Python 3.12 and `requirements-modelling.txt`:

| Output | Result of the re-run |
| --- | --- |
| `data/processed/*.csv.gz`, `data/olist_dashboard.csv.gz` | Same rows, columns and values (to floating-point precision) |
| Problem B feature selection (`regression_selected_features.*`) | Identical selected features |
| Problem B Linear, Lasso, Decision Tree and baseline results | Identical |
| Problem B tuned XGBoost (deployed model) | Test MAE 2.858 vs. 2.883 committed. Two hyperparameter candidates have near-identical CV scores, and the platform (macOS vs. Linux XGBoost builds) decides the tie |
| Problem A feature selection and classification results | Close but not identical: two near-tied features swap places in the permutation-importance ranking, so later metrics shift |

The committed tables, model and notebook outputs are the team's original runs
(Problem B: Python 3.12.7, scikit-learn 1.6.1, numpy 1.26.4, pandas 2.2.2,
xgboost 3.2.0, as recorded in `docs/report/tables/regression_best_params.json`).

## Deployed application

The Streamlit app contains:

- **EDA Explorer:** interactive order, revenue, geography, product, delivery,
  and review analysis.
- **Delivery Predictor:** replay of held-out historical orders, what-if
  scoring, held-out performance diagnostics, and batch CSV prediction.

Olist data ends in 2018, so the predictor simulates incoming orders using the
held-out August 2018 cohort rather than connecting to a live order feed. The
deployment layout and required artifacts are documented in
[`deployment/README.md`](deployment/README.md).

To rebuild the frozen deployment holdout after reproducing the regression
pipeline:

```bash
python src/scripts/export_holdout.py
```

## Validation

After installing `requirements.txt`, run:

```bash
python -m unittest discover -s deployment/tests -v
python -m compileall -q deployment src
```

The smoke test checks the deployment artifacts and scores a small held-out
batch with the saved model. On macOS the model test is skipped with a clear
message when `libomp` is not installed.

## Documentation

- [`docs/project_overview.md`](docs/project_overview.md): business problems and end-to-end flow
- [`docs/model_card.md`](docs/model_card.md): deployed model, results, and limitations
- [`docs/feedback_actions.md`](docs/feedback_actions.md): instructor feedback resolution
- [`docs/feature_engineering_plan.md`](docs/feature_engineering_plan.md): Milestone 2 feature engineering plan (English/Chinese)
- [`docs/report_outline.md`](docs/report_outline.md) and `docs/data_preparation_target_construction.tex`: report drafts

## Known limitations

- Results are based on historical Brazilian marketplace data from 2016–2018.
- Temporal drift and right-censoring can affect later-period performance.
- Earlier project iterations inspected the Test period; rerunning cannot
  recreate a completely independent holdout.
- The classification model is not presented as deployment-ready.

## Team and attribution

IT5006 AY2026/27 Semester 1, Team 2.

| Member | GitHub |
| --- | --- |
| Dong Yizhe | `amoeba-aoi` |
| Li Zihui | `lizihui06` |
| Tor Jia Fu | `jeff-tor` |
| Wu Bohan | `Wu-Bohan` |
| Zhuang Mingqiao | `zhuamnq` |

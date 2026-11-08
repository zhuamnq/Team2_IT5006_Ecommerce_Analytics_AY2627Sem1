# Instructor feedback actions

This file maps the main Phase 1 and Phase 2 feedback to the current repository.

| Feedback | Action in the repository | Status |
| --- | --- | --- |
| Add visual evidence for EDA claims | Added `notebooks/00_eda.ipynb` and the interactive EDA Explorer | Done |
| Name stakeholders and success criteria | Added to problem scoping and the classification operating-point analysis | Done |
| Use one classification selection criterion | Feature and model selection use late-class average precision | Done |
| Explain classification features | `03_feature_engineering_problemA.ipynb` contains a feature rationale table | Done |
| Add a 3×3 confusion matrix and diagnose over-prediction | Validation and Test matrices and alert shares are reported in notebook 04 | Done |
| Fix same-day late labels | Notebook 02 compares calendar dates, so same-day delivery is not late | Done |
| Report classification training-fold results | Notebook 04 reports temporal CV fold metrics | Done |
| Apply label-maturity rules to regression | Notebooks 03/05 purge labels unavailable at each cutoff | Done |
| Apply the fold-win rule across model families | Notebook 05 performs the final paired family comparison | Done |
| Strengthen literature synthesis and recent citations | Address in the final report rather than code | Final-report task |

The revised classification experiment still fails its provisional operational
precision and recall targets. This negative result is retained as a limitation
instead of being presented as deployment-ready performance.

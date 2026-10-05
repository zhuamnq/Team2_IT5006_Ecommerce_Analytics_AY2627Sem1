# Milestone 2 Technical Report Outline

Due: Sunday, 11 October 2026, 23:59. Length: 6–8 pages, not counting the cover page, references or appendices.
Submit as a PDF with the GitHub repository link on the cover page.

## Cover page
- Team, members, GitHub link

## 1. Problem statements and stakeholder context
- Problem A: late delivery (classification): stakeholder, target, success criteria
- Problem B: freight cost (regression): stakeholder, target, success criteria

## 2. Data preprocessing and cleaning
- Cleaning rules, missing values, outliers
- Train/test split and leakage controls

## 3. Feature engineering
- Engineered features and why they are available at prediction time

## 4. Models and training process
- Model families used (2–3 in total) and baseline for each family
- Pipelines and random seeds

## 5. Hyperparameter tuning
- Search method, grids, CV scheme

## 6. Results
- Table: classification metrics (Precision, Recall, F1, ROC-AUC, PR-AUC)
- Table: regression metrics (MAE, RMSE, R²)
- Table: cross-validation results (mean ± std)
- Ensemble / stacking results (if used)

## 7. Interpretability
- Feature importance and what it tells us

## 8. Model comparison and final selection
- Justify the chosen model for each problem

## 9. Actionable insights
- Recommendations for the stakeholder

## 10. Limitations and constraints
- Validation month (July 2018) is unrepresentative: late rate 3.4% vs 7.1% in train; models are chosen on time-ordered CV instead
- Test month (August 2018) is right-censored: only deliveries within 46 days could be observed
- Delivery time drifts down over time (train 13.2 days, validation 9.0, test 7.7); Problem B predicts it relative to the promised date
- 13.4% of test orders have a seller not seen in training; seller history falls back to the average

## References

## Appendices

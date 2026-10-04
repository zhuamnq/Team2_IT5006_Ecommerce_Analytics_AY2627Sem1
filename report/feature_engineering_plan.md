# Feature Engineering Plan — Milestone 2 / 特征工程计划 — 第二阶段

[English](#english) · [中文](#中文)

**Revision 3 — 2026-10-04.** Part A and Part B features are built and merged in `src/features.py`. This revision records what was built, what changed from Revision 2, and new findings.
**第 3 版 —— 2026-10-04。** 成员 A 与成员 B 的特征已在 `src/features.py` 中实现并合并。本版记录已实现的内容、相对第 2 版的改动以及新发现。

---

<a id="english"></a>

## English

### 0. What changed in Revision 3

| Area | Change |
|---|---|
| Code | All features are built in `src/features.py`. `load_feature_table("classification")` or `load_feature_table("regression")` returns the cleaned file with every feature added. |
| `est_date_on_weekend` | **Replaced** by `weekend_days_to_estimate`. Olist never promises a weekend date, so the original feature is always 0. |
| `peak_season_flag` | Black Friday week and Carnival only. **Christmas is dropped:** Dec 10–24 late rates are not raised. |
| `platform_orders_7d` | Counted from **all** raw orders, not only delivered ones, so it is no longer an approximation. `platform_load_ratio` added. |
| Seller features | Part B's notebook version was rebuilt to remove leakage (Section 5.5). `seller_backlog_7d`, `seller_past_avg_freight` and `route_past_late_rate` were missing and are now built. |
| Missing values | No `add_indicator`; `geolocation_missing` and `payment_missing` already flag every gap. |
| Interaction term | `distance × remote_flag` dropped: after `log1p` it equals `remote_flag` (correlation 1.00). |
| File names | The cleaned files are `.csv.gz`. |

### 1. Context

| | Problem A | Problem B |
|---|---|---|
| Question | Will the order arrive after the promised date? | How many days will delivery take? |
| Type | Classification | Regression |
| Target | `is_late` (6.8% late overall) | `delivery_days` |
| File | `late_delivery_classification.csv.gz` | `delivery_time_regression.csv.gz` |

- **Problem B has changed from freight cost to delivery time.** `freight_value` is now an **input feature**, known at purchase. The notebooks and the report outline still describe freight as Problem B and need updating.
- Every delivery is **domestic (Brazil)**. The useful distinction is interstate vs intrastate: 63.8% of items go to another state.
- Locations are known only by **5-digit zip prefix**, which is accurate to roughly 1–4 km.
- Distance is straight-line (haversine) distance between zip-prefix centroids. It is free and already in the files. Google Distance Matrix (about US$450–500) and self-hosted OSRM road distance are not needed.

### 2. Decisions

| # | Decision | Status |
|---|---|---|
| D1 | Prediction moment | ✅ **At purchase.** `order_approved_at` and `shipping_limit_date` are not used. |
| D2 | Row granularity | ✅ **One row per order** (96,454 delivered orders) |
| D3 | Train/test split | ✅ **Time-based:** train up to 2018-06-30 (83,947 rows), validation 2018-07 (6,156), test 2018-08 (6,351). See issue 9.1. |
| D4 | Distance method | ✅ **Haversine**, already in the files as min/mean/max over the order's sellers |
| D5 | Targets | ✅ **A = `is_late`, B = `delivery_days`** |
| — | `is_late` compares calendar dates, not timestamps | ✅ Done; checked against the raw data, 100% match |

### 3. The cleaned datasets

Produced by `notebooks/02_preprocessing.ipynb` in `data/processed/` (gitignored).

| File | Shape | Contents |
|---|---|---|
| `late_delivery_classification.csv.gz` | 96,454 × 52 | One row per order: 49 feature/ID columns + `split` + target `is_late` |
| `delivery_time_regression.csv.gz` | 96,454 × 52 | Same orders, same columns, target `delivery_days` |
| `order_seller_bridge.csv.gz` | 97,795 × 13 | One row per **order and seller**: `seller_id`, seller city/state/zip/coordinates, `seller_item_count`, `seller_order_value`, `seller_freight_value`, `estimated_seller_customer_distance_km`, `order_purchase_timestamp`, `split` |

`load_feature_table` adds 20 engineered columns, giving 96,454 × 72. Both merged tables can be saved as `features_late_delivery.csv.gz` and `features_delivery_time.csv.gz` in the same folder.

Checks that passed against the raw Olist data:
- The bridge's item counts, values and freight match the raw order items exactly. No order–seller pair appears twice, and its `split` matches the order-level files.
- `primary_seller_state` is the state of the seller of the first item (`order_item_id == 1`).
- `volumetric_weight_kg`, `billable_weight_kg`, `density_g_cm3` and `mean_item_price` recompute exactly.

**Leakage:** never use `is_late` as a feature for Problem B, nor `delivery_days` for Problem A. Also excluded: the delivered-to-carrier and delivered-to-customer dates, `delay_days`, `delivery_status` and `review_score`. All are listed in `LEAKAGE_COLUMNS`, and neither feature list contains any of them.

### 4. Feature catalogue: 47 core features, 43 available

✅ = already a column · 🔧 = built in `src/features.py` · ❌ = not available · ★ = priority feature

**G1 — Promise and timing (5): 1 ✅ · 1 🔧 · 3 ❌**

| Feature | Column / how | Status |
|---|---|---|
| `promised_days` ★ | `promised_delivery_days` | ✅ |
| `promised_days_per_km` | `promised_delivery_days / (estimated_mean_distance_km + 1)`; +1 km because 23 orders have a distance of 0 | 🔧 |
| `approval_lag_hours` | needs `order_approved_at` | ❌ excluded by D1 |
| `handling_window_days` | needs `shipping_limit_date` | ❌ excluded by D1 |
| `transit_budget_days` | needs `shipping_limit_date` | ❌ excluded by D1 |

**G2 — Geography and distance (8): 8 ✅**

| Feature | Column / how | Status |
|---|---|---|
| `haversine_km` ★ | `estimated_min/mean/max_distance_km` | ✅ |
| `same_state` | `same_state` (= **all** sellers in the customer's state) and `any_seller_same_state` | ✅ |
| `same_city` | `same_city`, `any_seller_same_city` | ✅ |
| `region_pair` ★ | `region_pair` (23 combinations occur), plus `customer_region` and `primary_seller_region` | ✅ |
| `customer_is_capital` | `customer_is_capital` | ✅ |
| `customer_zip_density` | `customer_zip_density` | ✅ |
| `remote_flag` ★ | `remote_flag` (AC, AM, AP, PA, RO, RR) | ✅ |
| `geo_imputed_flag` | `geolocation_missing` | ✅ |

Optional, not built: `road_km`, `bearing`.

**G3 — Product (8): 8 ✅**

| Feature | Column | Status |
|---|---|---|
| `weight_g` | `total_weight_g` | ✅ |
| `volume_cm3` | `total_volume_cm3` | ✅ |
| `volumetric_weight` | `volumetric_weight_kg` | ✅ |
| `billable_weight` | `billable_weight_kg` | ✅ |
| `density` | `density_g_cm3` (84 training orders have impossible values above 5 g/cm³, up to 66) | ✅ |
| `longest_side` | `longest_side_cm` | ✅ |
| `sum_of_dims` | `max_sum_of_dims_cm` | ✅ |
| `category` ★ | `product_category` (72 values), plus `product_category_count` | ✅ |

Optional, not built: `photos_qty`, `name_length`, `description_length`.

**G4 — Order composition (8): 8 ✅**

| Feature | Column | Status |
|---|---|---|
| `n_items` ★ | `item_count` | ✅ |
| `n_distinct_products` | `unique_product_count` | ✅ |
| `n_sellers` | `seller_count`, plus `seller_state_count` | ✅ |
| `multi_seller_flag` | `multi_seller_flag` | ✅ |
| `total_order_weight` | `total_weight_g` (**same column as `weight_g`**) | ✅ |
| `max_item_weight` | `max_item_weight_g` | ✅ |
| `price` | `mean_item_price`, `max_item_price` | ✅ |
| `order_total_price` | `order_value` | ✅ |

Also available as an input: `freight_value` (order total; not counted in the 47).

**G5 — Payment (4): 4 ✅**

| Feature | Column | Status |
|---|---|---|
| `payment_type` ★ | `payment_type` | ✅ |
| `installments` | `payment_installments` | ✅ |
| `n_payment_methods` | `payment_type_count` | ✅ |
| `used_voucher` | `used_voucher` (3.8%; includes vouchers used alongside another payment method) | ✅ |

Extra, from Part B: `payment_per_installment` = `payment_value / payment_installments` (🔧; 0 installments → missing).

**G6 — Seller history (7): 6 🔧 · 1 ❌.** All built from the bridge (see Section 5).

| Feature | Column / how | Status |
|---|---|---|
| `seller_past_orders` | the seller's orders placed before this one | 🔧 |
| `seller_backlog_7d` ★ | the seller's orders in the 7 days before purchase | 🔧 |
| `seller_is_new` | the seller's first order was less than 30 days earlier | 🔧 |
| `seller_past_avg_freight` | mean `seller_freight_value` of earlier orders (missing for a seller's first order) | 🔧 |
| `seller_past_late_rate` ★ | training orders delivered before purchase, smoothed. Also `seller_past_delivery_days` (same rules, for `delivery_days`) | 🔧 |
| `route_past_late_rate` | seller state → customer state, same rules | 🔧 |
| `seller_avg_handling_days` | needs the raw delivered-to-carrier date | ❌ optional raw join |

Extra: `seller_past_deliveries`, the number of training deliveries behind the rates. 0 = no history, which is used for the known vs new seller comparison.

**G7 — Calendar and platform load (7): 1 ✅ · 6 🔧**

| Feature | Column / how | Status |
|---|---|---|
| `purchase_month` ★ | `purchase_month_of_year` (1–12), plus `purchase_month_sin` / `purchase_month_cos`. The `"YYYY-MM"` text is not used. | ✅ (converted) |
| `purchase_weekday` | `purchase_weekday` (Monday = 0) | 🔧 |
| `purchase_hour` | `purchase_hour` | 🔧 |
| `days_to_holiday` ★ | days to the next national holiday (`BR_HOLIDAYS` in `src/config.py`; 0 on the holiday) | 🔧 |
| `peak_season_flag` | Black Friday week (Monday to Cyber Monday) and Carnival (Saturday to Ash Wednesday), in `PEAK_SEASONS`. Christmas left out. | 🔧 |
| `est_date_on_weekend` → `weekend_days_to_estimate` | weekend days between purchase and the promised date | 🔧 (replaced) |
| `platform_orders_7d` | orders in the 7 days before purchase, from **all** raw orders. Also `platform_load_ratio` = 7-day count ÷ average week over 28 days. | 🔧 |

Analysis only, not used in the model: `strike_period_flag` (truckers' strike, May 2018).

**Totals**

| | ✅ In the files | 🔧 Built | ❌ Not available | Total |
|---|---|---|---|---|
| Count | 30 | 13 | 4 | 47 |

`weight_g` and `total_order_weight` are the same column, so there are **46 distinct core features** and **42 distinct available features**. The extras (`purchase_month_sin/cos`, `platform_load_ratio`, `payment_per_installment`, `seller_past_delivery_days`, `seller_past_deliveries`) bring the engineered columns to **20**.

### 5. Seller data: kept out of the order files, joined later through the bridge

**Approach.** The order-level files stay small and contain **no `seller_id`**. Seller features are calculated from `order_seller_bridge.csv.gz` **after cleaning and before training**, reduced to one row per order, and joined on `order_id` (`add_seller_features`).

**Why later:** the strongest seller features use past outcomes (late rate, mean delivery days). Calculated over the whole dataset, they would include validation and test outcomes. Built in the training stage, they are fitted on training data only.

**5.1 Pipeline (as built)**

```
1. Order-level file (is_late or delivery_days)               ← notebooks/02_preprocessing.ipynb
2. add_features: Part A columns
3. add_seller_features, on the bridge (97,795 rows):
     a. Point-in-time (purchase times only; every split may use every earlier order):
        seller_past_orders, seller_backlog_7d, seller_is_new, seller_past_avg_freight
     b. Outcome-based (TRAINING orders only):
        seller_past_late_rate, seller_past_delivery_days, route_past_late_rate
        - history = training orders delivered BEFORE this order's purchase
          (delivered time = purchase + delivery_days), for EVERY split
        - smoothing: (sum + m × prior) / (n + m), m = 10, prior = training average
        - no history → the prior
        - "late" = is_late (calendar dates), the same as the target
4. Reduce to one row per order (aggregation rules in 5.2)
5. Merge on order_id with validate="one_to_one"; assert 96,454 rows
6. Use the existing split column (no re-splitting)
```

Steps 1–5 are a single call to `load_feature_table(problem)`. Still to do at the modelling stage: tune with `TimeSeriesSplit` on train, and evaluate known vs new sellers separately (`seller_past_deliveries == 0`).

**5.2 Aggregation rules for multi-seller orders**

1,275 orders have more than one seller, so the bridge has 97,795 rows for 96,454 orders. The order counts as delivered when its **last** parcel arrives, so aggregate towards the worst case:

| Seller feature | Rule per order |
|---|---|
| `seller_past_late_rate`, `seller_past_delivery_days`, `route_past_late_rate` | **max** |
| `seller_backlog_7d` | **max** |
| `seller_past_orders`, `seller_past_deliveries` | **min** (least experienced seller) |
| `seller_is_new` | **any** |
| `seller_past_avg_freight` | **mean** |

**5.3 Join rules** (all enforced in `add_seller_features`)
1. Aggregate to one row per order **before** merging; merging the bridge directly duplicates multi-seller orders.
2. The row count must stay at **96,454** after every join.
3. The join is code (`src/features.py`), not a one-off notebook cell.

**5.4 Sellers not seen in training**

| Split | Orders whose seller has no training history | New sellers |
|---|---|---|
| Validation | 6.6% (6.8% by `seller_past_deliveries == 0`) | 184 of 1,238 |
| Test | **13.4%** | 256 of 1,261 |

42% of training sellers have fewer than 5 orders. Smoothing and the fallback are **required**; list this as a limitation in the report.

**5.5 Fixes to the Part B notebook version** (`notebooks/03_feature_engineering_b.ipynb`)

| Problem in the notebook | Effect | Fix in `src/features.py` |
|---|---|---|
| History included earlier orders **not yet delivered** at purchase time | 8.5% of history entries, 87% of training rows; seller late-rate training correlation with `is_late` inflated from 0.031 to 0.097 | History uses only orders delivered before purchase |
| Validation/test used a profile of all training orders, delivered up to 2018-10-17 | Outcomes after the purchase date | Same delivered-before rule for every split |
| "Late" = `delivery_days > promised_delivery_days` | Disagrees with `is_late` on 776 orders | Uses `is_late` |
| Training rows used a running history; validation/test used a fixed profile | Features mean different things by split | One definition for every split |
| Late rate not smoothed | Noisy rates for small sellers | Smoothed like delivery days |
| Missing `seller_backlog_7d`, `seller_past_avg_freight`, `route_past_late_rate` | — | Built |
| Duplicate columns (see 5.6) | — | Not carried over |
| Windows paths, notebook only | Others cannot run it | Code in `src/features.py` |

Verified: scrambling validation/test outcomes leaves every seller feature unchanged, and a brute-force recomputation on 300 random orders matches exactly.

**5.6 Part B columns not carried over (overlaps)**

| Column | Reason |
|---|---|
| `purchase_weekday`, `purchase_hour`, `purchase_month` (overwritten as a number) | Identical to Part A's columns |
| `purchase_is_weekend` | = `purchase_weekday ≥ 5` |
| `multiple_payment_types` | Identical to `payment_type_count > 1` |
| `installment_flag` | = `payment_installments > 1` |
| `purchase_year` | Validation and test are entirely 2018, so it only measures growth over time |
| `mean_*` / `max_*` pairs, history item count and order value | 0.99+ correlated with each other (98.7% of orders have one seller) |
| `seller_hist_std_delivery_days` | Not in the plan; no signal on the test month |

### 6. Using Lasso: what changes

Lasso is a linear model with an L1 penalty. Compared with tree models it cannot handle missing values, it is sensitive to feature scale, and it cannot learn non-linear effects. Among strongly correlated features it keeps one more or less arbitrarily.

**6.1 Where Lasso fits**

| Problem | Linear family (baseline → regularised) |
|---|---|
| B: `delivery_days` (regression) | `LinearRegression` → `Ridge` → **`Lasso`** (optionally `ElasticNet`) |
| A: `is_late` (classification) | `LogisticRegression` → **L1 logistic regression** (`penalty="l1"`, `solver="saga"`, `class_weight="balanced"`) |

The plan stays within 2–3 model families (linear + tree-based). Models have not been chosen yet; `src/models.py` is still a placeholder.

**6.2 Preprocessing** (built: `build_preprocessor`, fitted on training data only)

| Requirement | Action (as built) |
|---|---|
| Scaling | `StandardScaler` on numeric columns; `scale=False` for tree models |
| Missing values (order file: 477 distances, 265 customer coordinates, 213 seller coordinates, 1 payment row; engineered: 2,881 `seller_past_avg_freight`, 477 `promised_days_per_km`) | `SimpleImputer(strategy="median")`. **No `add_indicator`**: `geolocation_missing` and `payment_missing` flag every gap, and the indicators would add about six near-identical columns |
| Skewed values (distance, weight, volume, price, freight, counts, seller history counts) | `log1p` before scaling |
| Categories (`product_category` 72, `region_pair` 23, `customer_state` 27, `payment_type`, `purchase_weekday`) | `OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20)`. Infrequent in training: 6 product categories, `N_to_NE/S/SE`, payment `Unknown` |
| `purchase_month` as `"YYYY-MM"` (test months never appear in training) | Replaced by month of year as sin/cos |
| Non-linear calendar effects | Fixed bands, one-hot encoded: `days_to_holiday` (0–7, 8–14, 15–30, 31–60, 61+), `purchase_hour` (night, morning, afternoon, evening) |
| `delivery_days` is right-skewed | Try `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)` (modelling stage) |

**6.3 Redundant columns removed for linear models** (`FEATURES_LINEAR`; `FEATURES_TREE` keeps everything)

| Redundant set | Kept in `FEATURES_LINEAR` |
|---|---|
| `total_weight_g` (also listed as `weight_g`) | Once |
| `total_volume_cm3`, `volumetric_weight_kg` (= volume / 6000) | `total_volume_cm3` |
| `density_g_cm3` | Dropped: = weight / volume, and has impossible values |
| `estimated_min/mean/max_distance_km` | `mean` |
| `payment_value` ≈ `order_value` + `freight_value`; `payment_per_installment` | Both dropped |
| `seller_count`, `multi_seller_flag`, `seller_state_count` | `multi_seller_flag` |
| `same_state`, `any_seller_same_state`, `region_pair` | `same_state`, `region_pair` |
| `same_city`, `any_seller_same_city` | `same_city` |
| `mean_item_price` × `item_count` = `order_value` | `order_value`, `item_count` |
| `customer_region`, `primary_seller_region`, `primary_seller_state` | Dropped; `region_pair` and `customer_state` cover them |
| Raw coordinates (`*_zip_lat/lng`) | Dropped; distance and region already describe them |
| `promised_days_per_km` | Dropped: after logs it is log(promised days) − log(distance), both already in |
| `weekend_days_to_estimate` | Dropped: 0.97 correlated with `promised_delivery_days` |
| `platform_orders_7d` | Dropped: grows ~40% by the test month; `platform_load_ratio` kept |
| `seller_past_orders`, `seller_past_deliveries` (0.998) | `seller_past_orders` |

Remaining in the linear list: the three weight/volume measures correlate 0.78–0.91.

**6.4 Non-linear terms for Lasso:** distance bands (< 100 / 100–500 / 500–1,500 / > 1,500 km) are built. **`distance × remote_flag` is dropped**: every remote order is 2,000–3,000 km away, so after `log1p` the term equals `remote_flag` (correlation 1.00). `billable_weight_kg` is already non-linear.

**6.5 Tuning:** `alpha` in `np.logspace(-4, 1, 30)` and `C` in `np.logspace(-3, 2, 30)`, using `GridSearchCV` with `cv=TimeSeriesSplit(5)`.

**6.6 Interpretability:** report how many features keep a non-zero coefficient and the largest standardised coefficients. Compare with tree feature importance (report section 7), and check that selection is stable across folds.

### 7. Work split

| | Person A: calendar, preprocessing and Lasso | Person B: seller features from the bridge |
|---|---|---|
| Features | ✅ 8 built (`est_date_on_weekend` replaced by `weekend_days_to_estimate`) | ✅ 6 built, merged into `src/features.py` with the fixes in 5.5 |
| Other work | ✅ `build_preprocessor`; ✅ `FEATURES_LINEAR` / `FEATURES_TREE`; ✅ bands; ✅ missing-value handling; ⏳ Lasso (models not chosen yet) | ✅ smoothing and fallback; ✅ aggregation (5.2); ✅ join and row-count check; ⏳ known vs new seller evaluation (modelling stage) |

**Not assigned to anyone yet:** tree models (DecisionTree, RandomForest) on `FEATURES_TREE`.

### 8. Priority features if time runs short (14, marked ★)

`promised_days`, `haversine_km`, `region_pair`, `remote_flag`, `category`, `n_items`, `payment_type`, `purchase_month` (as month of year), `days_to_holiday`, `seller_backlog_7d`, `seller_past_late_rate`, plus three already in the files that need no work: `customer_is_capital`, `billable_weight_kg`, `freight_value`.

All 14 are now built. Data checks suggest revisiting three of them (see 9.6–9.8): `purchase_month` and `seller_past_late_rate` are weak on the test month, and `days_to_holiday` covers different values there. `platform_orders_7d` (correlation 0.12 with `is_late` in training) and `seller_past_delivery_days` (0.21–0.27 with `delivery_days` on every split) are stronger candidates.

### 9. Open issues

1. **The validation month is unrepresentative.** Late rate: train 7.1%, **validation 3.4%** (July 2018), test 6.2%. Do not tune the decision threshold or choose models on validation alone; use `TimeSeriesSplit` on train, or merge validation into train.
2. **`same_state` changed meaning.** It is now 1 only if **all** sellers are in the customer's state (`any_seller_same_state` = at least one), which changes 126 orders compared with the first file. Document this in the report.
3. **Problem B wording.** `notebooks/01_problem_scoping.ipynb`, `notebooks/05_regression_freight.ipynb` and `report/report_outline.md` still describe freight cost as Problem B.
4. ~~`purchase_month` must be converted before modelling.~~ Done (Section 4, G7).
5. **`seller_avg_handling_days`** would need the raw delivered-to-carrier date. It is optional; drop it unless there is time.
6. **Seller and route late rates do not carry over to the test month.** Correlation with `is_late`:

   | | Train | Validation | Test |
   |---|---|---|---|
   | `seller_past_late_rate` | 0.031 | 0.045 | 0.004 |
   | `route_past_late_rate` | 0.103 | 0.039 | **−0.099** |
   | `seller_past_delivery_days` → `delivery_days` | 0.209 | 0.270 | 0.238 |

   Routes that were often late during the Feb–Mar 2018 spike (14–19% late) were on time in August. State this as a limitation; delivery-day history is reliable, late rates are not.
7. **Calendar features see different values in validation and test.** `peak_season_flag` is 0 for every validation and test row, so its effect (15.3% vs 6.6% late in training) cannot be checked there. `days_to_holiday`: validation is entirely 31+ days, test has no 0–7 or 61+ days.
8. **Month of year is mostly year-specific.** Each month appears in training for only one or two years; August is in training only as August 2017 (2.9% late) against 6.2% in the test month.
9. **The Part B notebook** still contains the leaky version and Windows paths. Replace its body with `load_feature_table`.
10. **Environment:** `requirements.txt` asks for `pandas<3`; the code also runs on pandas 3.0.3.

---

<a id="中文"></a>

## 中文

### 0. 第 3 版改动

| 方面 | 改动 |
|---|---|
| 代码 | 所有特征都在 `src/features.py` 中实现。`load_feature_table("classification")` 或 `load_feature_table("regression")` 返回加上全部特征的清洗后文件。 |
| `est_date_on_weekend` | **已替换**为 `weekend_days_to_estimate`。Olist 承诺的送达日期从不落在周末，原特征恒为 0。 |
| `peak_season_flag` | 只包括黑色星期五当周与狂欢节。**去掉圣诞节：** 12 月 10–24 日的延迟率没有升高。 |
| `platform_orders_7d` | 改为基于**全部**原始订单计数，不再只是已送达订单的近似值。另加 `platform_load_ratio`。 |
| 卖家特征 | 成员 B 的 notebook 版本已重建以消除数据泄漏（见 5.5）。原先缺少的 `seller_backlog_7d`、`seller_past_avg_freight` 和 `route_past_late_rate` 现已实现。 |
| 缺失值 | 不使用 `add_indicator`；`geolocation_missing` 与 `payment_missing` 已标记所有缺失。 |
| 交互项 | 去掉 `distance × remote_flag`：`log1p` 后与 `remote_flag` 完全相同（相关系数 1.00）。 |
| 文件名 | 清洗后的文件为 `.csv.gz`。 |

### 1. 背景

| | 问题 A | 问题 B |
|---|---|---|
| 问题 | 订单是否会晚于承诺日期送达？ | 送达需要多少天？ |
| 类型 | 分类 | 回归 |
| 目标变量 | `is_late`（整体延迟率 6.8%） | `delivery_days` |
| 文件 | `late_delivery_classification.csv.gz` | `delivery_time_regression.csv.gz` |

- **问题 B 已由运费预测改为送达时长预测。** `freight_value` 现在是**输入特征**（下单时已知）。notebook 与报告大纲仍把运费写作问题 B，需要更新。
- 所有配送均为**巴西国内配送**。有意义的区分是跨州 vs 州内：63.8% 的商品跨州配送。
- 位置只精确到 **5 位邮编前缀**，误差约 1–4 公里。
- 距离为邮编前缀中心点之间的直线距离（Haversine），免费且已在文件中。不需要 Google Distance Matrix（约 450–500 美元）或自建 OSRM 道路距离。

### 2. 决策

| # | 决策 | 状态 |
|---|---|---|
| D1 | 预测时间点 | ✅ **下单时。** 不使用 `order_approved_at` 和 `shipping_limit_date`。 |
| D2 | 数据粒度 | ✅ **每个订单一行**（96,454 个已送达订单） |
| D3 | 训练/测试集划分 | ✅ **按时间划分：** 训练集截至 2018-06-30（83,947 行），验证集 2018-07（6,156 行），测试集 2018-08（6,351 行）。见问题 9.1。 |
| D4 | 距离计算方法 | ✅ **Haversine**，文件中已有按订单内卖家计算的最小/平均/最大距离 |
| D5 | 目标变量 | ✅ **A = `is_late`，B = `delivery_days`** |
| — | `is_late` 按日期而非时间戳比较 | ✅ 已完成；与原始数据核对，100% 一致 |

### 3. 清洗后的数据集

由 `notebooks/02_preprocessing.ipynb` 生成于 `data/processed/`（已被 gitignore）。

| 文件 | 形状 | 内容 |
|---|---|---|
| `late_delivery_classification.csv.gz` | 96,454 × 52 | 每个订单一行：49 个特征/ID 列 + `split` + 目标 `is_late` |
| `delivery_time_regression.csv.gz` | 96,454 × 52 | 相同订单、相同列，目标为 `delivery_days` |
| `order_seller_bridge.csv.gz` | 97,795 × 13 | 每个**订单 + 卖家**一行：`seller_id`、卖家城市/州/邮编/坐标、`seller_item_count`、`seller_order_value`、`seller_freight_value`、`estimated_seller_customer_distance_km`、`order_purchase_timestamp`、`split` |

`load_feature_table` 新增 20 个工程特征列，得到 96,454 × 72。两张合并表可另存为同一文件夹下的 `features_late_delivery.csv.gz` 和 `features_delivery_time.csv.gz`。

与原始 Olist 数据核对通过的项目：
- 桥接表的商品数、金额和运费与原始订单商品表完全一致；没有重复的订单–卖家组合；其 `split` 与订单级文件一致。
- `primary_seller_state` 为第一件商品（`order_item_id == 1`）卖家所在的州。
- `volumetric_weight_kg`、`billable_weight_kg`、`density_g_cm3` 和 `mean_item_price` 均可准确复算。

**数据泄漏：** 问题 B 不能使用 `is_late` 作为特征，问题 A 也不能使用 `delivery_days`。同样排除：交给承运商日期与送达买家日期、`delay_days`、`delivery_status`、`review_score`。这些列都在 `LEAKAGE_COLUMNS` 中，两份特征列表均不包含它们。

### 4. 特征清单：47 个核心特征，43 个可用

✅ = 已有列 · 🔧 = 已在 `src/features.py` 中实现 · ❌ = 不可用 · ★ = 优先特征

**G1 —— 承诺时效与时间（5 个）：1 ✅ · 1 🔧 · 3 ❌**

| 特征 | 对应列 / 计算方式 | 状态 |
|---|---|---|
| `promised_days` ★ | `promised_delivery_days` | ✅ |
| `promised_days_per_km` | `promised_delivery_days / (estimated_mean_distance_km + 1)`；加 1 公里是因为有 23 个订单距离为 0 | 🔧 |
| `approval_lag_hours` | 需要 `order_approved_at` | ❌ 因 D1 排除 |
| `handling_window_days` | 需要 `shipping_limit_date` | ❌ 因 D1 排除 |
| `transit_budget_days` | 需要 `shipping_limit_date` | ❌ 因 D1 排除 |

**G2 —— 地理与距离（8 个）：8 ✅**

| 特征 | 对应列 / 计算方式 | 状态 |
|---|---|---|
| `haversine_km` ★ | `estimated_min/mean/max_distance_km` | ✅ |
| `same_state` | `same_state`（= **所有**卖家都与买家同州）及 `any_seller_same_state` | ✅ |
| `same_city` | `same_city`、`any_seller_same_city` | ✅ |
| `region_pair` ★ | `region_pair`（实际出现 23 种组合），另有 `customer_region`、`primary_seller_region` | ✅ |
| `customer_is_capital` | `customer_is_capital` | ✅ |
| `customer_zip_density` | `customer_zip_density` | ✅ |
| `remote_flag` ★ | `remote_flag`（AC、AM、AP、PA、RO、RR） | ✅ |
| `geo_imputed_flag` | `geolocation_missing` | ✅ |

可选、未构建：`road_km`、`bearing`。

**G3 —— 商品（8 个）：8 ✅**

| 特征 | 对应列 | 状态 |
|---|---|---|
| `weight_g` | `total_weight_g` | ✅ |
| `volume_cm3` | `total_volume_cm3` | ✅ |
| `volumetric_weight` | `volumetric_weight_kg` | ✅ |
| `billable_weight` | `billable_weight_kg` | ✅ |
| `density` | `density_g_cm3`（训练集中有 84 个订单的密度不合理，超过 5 g/cm³，最高 66） | ✅ |
| `longest_side` | `longest_side_cm` | ✅ |
| `sum_of_dims` | `max_sum_of_dims_cm` | ✅ |
| `category` ★ | `product_category`（72 个取值），另有 `product_category_count` | ✅ |

可选、未构建：`photos_qty`、`name_length`、`description_length`。

**G4 —— 订单构成（8 个）：8 ✅**

| 特征 | 对应列 | 状态 |
|---|---|---|
| `n_items` ★ | `item_count` | ✅ |
| `n_distinct_products` | `unique_product_count` | ✅ |
| `n_sellers` | `seller_count`，另有 `seller_state_count` | ✅ |
| `multi_seller_flag` | `multi_seller_flag` | ✅ |
| `total_order_weight` | `total_weight_g`（**与 `weight_g` 是同一列**） | ✅ |
| `max_item_weight` | `max_item_weight_g` | ✅ |
| `price` | `mean_item_price`、`max_item_price` | ✅ |
| `order_total_price` | `order_value` | ✅ |

另有可用输入：`freight_value`（订单运费合计；不计入 47 个）。

**G5 —— 支付（4 个）：4 ✅**

| 特征 | 对应列 | 状态 |
|---|---|---|
| `payment_type` ★ | `payment_type` | ✅ |
| `installments` | `payment_installments` | ✅ |
| `n_payment_methods` | `payment_type_count` | ✅ |
| `used_voucher` | `used_voucher`（3.8%；包括与其他支付方式同时使用的优惠券） | ✅ |

额外特征（来自成员 B）：`payment_per_installment` = `payment_value / payment_installments`（🔧；分期数为 0 时记为缺失）。

**G6 —— 卖家历史（7 个）：6 🔧 · 1 ❌。** 全部由桥接表构建（见第 5 节）。

| 特征 | 对应列 / 计算方式 | 状态 |
|---|---|---|
| `seller_past_orders` | 该卖家在本订单之前的订单数 | 🔧 |
| `seller_backlog_7d` ★ | 下单前 7 天内该卖家的订单数 | 🔧 |
| `seller_is_new` | 卖家首单距今不足 30 天 | 🔧 |
| `seller_past_avg_freight` | 更早订单的 `seller_freight_value` 平均值（卖家首单时缺失） | 🔧 |
| `seller_past_late_rate` ★ | 下单前已送达的训练集订单，经平滑。另有 `seller_past_delivery_days`（相同规则，针对 `delivery_days`） | 🔧 |
| `route_past_late_rate` | 卖家州 → 买家州，相同规则 | 🔧 |
| `seller_avg_handling_days` | 需要原始数据中的交给承运商日期 | ❌ 可选的原始数据连接 |

额外特征：`seller_past_deliveries`，即历史比率所依据的训练集送达订单数。0 = 无历史，用于比较已知卖家与新卖家。

**G7 —— 日历与平台负载（7 个）：1 ✅ · 6 🔧**

| 特征 | 对应列 / 计算方式 | 状态 |
|---|---|---|
| `purchase_month` ★ | `purchase_month_of_year`（1–12），另有 `purchase_month_sin` / `purchase_month_cos`。不再使用 `"YYYY-MM"` 文本。 | ✅（已转换） |
| `purchase_weekday` | `purchase_weekday`（周一 = 0） | 🔧 |
| `purchase_hour` | `purchase_hour` | 🔧 |
| `days_to_holiday` ★ | 距下一个全国性节假日的天数（`src/config.py` 中的 `BR_HOLIDAYS`；节假日当天为 0） | 🔧 |
| `peak_season_flag` | 黑色星期五当周（周一至网络星期一）与狂欢节（周六至圣灰星期三），见 `PEAK_SEASONS`。不含圣诞节。 | 🔧 |
| `est_date_on_weekend` → `weekend_days_to_estimate` | 下单日到承诺日期之间的周末天数 | 🔧（已替换） |
| `platform_orders_7d` | 下单前 7 天的订单量，基于**全部**原始订单。另有 `platform_load_ratio` = 7 天订单量 ÷ 过去 28 天的周平均。 | 🔧 |

仅用于分析、不放入模型：`strike_period_flag`（2018 年 5 月卡车司机罢工）。

**合计**

| | ✅ 已有 | 🔧 已实现 | ❌ 不可用 | 合计 |
|---|---|---|---|---|
| 数量 | 30 | 13 | 4 | 47 |

`weight_g` 与 `total_order_weight` 是同一列，因此实际为 **46 个不同的核心特征**、**42 个不同的可用特征**。加上额外特征（`purchase_month_sin/cos`、`platform_load_ratio`、`payment_per_installment`、`seller_past_delivery_days`、`seller_past_deliveries`），工程特征列共 **20 个**。

### 5. 卖家数据：不放入订单级文件，之后通过桥接表连接

**做法。** 订单级文件保持精简，**不含 `seller_id`**。卖家特征在**清洗之后、训练之前**由 `order_seller_bridge.csv.gz` 计算，归约为每个订单一行后，以 `order_id` 连接（`add_seller_features`）。

**为什么之后再连接：** 最强的卖家特征依赖历史结果（延迟率、平均送达天数）。若基于全量数据计算，会包含验证集与测试集的结果；在训练阶段构建，则只基于训练集拟合。

**5.1 流程（已实现）**

```
1. 订单级文件（is_late 或 delivery_days）                     ← notebooks/02_preprocessing.ipynb
2. add_features：成员 A 的特征列
3. add_seller_features，在桥接表（97,795 行）上计算：
     a. 时点特征（只用下单时间；所有数据集都可使用所有更早的订单）：
        seller_past_orders、seller_backlog_7d、seller_is_new、seller_past_avg_freight
     b. 基于结果的特征（只用训练集订单）：
        seller_past_late_rate、seller_past_delivery_days、route_past_late_rate
        - 历史 = 在本订单下单之前已送达的训练集订单
          （送达时间 = 下单时间 + delivery_days），对所有数据集一致
        - 平滑：(合计 + m × 先验) / (n + m)，m = 10，先验 = 训练集平均值
        - 无历史 → 使用先验值
        - "延迟" = is_late（按日期比较），与目标变量一致
4. 归约为每个订单一行（聚合规则见 5.2）
5. 以 order_id 合并，validate="one_to_one"；断言行数为 96,454
6. 沿用已有的 split 列（不重新划分）
```

第 1–5 步只需调用 `load_feature_table(problem)`。建模阶段尚需完成：在训练集上用 `TimeSeriesSplit` 调参，并分别评估已知卖家与新卖家（`seller_past_deliveries == 0`）。

**5.2 多卖家订单的聚合规则**

1,275 个订单有多个卖家，因此桥接表有 97,795 行、对应 96,454 个订单。订单在**最后一个**包裹到达时才算送达，因此按最坏情况聚合：

| 卖家特征 | 每个订单的规则 |
|---|---|
| `seller_past_late_rate`、`seller_past_delivery_days`、`route_past_late_rate` | **最大值** |
| `seller_backlog_7d` | **最大值** |
| `seller_past_orders`、`seller_past_deliveries` | **最小值**（经验最少的卖家） |
| `seller_is_new` | **任一为真** |
| `seller_past_avg_freight` | **平均值** |

**5.3 连接规则**（均已在 `add_seller_features` 中强制执行）
1. 合并**前**先聚合为每个订单一行；直接合并桥接表会导致多卖家订单重复。
2. 每次连接后行数必须保持 **96,454**。
3. 连接写成代码（`src/features.py`），而不是一次性的 notebook 单元格。

**5.4 训练集中未出现过的卖家**

| 数据集 | 卖家无训练集历史的订单占比 | 新卖家 |
|---|---|---|
| 验证集 | 6.6%（按 `seller_past_deliveries == 0` 计为 6.8%） | 1,238 个中有 184 个 |
| 测试集 | **13.4%** | 1,261 个中有 256 个 |

训练集中 42% 的卖家订单数少于 5 个。平滑与默认值是**必需**的，并需在报告中列为局限性。

**5.5 对成员 B notebook 版本的修正**（`notebooks/03_feature_engineering_b.ipynb`）

| notebook 中的问题 | 影响 | `src/features.py` 中的修正 |
|---|---|---|
| 历史包含下单时**尚未送达**的更早订单 | 8.5% 的历史记录、87% 的训练集行受影响；卖家延迟率与 `is_late` 的训练集相关性由 0.031 虚高到 0.097 | 只使用下单前已送达的订单 |
| 验证集/测试集使用全部训练集订单的画像，其送达时间最晚到 2018-10-17 | 使用了下单日期之后的结果 | 所有数据集采用相同的"下单前已送达"规则 |
| "延迟" = `delivery_days > promised_delivery_days` | 与 `is_late` 有 776 个订单不一致 | 改用 `is_late` |
| 训练集行用滚动历史，验证集/测试集用固定画像 | 特征在不同数据集中含义不同 | 所有数据集采用同一定义 |
| 延迟率未平滑 | 小卖家的比率噪声大 | 与送达天数一样做平滑 |
| 缺少 `seller_backlog_7d`、`seller_past_avg_freight`、`route_past_late_rate` | — | 已实现 |
| 重复列（见 5.6） | — | 未保留 |
| Windows 路径，只存在于 notebook | 其他成员无法运行 | 代码放在 `src/features.py` |

已验证：打乱验证集/测试集的结果后，所有卖家特征保持不变；在 300 个随机订单上逐一暴力复算，结果完全一致。

**5.6 未保留的成员 B 特征列（重叠）**

| 列 | 原因 |
|---|---|
| `purchase_weekday`、`purchase_hour`、`purchase_month`（被覆盖为数字） | 与成员 A 的列完全相同 |
| `purchase_is_weekend` | = `purchase_weekday ≥ 5` |
| `multiple_payment_types` | 与 `payment_type_count > 1` 完全相同 |
| `installment_flag` | = `payment_installments > 1` |
| `purchase_year` | 验证集和测试集全部在 2018 年，该列只反映平台随时间的增长 |
| `mean_*` / `max_*` 成对列、历史商品数与历史订单金额 | 彼此相关系数在 0.99 以上（98.7% 的订单只有一个卖家） |
| `seller_hist_std_delivery_days` | 不在计划中；在测试月份没有信号 |

### 6. 使用 Lasso：需要调整的地方

Lasso 是带 L1 惩罚项的线性模型。与树模型相比，它不能处理缺失值，对特征尺度敏感，也无法学习非线性关系。在高度相关的特征中，它会较为随意地只保留其中一个。

**6.1 Lasso 的定位**

| 问题 | 线性模型族（基线 → 正则化） |
|---|---|
| B：`delivery_days`（回归） | `LinearRegression` → `Ridge` → **`Lasso`**（可选 `ElasticNet`） |
| A：`is_late`（分类） | `LogisticRegression` → **L1 逻辑回归**（`penalty="l1"`、`solver="saga"`、`class_weight="balanced"`） |

整体仍保持 2–3 个模型族（线性 + 树模型）。模型尚未确定；`src/models.py` 仍是占位文件。

**6.2 预处理**（已实现：`build_preprocessor`，只基于训练集拟合）

| 要求 | 做法（已实现） |
|---|---|
| 标准化 | 对数值列使用 `StandardScaler`；树模型用 `scale=False` |
| 缺失值（订单文件：477 个距离、265 个买家坐标、213 个卖家坐标、1 行支付数据；工程特征：2,881 个 `seller_past_avg_freight`、477 个 `promised_days_per_km`） | `SimpleImputer(strategy="median")`。**不使用 `add_indicator`**：`geolocation_missing` 与 `payment_missing` 已标记所有缺失，指示列只会多出约六个几乎相同的列 |
| 偏态（距离、重量、体积、价格、运费、计数、卖家历史计数） | 标准化前先做 `log1p` |
| 类别变量（`product_category` 72 个、`region_pair` 23 个、`customer_state` 27 个、`payment_type`、`purchase_weekday`） | `OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20)`。训练集中的低频类别：6 个商品类别、`N_to_NE/S/SE`、支付方式 `Unknown` |
| `purchase_month` 为 `"YYYY-MM"`（测试集月份从未在训练集出现） | 替换为月份的 sin/cos |
| 非线性的日历效应 | 固定分段后独热编码：`days_to_holiday`（0–7、8–14、15–30、31–60、61+）、`purchase_hour`（夜间、上午、下午、晚上） |
| `delivery_days` 右偏 | 尝试 `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)`（建模阶段） |

**6.3 线性模型已去除的冗余列**（`FEATURES_LINEAR`；`FEATURES_TREE` 保留全部）

| 冗余组合 | `FEATURES_LINEAR` 保留 |
|---|---|
| `total_weight_g`（同时被列为 `weight_g`） | 只保留一次 |
| `total_volume_cm3`、`volumetric_weight_kg`（= 体积 / 6000） | `total_volume_cm3` |
| `density_g_cm3` | 删除：= 重量 / 体积，且有不合理的值 |
| `estimated_min/mean/max_distance_km` | `mean` |
| `payment_value` ≈ `order_value` + `freight_value`；`payment_per_installment` | 两者都删除 |
| `seller_count`、`multi_seller_flag`、`seller_state_count` | `multi_seller_flag` |
| `same_state`、`any_seller_same_state`、`region_pair` | `same_state`、`region_pair` |
| `same_city`、`any_seller_same_city` | `same_city` |
| `mean_item_price` × `item_count` = `order_value` | `order_value`、`item_count` |
| `customer_region`、`primary_seller_region`、`primary_seller_state` | 删除；`region_pair` 与 `customer_state` 已涵盖 |
| 原始坐标（`*_zip_lat/lng`） | 删除；距离与大区已表达相同信息 |
| `promised_days_per_km` | 删除：取对数后等于 log(承诺天数) − log(距离)，两者都已在列表中 |
| `weekend_days_to_estimate` | 删除：与 `promised_delivery_days` 相关系数 0.97 |
| `platform_orders_7d` | 删除：到测试月份增长约 40%；保留 `platform_load_ratio` |
| `seller_past_orders`、`seller_past_deliveries`（0.998） | `seller_past_orders` |

线性列表中剩余的强相关列：三个重量/体积指标之间的相关系数为 0.78–0.91。

**6.4 为 Lasso 加入的非线性项：** 已实现距离分段（< 100 / 100–500 / 500–1,500 / > 1,500 公里）。**去掉 `distance × remote_flag`**：偏远州订单的距离都在 2,000–3,000 公里，`log1p` 后该项与 `remote_flag` 完全相同（相关系数 1.00）。`billable_weight_kg` 本身已是非线性。

**6.5 调参：** `alpha` 取 `np.logspace(-4, 1, 30)`，`C` 取 `np.logspace(-3, 2, 30)`，使用 `GridSearchCV` 并设置 `cv=TimeSeriesSplit(5)`。

**6.6 可解释性：** 报告系数不为零的特征数量及标准化系数最大的特征。与树模型的特征重要性对比（报告第 7 节），并检查各折之间特征选择是否稳定。

### 7. 工作分工

| | 成员 A：日历、预处理与 Lasso | 成员 B：基于桥接表的卖家特征 |
|---|---|---|
| 特征 | ✅ 已实现 8 个（`est_date_on_weekend` 替换为 `weekend_days_to_estimate`） | ✅ 已实现 6 个，按 5.5 的修正合并进 `src/features.py` |
| 其他工作 | ✅ `build_preprocessor`；✅ `FEATURES_LINEAR` / `FEATURES_TREE`；✅ 分段；✅ 缺失值处理；⏳ Lasso（模型尚未确定） | ✅ 平滑与默认值；✅ 聚合（5.2）；✅ 连接与行数检查；⏳ 分别评估已知卖家与新卖家（建模阶段） |

**尚未分配给任何人：** 基于 `FEATURES_TREE` 的树模型（DecisionTree、RandomForest）。

### 8. 时间不足时的优先特征（14 个，标 ★）

`promised_days`、`haversine_km`、`region_pair`、`remote_flag`、`category`、`n_items`、`payment_type`、`purchase_month`（转换为月份）、`days_to_holiday`、`seller_backlog_7d`、`seller_past_late_rate`，以及三个已在文件中、无需额外工作的特征：`customer_is_capital`、`billable_weight_kg`、`freight_value`。

14 个特征现已全部实现。数据检查表明其中三个值得重新考虑（见 9.6–9.8）：`purchase_month` 和 `seller_past_late_rate` 在测试月份信号很弱，`days_to_holiday` 在测试月份的取值范围不同。`platform_orders_7d`（训练集中与 `is_late` 的相关系数为 0.12）和 `seller_past_delivery_days`（在所有数据集中与 `delivery_days` 的相关系数为 0.21–0.27）是更强的候选特征。

### 9. 待解决问题

1. **验证集月份不具代表性。** 延迟率：训练集 7.1%，**验证集 3.4%**（2018 年 7 月），测试集 6.2%。不要只用验证集来调整分类阈值或选择模型；应在训练集上使用 `TimeSeriesSplit`，或把验证集并入训练集。
2. **`same_state` 含义已改变。** 现在只有**所有**卖家都与买家同州时才为 1（`any_seller_same_state` = 至少一个），与第一版文件相比有 126 个订单发生变化。需在报告中说明。
3. **问题 B 的描述。** `notebooks/01_problem_scoping.ipynb`、`notebooks/05_regression_freight.ipynb` 和 `report/report_outline.md` 仍把运费写作问题 B。
4. ~~`purchase_month` 建模前必须转换。~~ 已完成（第 4 节 G7）。
5. **`seller_avg_handling_days`** 需要原始数据中的交给承运商日期。属于可选项；时间不够就放弃。
6. **卖家与线路延迟率无法延续到测试月份。** 与 `is_late` 的相关系数：

   | | 训练集 | 验证集 | 测试集 |
   |---|---|---|---|
   | `seller_past_late_rate` | 0.031 | 0.045 | 0.004 |
   | `route_past_late_rate` | 0.103 | 0.039 | **−0.099** |
   | `seller_past_delivery_days` → `delivery_days` | 0.209 | 0.270 | 0.238 |

   在 2018 年 2–3 月延迟高峰（延迟率 14–19%）中经常延迟的线路，在 8 月却按时送达。需在报告中列为局限性：送达天数历史可靠，延迟率不可靠。
7. **日历特征在验证集和测试集中的取值不同。** `peak_season_flag` 在所有验证集和测试集行中都为 0，因此无法在其中检验其效果（训练集中延迟率 15.3% 对比 6.6%）。`days_to_holiday`：验证集全部在 31 天以上，测试集没有 0–7 天和 61 天以上的值。
8. **月份主要反映特定年份。** 每个月份在训练集中只出现一到两年；8 月在训练集中只有 2017 年 8 月（延迟率 2.9%），而测试月份为 6.2%。
9. **成员 B 的 notebook** 仍包含有泄漏的版本和 Windows 路径。应将其主体替换为调用 `load_feature_table`。
10. **环境：** `requirements.txt` 要求 `pandas<3`；代码在 pandas 3.0.3 上也能运行。

# Feature Engineering Plan — Milestone 2 / 特征工程计划 — 第二阶段

[English](#english) · [中文](#中文)

**Revision 2 — 2026-10-04.** Updated for the revised cleaned datasets: `late_delivery_classification.csv`, `delivery_time_regression.csv` and `order_seller_bridge.csv`.
**第 2 版 —— 2026-10-04。** 已根据修订后的清洗数据集更新：`late_delivery_classification.csv`、`delivery_time_regression.csv` 和 `order_seller_bridge.csv`。

---

<a id="english"></a>

## English

### 1. Context

| | Problem A | Problem B |
|---|---|---|
| Question | Will the order arrive after the promised date? | How many days will delivery take? |
| Type | Classification | Regression |
| Target | `is_late` (6.8% late overall) | `delivery_days` |
| File | `late_delivery_classification.csv` | `delivery_time_regression.csv` |

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

| File | Shape | Contents |
|---|---|---|
| `late_delivery_classification.csv` | 96,454 × 52 | One row per order: 49 feature/ID columns + `split` + target `is_late` |
| `delivery_time_regression.csv` | 96,454 × 52 | Same orders, same columns, target `delivery_days` |
| `order_seller_bridge.csv` | 97,795 × 13 | One row per **order and seller**: `seller_id`, seller city/state/zip/coordinates, `seller_item_count`, `seller_order_value`, `seller_freight_value`, `estimated_seller_customer_distance_km`, `order_purchase_timestamp`, `split` |

Checks that passed against the raw Olist data:
- The bridge's item counts, values and freight match the raw order items exactly. No order–seller pair appears twice, and its `split` matches the order-level files.
- `primary_seller_state` is the state of the seller of the first item (`order_item_id == 1`).
- `volumetric_weight_kg`, `billable_weight_kg`, `density_g_cm3` and `mean_item_price` recompute exactly.

**Leakage:** never use `is_late` as a feature for Problem B, nor `delivery_days` for Problem A. Also excluded: the delivered-to-carrier and delivered-to-customer dates, `delay_days`, `delivery_status` and `review_score`.

### 4. Feature catalogue: 47 core features, 43 available

✅ = already a column · ➕ = to be calculated · ❌ = not available · ★ = priority feature

**G1 — Promise and timing (5): 1 ✅ · 1 ➕ · 3 ❌**

| Feature | Column / how | Status |
|---|---|---|
| `promised_days` ★ | `promised_delivery_days` | ✅ |
| `promised_days_per_km` | `promised_delivery_days / estimated_mean_distance_km` | ➕ |
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
| `density` | `density_g_cm3` | ✅ |
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

**G6 — Seller history (7): 6 ➕ · 1 ❌.** All built from the bridge (see Section 5).

| Feature | How | Status |
|---|---|---|
| `seller_past_orders` | count of the seller's earlier orders in the bridge | ➕ |
| `seller_backlog_7d` ★ | the seller's orders in the 7 days before purchase | ➕ |
| `seller_is_new` | the seller's first order was less than 30 days earlier | ➕ |
| `seller_past_avg_freight` | mean of earlier `seller_freight_value` | ➕ |
| `seller_past_late_rate` ★ | bridge + `is_late`, fitted on train only (for Problem B: seller's past mean `delivery_days`) | ➕ |
| `route_past_late_rate` | seller state → customer state, fitted on train only | ➕ |
| `seller_avg_handling_days` | needs the raw delivered-to-carrier date | ❌ optional raw join |

**G7 — Calendar and platform load (7): 1 ✅ · 6 ➕**

| Feature | How | Status |
|---|---|---|
| `purchase_month` ★ | `purchase_month` is text `"YYYY-MM"`: **convert it to month of year** | ✅ (convert) |
| `purchase_weekday` | from `order_purchase_timestamp` | ➕ |
| `purchase_hour` | from `order_purchase_timestamp` | ➕ |
| `days_to_holiday` ★ | Brazilian holiday calendar | ➕ |
| `peak_season_flag` | Black Friday, Christmas, Carnival | ➕ |
| `est_date_on_weekend` | purchase date + `promised_delivery_days` falls on a weekend | ➕ |
| `platform_orders_7d` | orders in the 7 days before purchase (approximation: only delivered orders are in the files) | ➕ |

Analysis only, not used in the model: `strike_period_flag` (truckers' strike, May 2018).

**Totals**

| | ✅ In the files | ➕ To calculate | ❌ Not available | Total |
|---|---|---|---|---|
| Count | 30 | 13 | 4 | 47 |

`weight_g` and `total_order_weight` are the same column, so there are **46 distinct core features** and **42 distinct available features**.

### 5. Seller data: kept out of the order files, joined later through the bridge

**Approach.** The order-level files stay small and contain **no `seller_id`**. Seller features are calculated from `order_seller_bridge.csv` **after cleaning and before training**, reduced to one row per order, and joined on `order_id`.

**Why later:** the strongest seller features use past outcomes (late rate, mean delivery days). Calculated over the whole dataset, they would include validation and test outcomes. Built in the training stage, they are fitted on training data only.

**5.1 Pipeline order**

```
1. Order-level file (is_late or delivery_days)               ← cleaning owner
2. Per-seller features on the bridge (97,795 rows)
     a. Point-in-time (can't leak; use only earlier orders):
        seller_past_orders, seller_backlog_7d, seller_is_new, seller_past_avg_freight
     b. Fitted on train only (use outcomes):
        seller_past_late_rate / seller past mean delivery_days, route_past_late_rate
        - smoothing: (late + m × prior) / (n + m), prior = train average
        - training rows: only orders delivered BEFORE this order's purchase
          (delivered time = purchase + delivery_days)
        - validation/test rows: statistics from the training set only
        - seller not seen in training → fall back to the prior
3. Reduce to one row per order (aggregation rules in 5.2)
4. Merge on order_id with validate="one_to_one"; assert 96,454 rows
5. Use the existing split column (no re-splitting)
6. Tune with TimeSeriesSplit on train
7. Evaluate known vs new sellers separately
```

**5.2 Aggregation rules for multi-seller orders**

1,275 orders have more than one seller, so the bridge has 97,795 rows for 96,454 orders. The order counts as delivered when its **last** parcel arrives, so aggregate towards the worst case:

| Seller feature | Rule per order |
|---|---|
| `seller_past_late_rate`, `route_past_late_rate`, past mean delivery days | **max** |
| `seller_backlog_7d` | **max** |
| `seller_past_orders` | **min** (least experienced seller) |
| `seller_is_new` | **any** |
| `seller_past_avg_freight` | **mean** |

**5.3 Join rules**
1. Aggregate to one row per order **before** merging; merging the bridge directly duplicates multi-seller orders.
2. The row count must stay at **96,454** after every join.
3. Write the join in code (`src/features.py`), not as a one-off notebook cell.

**5.4 Sellers not seen in training**

| Split | Orders whose seller is not in train | New sellers |
|---|---|---|
| Validation | 6.6% | 184 of 1,238 |
| Test | **13.4%** | 256 of 1,261 |

42% of training sellers have fewer than 5 orders. Smoothing and the fallback are **required**; list this as a limitation in the report.

### 6. Using Lasso: what changes

Lasso is a linear model with an L1 penalty. Compared with tree models it cannot handle missing values, it is sensitive to feature scale, and it cannot learn non-linear effects. Among strongly correlated features it keeps one more or less arbitrarily.

**6.1 Where Lasso fits**

| Problem | Linear family (baseline → regularised) |
|---|---|
| B: `delivery_days` (regression) | `LinearRegression` → `Ridge` → **`Lasso`** (optionally `ElasticNet`) |
| A: `is_late` (classification) | `LogisticRegression` → **L1 logistic regression** (`penalty="l1"`, `solver="saga"`, `class_weight="balanced"`) |

The plan stays within 2–3 model families (linear + tree-based).

**6.2 Preprocessing** (all inside `build_preprocessor`, fitted on training data only)

| Requirement | Action |
|---|---|
| Scaling | `StandardScaler` on numeric columns |
| Missing values (477 order distances, 265 customer coordinates, 487 bridge distances, 217 seller coordinates, 1 payment row) | `SimpleImputer(strategy="median", add_indicator=True)` |
| Skewed values (distance, weight, volume, price, freight, counts) | `log1p` before scaling |
| Categories (`product_category` 72, `region_pair` 23, `customer_state` 27) | `OneHotEncoder(handle_unknown="ignore", min_frequency=...)` |
| `purchase_month` as `"YYYY-MM"` (test months never appear in training) | Replace with month of year, weekday and hour (sin/cos or one-hot) |
| `delivery_days` is right-skewed | Try `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)` |

**6.3 Redundant columns to remove for linear models**

| Redundant set | Keep in `FEATURES_LINEAR` |
|---|---|
| `total_weight_g` (also listed as `weight_g`) | Once |
| `total_volume_cm3`, `volumetric_weight_kg` (= volume / 6000) | One |
| `estimated_min/mean/max_distance_km` | `mean` (plus optionally `max − min`) |
| `payment_value` ≈ `order_value` + `freight_value` | Drop `payment_value` |
| `seller_count`, `multi_seller_flag`, `seller_state_count` | One or two |
| `same_state`, `any_seller_same_state`, `region_pair` | One or two |
| `same_city`, `any_seller_same_city` | One |
| `mean_item_price` × `item_count` = `order_value` | Two of the three |
| Raw coordinates (`*_zip_lat/lng`) | Drop; distance and region already describe them |

Keep **two lists** in `src/features.py`: `FEATURES_LINEAR` (deduplicated) and `FEATURES_TREE` (everything).

**6.4 Non-linear terms for Lasso:** `estimated_mean_distance_km × remote_flag`, distance bands (< 100 / 100–500 / 500–1,500 / > 1,500 km). `promised_days_per_km` and `billable_weight_kg` are already non-linear.

**6.5 Tuning:** `alpha` in `np.logspace(-4, 1, 30)` and `C` in `np.logspace(-3, 2, 30)`, using `GridSearchCV` with `cv=TimeSeriesSplit(5)`.

**6.6 Interpretability:** report how many features keep a non-zero coefficient and the largest standardised coefficients. Compare with tree feature importance (report section 7), and check that selection is stable across folds.

### 7. Remaining work split

Most features already exist, so the split covers **the work still to do**.

| | Person A: calendar, preprocessing and Lasso | Person B: seller features from the bridge |
|---|---|---|
| Features | 8: `promised_days_per_km`, month of year (from `purchase_month`), `purchase_weekday`, `purchase_hour`, `days_to_holiday`, `peak_season_flag`, `est_date_on_weekend`, `platform_orders_7d` | 6: `seller_past_orders`, `seller_backlog_7d`, `seller_is_new`, `seller_past_avg_freight`, `seller_past_late_rate` (and past mean `delivery_days`), `route_past_late_rate` |
| Other work | `build_preprocessor`; `FEATURES_LINEAR` / `FEATURES_TREE`; interaction terms; missing-value handling | Smoothing and fallback for new sellers; aggregation to one row per order (5.2); join and row-count check; evaluation of known vs new sellers |
| Nature | Many simple formulas plus the preprocessing step | Fewer features, but leakage-sensitive |

Person B's features need no input from Person A, so both can start immediately.

### 8. Priority features if time runs short (14, marked ★)

`promised_days`, `haversine_km`, `region_pair`, `remote_flag`, `category`, `n_items`, `payment_type`, `purchase_month` (as month of year), `days_to_holiday`, `seller_backlog_7d`, `seller_past_late_rate`, plus three already in the files that need no work: `customer_is_capital`, `billable_weight_kg`, `freight_value`.

### 9. Open issues

1. **The validation month is unrepresentative.** Late rate: train 7.1%, **validation 3.4%** (July 2018), test 6.2%. Do not tune the decision threshold or choose models on validation alone; use `TimeSeriesSplit` on train, or merge validation into train.
2. **`same_state` changed meaning.** It is now 1 only if **all** sellers are in the customer's state (`any_seller_same_state` = at least one), which changes 126 orders compared with the first file. Document this in the report.
3. **Problem B wording.** `notebooks/01_problem_scoping.ipynb`, `notebooks/05_regression_freight.ipynb` and `report/report_outline.md` still describe freight cost as Problem B.
4. **`purchase_month`** must be converted before modelling (Section 6.2).
5. **`seller_avg_handling_days`** would need the raw delivered-to-carrier date. It is optional; drop it unless there is time.

---

<a id="中文"></a>

## 中文

### 1. 背景

| | 问题 A | 问题 B |
|---|---|---|
| 问题 | 订单是否会晚于承诺日期送达？ | 送达需要多少天？ |
| 类型 | 分类 | 回归 |
| 目标变量 | `is_late`（整体延迟率 6.8%） | `delivery_days` |
| 文件 | `late_delivery_classification.csv` | `delivery_time_regression.csv` |

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

| 文件 | 形状 | 内容 |
|---|---|---|
| `late_delivery_classification.csv` | 96,454 × 52 | 每个订单一行：49 个特征/ID 列 + `split` + 目标 `is_late` |
| `delivery_time_regression.csv` | 96,454 × 52 | 相同订单、相同列，目标为 `delivery_days` |
| `order_seller_bridge.csv` | 97,795 × 13 | 每个**订单 + 卖家**一行：`seller_id`、卖家城市/州/邮编/坐标、`seller_item_count`、`seller_order_value`、`seller_freight_value`、`estimated_seller_customer_distance_km`、`order_purchase_timestamp`、`split` |

与原始 Olist 数据核对通过的项目：
- 桥接表的商品数、金额和运费与原始订单商品表完全一致；没有重复的订单–卖家组合；其 `split` 与订单级文件一致。
- `primary_seller_state` 为第一件商品（`order_item_id == 1`）卖家所在的州。
- `volumetric_weight_kg`、`billable_weight_kg`、`density_g_cm3` 和 `mean_item_price` 均可准确复算。

**数据泄漏：** 问题 B 不能使用 `is_late` 作为特征，问题 A 也不能使用 `delivery_days`。同样排除：交给承运商日期与送达买家日期、`delay_days`、`delivery_status`、`review_score`。

### 4. 特征清单：47 个核心特征，43 个可用

✅ = 已有列 · ➕ = 需计算 · ❌ = 不可用 · ★ = 优先特征

**G1 —— 承诺时效与时间（5 个）：1 ✅ · 1 ➕ · 3 ❌**

| 特征 | 对应列 / 计算方式 | 状态 |
|---|---|---|
| `promised_days` ★ | `promised_delivery_days` | ✅ |
| `promised_days_per_km` | `promised_delivery_days / estimated_mean_distance_km` | ➕ |
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
| `density` | `density_g_cm3` | ✅ |
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

**G6 —— 卖家历史（7 个）：6 ➕ · 1 ❌。** 全部由桥接表构建（见第 5 节）。

| 特征 | 计算方式 | 状态 |
|---|---|---|
| `seller_past_orders` | 卖家在桥接表中更早的订单数 | ➕ |
| `seller_backlog_7d` ★ | 下单前 7 天内该卖家的订单数 | ➕ |
| `seller_is_new` | 卖家首单距今不足 30 天 | ➕ |
| `seller_past_avg_freight` | 更早订单的 `seller_freight_value` 平均值 | ➕ |
| `seller_past_late_rate` ★ | 桥接表 + `is_late`，只基于训练集拟合（问题 B 用卖家历史平均 `delivery_days`） | ➕ |
| `route_past_late_rate` | 卖家州 → 买家州，只基于训练集拟合 | ➕ |
| `seller_avg_handling_days` | 需要原始数据中的交给承运商日期 | ❌ 可选的原始数据连接 |

**G7 —— 日历与平台负载（7 个）：1 ✅ · 6 ➕**

| 特征 | 计算方式 | 状态 |
|---|---|---|
| `purchase_month` ★ | `purchase_month` 是文本 `"YYYY-MM"`：**需转换为月份（1–12）** | ✅（需转换） |
| `purchase_weekday` | 来自 `order_purchase_timestamp` | ➕ |
| `purchase_hour` | 来自 `order_purchase_timestamp` | ➕ |
| `days_to_holiday` ★ | 巴西节假日日历 | ➕ |
| `peak_season_flag` | 黑色星期五、圣诞节、狂欢节 | ➕ |
| `est_date_on_weekend` | 下单日期 + `promised_delivery_days` 是否为周末 | ➕ |
| `platform_orders_7d` | 下单前 7 天的订单量（近似值：文件中只有已送达订单） | ➕ |

仅用于分析、不放入模型：`strike_period_flag`（2018 年 5 月卡车司机罢工）。

**合计**

| | ✅ 已有 | ➕ 需计算 | ❌ 不可用 | 合计 |
|---|---|---|---|---|
| 数量 | 30 | 13 | 4 | 47 |

`weight_g` 与 `total_order_weight` 是同一列，因此实际为 **46 个不同的核心特征**、**42 个不同的可用特征**。

### 5. 卖家数据：不放入订单级文件，之后通过桥接表连接

**做法。** 订单级文件保持精简，**不含 `seller_id`**。卖家特征在**清洗之后、训练之前**由 `order_seller_bridge.csv` 计算，归约为每个订单一行后，以 `order_id` 连接。

**为什么之后再连接：** 最强的卖家特征依赖历史结果（延迟率、平均送达天数）。若基于全量数据计算，会包含验证集与测试集的结果；在训练阶段构建，则只基于训练集拟合。

**5.1 流程顺序**

```
1. 订单级文件（is_late 或 delivery_days）                     ← 清洗负责人
2. 在桥接表（97,795 行）上计算卖家特征
     a. 时点特征（不会泄漏；只使用更早的订单）：
        seller_past_orders、seller_backlog_7d、seller_is_new、seller_past_avg_freight
     b. 只基于训练集拟合（使用结果变量）：
        seller_past_late_rate / 卖家历史平均 delivery_days、route_past_late_rate
        - 平滑：(延迟数 + m × 先验) / (n + m)，先验 = 训练集平均值
        - 训练集行：只使用在本订单下单之前已送达的订单
          （送达时间 = 下单时间 + delivery_days）
        - 验证集/测试集行：只使用训练集的统计量
        - 训练集中未出现的卖家 → 使用先验值
3. 归约为每个订单一行（聚合规则见 5.2）
4. 以 order_id 合并，validate="one_to_one"；断言行数为 96,454
5. 沿用已有的 split 列（不重新划分）
6. 在训练集上用 TimeSeriesSplit 调参
7. 分别评估已知卖家与新卖家
```

**5.2 多卖家订单的聚合规则**

1,275 个订单有多个卖家，因此桥接表有 97,795 行、对应 96,454 个订单。订单在**最后一个**包裹到达时才算送达，因此按最坏情况聚合：

| 卖家特征 | 每个订单的规则 |
|---|---|
| `seller_past_late_rate`、`route_past_late_rate`、历史平均送达天数 | **最大值** |
| `seller_backlog_7d` | **最大值** |
| `seller_past_orders` | **最小值**（经验最少的卖家） |
| `seller_is_new` | **任一为真** |
| `seller_past_avg_freight` | **平均值** |

**5.3 连接规则**
1. 合并**前**先聚合为每个订单一行；直接合并桥接表会导致多卖家订单重复。
2. 每次连接后行数必须保持 **96,454**。
3. 把连接写成代码（`src/features.py`），而不是一次性的 notebook 单元格。

**5.4 训练集中未出现过的卖家**

| 数据集 | 卖家不在训练集中的订单占比 | 新卖家 |
|---|---|---|
| 验证集 | 6.6% | 1,238 个中有 184 个 |
| 测试集 | **13.4%** | 1,261 个中有 256 个 |

训练集中 42% 的卖家订单数少于 5 个。平滑与默认值是**必需**的，并需在报告中列为局限性。

### 6. 使用 Lasso：需要调整的地方

Lasso 是带 L1 惩罚项的线性模型。与树模型相比，它不能处理缺失值，对特征尺度敏感，也无法学习非线性关系。在高度相关的特征中，它会较为随意地只保留其中一个。

**6.1 Lasso 的定位**

| 问题 | 线性模型族（基线 → 正则化） |
|---|---|
| B：`delivery_days`（回归） | `LinearRegression` → `Ridge` → **`Lasso`**（可选 `ElasticNet`） |
| A：`is_late`（分类） | `LogisticRegression` → **L1 逻辑回归**（`penalty="l1"`、`solver="saga"`、`class_weight="balanced"`） |

整体仍保持 2–3 个模型族（线性 + 树模型）。

**6.2 预处理**（全部放在 `build_preprocessor` 中，只基于训练集拟合）

| 要求 | 做法 |
|---|---|
| 标准化 | 对数值列使用 `StandardScaler` |
| 缺失值（477 个订单距离、265 个买家坐标、487 个桥接表距离、217 个卖家坐标、1 行支付数据） | `SimpleImputer(strategy="median", add_indicator=True)` |
| 偏态（距离、重量、体积、价格、运费、计数） | 标准化前先做 `log1p` |
| 类别变量（`product_category` 72 个、`region_pair` 23 个、`customer_state` 27 个） | `OneHotEncoder(handle_unknown="ignore", min_frequency=...)` |
| `purchase_month` 为 `"YYYY-MM"`（测试集月份从未在训练集出现） | 替换为月份、星期几和小时（sin/cos 或独热编码） |
| `delivery_days` 右偏 | 尝试 `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)` |

**6.3 线性模型需去除的冗余列**

| 冗余组合 | `FEATURES_LINEAR` 保留 |
|---|---|
| `total_weight_g`（同时被列为 `weight_g`） | 只保留一次 |
| `total_volume_cm3`、`volumetric_weight_kg`（= 体积 / 6000） | 二选一 |
| `estimated_min/mean/max_distance_km` | `mean`（可另加 `max − min`） |
| `payment_value` ≈ `order_value` + `freight_value` | 删除 `payment_value` |
| `seller_count`、`multi_seller_flag`、`seller_state_count` | 保留一到两个 |
| `same_state`、`any_seller_same_state`、`region_pair` | 保留一到两个 |
| `same_city`、`any_seller_same_city` | 二选一 |
| `mean_item_price` × `item_count` = `order_value` | 三者保留两个 |
| 原始坐标（`*_zip_lat/lng`） | 删除；距离与大区已表达相同信息 |

在 `src/features.py` 中维护**两份列表**：`FEATURES_LINEAR`（去重后）和 `FEATURES_TREE`（全部特征）。

**6.4 为 Lasso 加入的非线性项：** `estimated_mean_distance_km × remote_flag`、距离分段（< 100 / 100–500 / 500–1,500 / > 1,500 公里）。`promised_days_per_km` 和 `billable_weight_kg` 本身已是非线性。

**6.5 调参：** `alpha` 取 `np.logspace(-4, 1, 30)`，`C` 取 `np.logspace(-3, 2, 30)`，使用 `GridSearchCV` 并设置 `cv=TimeSeriesSplit(5)`。

**6.6 可解释性：** 报告系数不为零的特征数量及标准化系数最大的特征。与树模型的特征重要性对比（报告第 7 节），并检查各折之间特征选择是否稳定。

### 7. 剩余工作分工

大部分特征已经存在，因此分工针对**尚未完成的工作**。

| | 成员 A：日历、预处理与 Lasso | 成员 B：基于桥接表的卖家特征 |
|---|---|---|
| 特征 | 8 个：`promised_days_per_km`、月份（由 `purchase_month` 转换）、`purchase_weekday`、`purchase_hour`、`days_to_holiday`、`peak_season_flag`、`est_date_on_weekend`、`platform_orders_7d` | 6 个：`seller_past_orders`、`seller_backlog_7d`、`seller_is_new`、`seller_past_avg_freight`、`seller_past_late_rate`（及历史平均 `delivery_days`）、`route_past_late_rate` |
| 其他工作 | `build_preprocessor`；`FEATURES_LINEAR` / `FEATURES_TREE`；交互项；缺失值处理 | 新卖家的平滑与默认值；聚合为每个订单一行（5.2）；连接与行数检查；分别评估已知卖家与新卖家 |
| 工作性质 | 较多简单公式，加上预处理 | 特征较少，但对数据泄漏敏感 |

成员 B 的特征不依赖成员 A 的产出，两人可以同时开始。

### 8. 时间不足时的优先特征（14 个，标 ★）

`promised_days`、`haversine_km`、`region_pair`、`remote_flag`、`category`、`n_items`、`payment_type`、`purchase_month`（转换为月份）、`days_to_holiday`、`seller_backlog_7d`、`seller_past_late_rate`，以及三个已在文件中、无需额外工作的特征：`customer_is_capital`、`billable_weight_kg`、`freight_value`。

### 9. 待解决问题

1. **验证集月份不具代表性。** 延迟率：训练集 7.1%，**验证集 3.4%**（2018 年 7 月），测试集 6.2%。不要只用验证集来调整分类阈值或选择模型；应在训练集上使用 `TimeSeriesSplit`，或把验证集并入训练集。
2. **`same_state` 含义已改变。** 现在只有**所有**卖家都与买家同州时才为 1（`any_seller_same_state` = 至少一个），与第一版文件相比有 126 个订单发生变化。需在报告中说明。
3. **问题 B 的描述。** `notebooks/01_problem_scoping.ipynb`、`notebooks/05_regression_freight.ipynb` 和 `report/report_outline.md` 仍把运费写作问题 B。
4. **`purchase_month`** 建模前必须转换（第 6.2 节）。
5. **`seller_avg_handling_days`** 需要原始数据中的交给承运商日期。属于可选项；时间不够就放弃。

# T1 Late-Delivery Classifier — Model Card

Owner: Member C (WANG YIHUI). Code: `T1_modeling.ipynb` (full run), `t1_pipeline.py` (reusable functions), `inference.py` (serving).

## 1. Purpose
Score each order **at placement** with the probability that it will be delivered late (`Late_delivery_risk = 1`), so that
fulfilment can review risky orders before dispatch. One prediction per `Order Id`.

## 2. Data and definitions
| Item | Definition |
|---|---|
| Source | `A/cleaned_order_items.csv`, aggregated by `Order Id` (reconciled with `A/order_level_model_data.csv`) |
| Rows | 62,897 orders (late 36,048 / not late 26,849, late rate 57.3%) |
| Excluded | 2,855 `Shipping canceled` orders (label forced to 0; analysed separately) |
| Never used (leakage) | `Days for shipping (real)`, `shipping date`, `Delivery Status`, `Order Status`, `Benefit per order`, `Order Profit Per Order`, `Order Item Profit Ratio` |
| Redundant | `Days for shipment (scheduled)` = 1:1 recoding of `Shipping Mode` (Standard 4, Second 2, First 1, Same Day 0) |
| Split (by order date) | Train 2015-01-01 – 2017-03-31 (44,883) · Validation 2017-04 – 2017-09 (9,971) · Test 2017-10 – 2018-01 (8,043) |

Features: `shipping_mode`, `market`, `order_region`, `payment_type`, `customer_segment` (one-hot); `total_quantity`, `total_sales`
(pre-discount), `total_discount`, `item_count`, `unique_products`, `unique_categories`, units per department (`dept_qty_*`, 11 columns);
derived inside the pipeline: `net_sales`, `discount_rate`, `order_hour`, `order_dow`, `order_month`.
Imputation, scaling and encoding are fitted on Train only.

## 3. Models compared, simple → complex (validation)
Following the tutor's feedback, the comparison covers simple and complex non-linear models, including XGBoost.

| Model | Complexity | Feature set | ROC-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|---|
| Mode-only benchmark | benchmark | shipping mode only | 0.742 | 1.000 | 0.570 | 0.726 |
| Logistic Regression (C=10) — `t1_baseline` | simple linear | Full | 0.749 | 0.750 | 0.673 | 0.709 |
| Decision Tree (depth 5, leaf 500) | simple non-linear | Full | 0.777 | 0.911 | 0.618 | 0.737 |
| Random Forest (300 trees, depth 16, leaf 200, max_features 0.5) | complex, bagging | Full | 0.776 | 0.750 | 0.706 | 0.728 |
| **XGBoost (200 trees, depth 3, lr 0.03, min_child_weight 20, subsample 0.8, colsample 0.8) — `t1_final`** | complex, boosting | Full | 0.777 | 0.751 | 0.707 | 0.728 |
| Best NoMode model (RF) | — | without shipping mode | 0.527 | 0.753 | 0.584 | 0.658 |

All four models without shipping mode score 0.523–0.527. Selection rule: highest validation ROC-AUC; models within 0.005 AUC
are ranked by precision at recall ≥ 0.75. Decision Tree, Random Forest and XGBoost are all within 0.0012 AUC; XGBoost has
the highest precision (0.7072 vs RF 0.7063, a negligible margin) and is selected under the pre-set rule.
Full tuning grid: `results/tuning_log.csv`.

**Simple vs complex** (`results/simple_vs_complex.csv`, `results/figures/simple_vs_complex.png`; test AUC reported after
selection was frozen): test ROC-AUC is 0.753 for Logistic Regression and 0.776–0.777 for Decision Tree, Random Forest and
XGBoost, equal to the five-cell lookup ceiling (0.776). A non-linear model is needed (to capture the Same Day noon cut-off and
the mode effects), but going beyond a single tree to bagging or boosting brings no measurable gain: the limit is the information
in the pre-shipment features, not model capacity.

## 4. Final model on the untouched test set
Threshold **0.4059** (highest validation precision with validation recall ≥ 0.75; not tuned on test).

| Metric | Final XGBoost | Mode-only benchmark | RF without shipping mode |
|---|---|---|---|
| ROC-AUC | **0.777** | 0.745 | 0.532 |
| Late recall | **0.774** | 1.000 | 0.739 |
| Precision | 0.696 | 0.576 | 0.585 |
| F1 | 0.733 | 0.731 | 0.653 |
| Share of orders flagged | 64.1% | 100% | 72.8% |

Confusion matrix (test): TN 1,839 · FP 1,568 · FN 1,046 · TP 3,590.
Targets: recall ≥ 0.75 **met**; ROC-AUC ≥ 0.80 **not met**.
For capacity-based alerting (top-K% riskiest orders) see `results/test_capacity_table.csv`, e.g. flagging 40% of orders gives precision 0.87 at recall 0.60.

## 5. What the model has learned (C5)
- Permutation importance (test ROC-AUC drop): `shipping_mode` 0.262, `order_date` 0.046 (through order hour), all other features ≤ 0.0013.
- **First Class** is always late (scheduled 1 day, actual always 2) and **Second Class** is late 80% of the time; both are always flagged.
- **Same Day** orders placed at or after 12:00 are always late, before 12:00 never late — the model separates them perfectly.
- **Standard Class** (60% of orders, 40% late) and Second Class: within each mode the model's ROC-AUC is ≈ 0.50, i.e. no
  pre-shipment feature predicts which of these orders will be late. Standard Class flags are effectively random, which is where
  all false negatives and most false positives come from.
- A five-cell lookup (shipping mode, Same Day after 12:00) learned on Train reaches test ROC-AUC 0.776, the same as XGBoost.
  This is the practical ceiling with the available features, which is why 0.80 is not reachable.
- Segment differences by market / department (`results/segment_errors.csv`) mainly reflect each segment's shipping-mode mix.
- Associations are not causal: the data do not show that switching an order's shipping mode would prevent lateness.

## 6. Profit fields check (not used in the model)
`Benefit per order` = `Order Profit Per Order` (identical in every row) = `Order Item Profit Ratio` × `Order Item Total`
(ratio rounded to 2 decimals), so the three fields carry one piece of information. Order-level profit features (total profit,
profit ratio, mean / min item ratio, share of loss-making items) were tested (`results/profit_univariate_auc.csv`,
`results/profit_feature_check.csv`):
- Univariate ROC-AUC 0.497–0.509 on every split, and the direction flips between Train (< 0.5) and Test (> 0.5).
- With the shipped parameters, adding them changes validation AUC by −0.007 to +0.0002; XGBoost test AUC falls from 0.777 to 0.773
  (bootstrap 95% CI of the difference −0.007 to −0.001).
- Re-tuned on validation with the profit features, the best validation AUC is 0.7766 versus 0.7767 without them, so the
  validation-based rule gives no reason to add them. A deeper XGBoost (depth 7, 500 trees) gains +0.005 on test only
  (CI −0.000 to +0.009), which cannot be used for selection.

Conclusion: the profit fields stay out of the model; excluding them loses no measurable signal.

## 7. Canceled orders (excluded from training)
2,855 orders: `CANCELED` 1,367, `SUSPECTED_FRAUD` 1,488. All are paid by `TRANSFER` (15.8% of TRANSFER orders are canceled, 0% for
other payment types); shipping-mode, market and segment mix match delivered orders. The final model would flag 62% of them
(mean risk 0.58), so cancellation is not something this model detects (`results/canceled_*.csv`).

## 8. Limitations
- Markets appear in blocks of months (Validation = LATAM + Europe, Test = Europe + Pacific Asia), and from 2017-10 almost every
  order contains a single product, so test results do not cover Africa, USCA or LATAM.
- Shipping-mode rules (First Class always late, Same Day noon cut-off) look like artefacts of how the dataset was generated; they
  should be re-checked on new data before a real roll-out.
- The model is fitted on Train only (to 2017-03-31) so that its validation threshold stays valid. A refit on newer data must
  re-select the threshold on a later hold-out period.

## 9. Usage (for member E)
**Files**: `models/t1_final.joblib` + `models/t1_final_meta.json` (threshold, input schema, metrics, versions);
`models/t1_baseline.*` is the logistic-regression version for end-to-end testing. Example I/O: `models/example_request.json`,
`models/example_response.json`.

**Input**: one record per order with the columns in `meta["required_columns"]`; `dept_qty_*` are optional (default 0);
`order_id` is optional and echoed back. `order_date` format `YYYY-MM-DD HH:MM:SS`. Build the fields from item rows with
`t1_pipeline.aggregate_orders(items_df)`.

**Output**: `{"predictions": [{"order_id", "risk_score" (P(late)), "pred_label" (risk_score >= threshold), "model_version"}]}`.

**Local test**
```python
import json, inference
model = inference.model_fn("models")
body = open("models/example_request.json").read()
print(inference.output_fn(inference.predict_fn(inference.input_fn(body, "application/json"), model)))
```

**SageMaker**: package `t1_final.joblib` and `t1_final_meta.json` in `model.tar.gz`; use `inference.py` as `entry_point` with a
`source_dir` that also contains `t1_pipeline.py` (needed to unpickle the pipeline) and `requirements.txt`: the final model is
XGBoost, which the SageMaker scikit-learn container does not include. Set env `T1_MODEL_NAME=t1_baseline` to serve the
logistic-regression baseline (no extra dependency).

**Version mismatch**: the models were saved with Python 3.14 / scikit-learn 1.9.0 / xgboost 3.4.1. If the serving container
has other versions (or cannot install xgboost 3.4.1, which needs Python ≥ 3.10), adjust `requirements.txt` and rebuild the
model there (requires `A/cleaned_order_items.csv`):
```bash
python t1_pipeline.py retrain models/t1_final_meta.json t1_final.joblib
```
Retraining reproduces the shipped XGBoost scores exactly in the same environment; the earlier Random Forest version was also
rebuilt under scikit-learn 1.6.1 with scores agreeing to about 3 decimals.

## 10. Write-back to the warehouse (agreed with B / E)
`Order Id`, `risk_score`, `pred_label`, `model_version`, `scored_at`.

"""T1 late-delivery classifier: data preparation, model pipelines and evaluation helpers.

The fitted pipelines start with a FunctionTransformer that references `add_derived`
in this module, so this file must be importable wherever a saved model is loaded.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
ITEMS_PATH = ROOT / "A" / "cleaned_order_items.csv"
ORDERS_PATH = ROOT / "A" / "order_level_model_data.csv"

TARGET = "late_delivery_risk"
TRAIN_END = pd.Timestamp("2017-04-01")
VALID_END = pd.Timestamp("2017-10-01")
CANCELED = "Shipping canceled"
RANDOM_STATE = 42

DEPARTMENTS = ["Apparel", "Book Shop", "Discs Shop", "Fan Shop", "Fitness", "Footwear", "Golf",
               "Health and Beauty", "Outdoors", "Pet Shop", "Technology"]
DEPT_COLS = ["dept_qty_" + d.lower().replace(" ", "_") for d in DEPARTMENTS]

BASE_NUM = ["total_quantity", "total_sales", "total_discount", "item_count", "unique_products",
            "unique_categories"]
INPUT_COLS = ["order_date", "shipping_mode", "market", "order_region", "payment_type",
              "customer_segment"] + BASE_NUM + DEPT_COLS
DERIVED_NUM = ["net_sales", "discount_rate", "order_hour", "order_dow", "order_month"]
NUM_COLS = BASE_NUM + DERIVED_NUM + DEPT_COLS
CAT_FULL = ["shipping_mode", "market", "order_region", "payment_type", "customer_segment"]
FEATURE_SETS = {
    "Full": {"cat": CAT_FULL, "num": NUM_COLS},
    "NoMode": {"cat": [c for c in CAT_FULL if c != "shipping_mode"], "num": NUM_COLS},
}


def aggregate_orders(items):
    """Aggregate item-level rows (A/cleaned_order_items.csv schema) to one row per order."""
    it = items.assign(order_date=pd.to_datetime(items["order date (DateOrders)"]),
                      dept=items["Department Name"].str.strip())
    orders = it.groupby("Order Id").agg(
        order_date=("order_date", "first"),
        shipping_mode=("Shipping Mode", "first"),
        scheduled_days=("Days for shipment (scheduled)", "first"),
        market=("Market", "first"),
        order_region=("Order Region", "first"),
        payment_type=("Type", "first"),
        customer_segment=("Customer Segment", "first"),
        delivery_status=("Delivery Status", "first"),
        order_status=("Order Status", "first"),
        late_delivery_risk=("Late_delivery_risk", "first"),
        total_quantity=("Order Item Quantity", "sum"),
        total_sales=("Sales", "sum"),
        total_discount=("Order Item Discount", "sum"),
        item_count=("Order Item Id", "size"),
        unique_products=("Product Card Id", "nunique"),
        unique_categories=("Category Id", "nunique"),
    )
    dept = (it.pivot_table(index="Order Id", columns="dept", values="Order Item Quantity",
                           aggfunc="sum", fill_value=0)
            .reindex(columns=DEPARTMENTS, fill_value=0))
    orders["main_department"] = dept.idxmax(axis=1)
    dept.columns = DEPT_COLS
    return orders.join(dept).reset_index()


def load_orders(items_path=ITEMS_PATH):
    """Return (modelling orders without canceled shipments, canceled orders)."""
    orders = aggregate_orders(pd.read_csv(items_path))
    canceled = orders["delivery_status"] == CANCELED
    return (orders[~canceled].sort_values("order_date").reset_index(drop=True),
            orders[canceled].sort_values("order_date").reset_index(drop=True))


def chronological_split(df):
    train = df[df["order_date"] < TRAIN_END]
    valid = df[(df["order_date"] >= TRAIN_END) & (df["order_date"] < VALID_END)]
    test = df[df["order_date"] >= VALID_END]
    return train, valid, test


def add_derived(X):
    X = X.copy()
    date = pd.to_datetime(X["order_date"])
    X["order_hour"], X["order_dow"], X["order_month"] = date.dt.hour, date.dt.dayofweek, date.dt.month
    X["net_sales"] = X["total_sales"] - X["total_discount"]
    X["discount_rate"] = X["total_discount"] / X["total_sales"].where(X["total_sales"] > 0)
    for c in DEPT_COLS:
        if c not in X:
            X[c] = 0
    return X


def make_pipeline(model, feature_set="Full"):
    fs = FEATURE_SETS[feature_set]
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), fs["num"]),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
         fs["cat"]),
    ])
    return Pipeline([("derive", FunctionTransformer(add_derived)), ("prep", prep), ("model", model)])


class ModeRateBaseline(BaseEstimator, ClassifierMixin):
    """Naive benchmark: score = training late rate of the order's shipping mode."""

    def fit(self, X, y):
        self.rates_ = pd.Series(np.asarray(y), index=X["shipping_mode"].values).groupby(level=0).mean()
        self.prior_ = float(np.mean(y))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = X["shipping_mode"].map(self.rates_).fillna(self.prior_).to_numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def choose_threshold(y, score, min_recall=0.75):
    """Highest-precision threshold whose recall is still >= min_recall (ties -> higher threshold)."""
    precision, recall, thr = precision_recall_curve(y, score)
    ok = np.where(recall[:-1] >= min_recall)[0]
    best = ok[np.flatnonzero(precision[ok] == precision[ok].max())[-1]]
    return float(thr[best])


def evaluate(y, score, threshold):
    y = np.asarray(y)
    pred = (np.asarray(score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y, score) if len(np.unique(y)) > 1 else np.nan,
        "recall": recall_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "alert_rate": pred.mean(),
        "fpr": fp / (fp + tn) if fp + tn else np.nan,
        "threshold": threshold, "tn": tn, "fp": fp, "fn": fn, "tp": tp, "n": len(y),
    }


def tune(make_model, grid, feature_set, train, valid):
    """Fit every grid point on train and score ROC-AUC on validation."""
    rows = []
    for params in ParameterGrid(grid):
        pipe = make_pipeline(make_model(**params), feature_set).fit(train[INPUT_COLS], train[TARGET])
        rows.append({**params, "params": params, "feature_set": feature_set,
                     "valid_auc": roc_auc_score(valid[TARGET], pipe.predict_proba(valid[INPUT_COLS])[:, 1])})
    return pd.DataFrame(rows).sort_values("valid_auc", ascending=False).reset_index(drop=True)


MODEL_FACTORIES = {}


def _register():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    MODEL_FACTORIES.update({
        "LogisticRegression": lambda **p: LogisticRegression(max_iter=2000, **p),
        "DecisionTree": lambda **p: DecisionTreeClassifier(random_state=RANDOM_STATE, **p),
        "RandomForest": lambda **p: RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **p),
        "XGBoost": _xgboost,
    })


def _xgboost(**p):
    from xgboost import XGBClassifier
    return XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist", eval_metric="logloss", **p)


_register()


def retrain_from_meta(meta_path, out_path, items_path=ITEMS_PATH):
    """Rebuild a saved model in the current environment (e.g. a SageMaker container whose
    scikit-learn version differs from the one used for training)."""
    import joblib
    meta = json.loads(Path(meta_path).read_text())
    df, _ = load_orders(items_path)
    fit_df = df[df["order_date"] < pd.Timestamp(meta["fit_end"])]
    model = MODEL_FACTORIES[meta["algorithm"]](**meta["params"])
    pipe = make_pipeline(model, meta["feature_set"]).fit(fit_df[INPUT_COLS], fit_df[TARGET])
    joblib.dump(pipe, out_path)
    return pipe


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "retrain":
        sys.exit("usage: python t1_pipeline.py retrain <meta.json> <out.joblib>")
    import t1_pipeline  # pickle add_derived as t1_pipeline.add_derived, not __main__.add_derived
    t1_pipeline.retrain_from_meta(sys.argv[2], sys.argv[3])

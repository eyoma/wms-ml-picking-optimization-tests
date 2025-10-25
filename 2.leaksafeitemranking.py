import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ndcg_score
import lightgbm as lgb

# --- Load & basic parsing ---
df = pd.read_csv("wms/synthetic_warehouse_picks_sample.csv")
df["datetime_picked"] = pd.to_datetime(df["datetime_picked"], errors="coerce")
df = df.sort_values(["order_id", "datetime_picked"])

def parse_aisle(tok):
    try: return tok.split("-")[1]  # e.g., "A03"
    except: return None

def parse_section(tok):
    try: return tok.split("-")[2]  # e.g., "S015"
    except: return None

df["aisle"] = df["bin_id"].apply(parse_aisle)
df["section"] = df["bin_id"].apply(parse_section)

# True position of each line within its order (0 = first picked)
df["true_rank"] = df.groupby("order_id")["datetime_picked"].rank(method="first").astype(int) - 1
# Relevance (higher is better). n - rank makes the first-picked the most relevant.
df["group_size"] = df.groupby("order_id")["order_id"].transform("count")
df["relevance"] = (df["group_size"] - df["true_rank"]).astype(int)

# --- Split by ORDER before any aggregates ---
all_orders = df["order_id"].unique()
train_orders, test_orders = train_test_split(all_orders, test_size=0.2, random_state=42)
assert len(set(train_orders) & set(test_orders)) == 0

train_df = df[df["order_id"].isin(train_orders)].copy()
test_df  = df[df["order_id"].isin(test_orders)].copy()

# --- Aggregates computed on TRAIN ONLY, merged to both ---
# (these are safe, stationary features; avoid anything derived from the actual pick timestamp sequence)
picker_avg_delta = (
    train_df.assign(delta_s=train_df.groupby("picker_id")["datetime_picked"].diff().dt.total_seconds())
            .groupby("picker_id")["delta_s"].mean()
            .rename("avg_picker_gap_s")
)
item_pop = (
    train_df["item_id"].value_counts().rename_axis("item_id").rename("item_pick_count")
)

def attach_aggs(part):
    out = part.merge(picker_avg_delta, on="picker_id", how="left")
    out = out.merge(item_pop,       on="item_id",   how="left")
    out["avg_picker_gap_s"] = out["avg_picker_gap_s"].fillna(picker_avg_delta.median())
    out["item_pick_count"]  = out["item_pick_count"].fillna(0).astype(int)
    return out

train_df = attach_aggs(train_df)
test_df  = attach_aggs(test_df)

# --- Item-level feature table (one row per order line) ---
# IMPORTANT: do NOT include 'datetime_picked' or any feature derived from the realized sequence
# (e.g., per-line time deltas) — those leak the target.
def prep_features(part):
    X = part[[
        "warehouse_id", "aisle", "section",
        "item_pick_count", "avg_picker_gap_s",
        # you may add static features known at pick time (e.g., item size/class, zone)
    ]].copy()
    y = part["relevance"].astype(int)
    q = part["order_id"].astype(str)
    return X, y, q

X_train, y_train, q_train = prep_features(train_df)
X_test,  y_test,  q_test  = prep_features(test_df)

# --- Fit categorical encoders on TRAIN only, transform TEST with same mapping ---
cats = ["warehouse_id", "aisle", "section"]
encoders = {}
for c in cats:
    le = LabelEncoder()
    X_train[c] = le.fit_transform(X_train[c].astype(str))
    # Map unseen labels in test to a fallback (here the first known class)
    fallback = le.classes_[0]
    X_test[c] = X_test[c].astype(str).map(lambda x: x if x in le.classes_ else fallback)
    X_test[c] = le.transform(X_test[c])
    encoders[c] = le

# --- Build LightGBM ranking datasets with proper groups ---
def to_group_sizes(q_series):
    # q_series must be in the same row order as X/y
    return q_series.value_counts().sort_index().reindex(q_series.sort_values().unique()).values

# Sort both splits by order_id to align groups
order_train = q_train.sort_values().unique()
order_test  = q_test.sort_values().unique()

idx_train = q_train.argsort(kind="mergesort")
idx_test  = q_test.argsort(kind="mergesort")

X_train = X_train.iloc[idx_train].reset_index(drop=True)
y_train = y_train.iloc[idx_train].reset_index(drop=True)
q_train = q_train.iloc[idx_train].reset_index(drop=True)

X_test  = X_test.iloc[idx_test].reset_index(drop=True)
y_test  = y_test.iloc[idx_test].reset_index(drop=True)
q_test  = q_test.iloc[idx_test].reset_index(drop=True)

group_train = q_train.value_counts().sort_index().values
group_test  = q_test.value_counts().sort_index().values

lgb_train = lgb.Dataset(
    X_train, label=y_train, group=group_train,
    categorical_feature=[X_train.columns.get_loc(c) for c in cats],
    free_raw_data=False
)
lgb_valid = lgb.Dataset(
    X_test, label=y_test, group=group_test, reference=lgb_train,
    categorical_feature=[X_test.columns.get_loc(c) for c in cats],
    free_raw_data=False
)

params = {
    "objective": "lambdarank",
    "metric": ["ndcg","map"],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "seed": 42,
    "verbosity": -1
}

model = lgb.train(
    params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train, lgb_valid],
    valid_names=["train","valid"],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)

# --- Proper evaluation: NDCG over ITEMS per order, not pairs ---
# Predict scores for items, then compute NDCG per order and average.
pred = model.predict(X_test, num_iteration=model.best_iteration)

eval_df = X_test.copy()
eval_df["order_id"] = q_test.values
eval_df["relevance"] = y_test.values
eval_df["score"] = pred

ndcgs = []
for oid, g in eval_df.groupby("order_id"):
    if len(g) <= 1:
        continue
    ndcgs.append(ndcg_score([g["relevance"].values], [g["score"].values]))

import numpy as np
print(f"NDCG (mean across test orders): {np.mean(ndcgs):.4f}  (orders: {len(ndcgs)})")

# Optional: Kendall-τ for direct rank agreement (requires scipy)
# from scipy.stats import kendalltau
# taus = []
# for oid, g in eval_df.groupby("order_id"):
#     if len(g) <= 1: continue
#     pred_order = g.sort_values("score", ascending=False).index
#     true_order = g.sort_values("relevance", ascending=False).index
#     taus.append(kendalltau(range(len(pred_order)), pd.Series(pred_order).rank().values).correlation)
# print("Kendall-τ (mean):", np.nanmean(taus))
# Time-based, leakage-safe item-level ranking with LGBMRanker
# - Splits by time (80/20)
# - Aggregates computed on TRAIN ONLY and merged to TEST with safe defaults
# - Item-level listwise ranking (one row per order line)
# - Evaluates NDCG and Kendall-τ per order
# Author: you :)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ndcg_score
from scipy.stats import kendalltau
from lightgbm import LGBMRanker

RANDOM_STATE = 42
CSV_PATH = "synthetic_warehouse_picks_sample.csv"  # <-- change if needed

# -----------------------------
# 1) Load & parse
# -----------------------------
df = pd.read_csv(CSV_PATH)
df["datetime_picked"] = pd.to_datetime(df["datetime_picked"], errors="coerce")
df = df.sort_values(["order_id", "datetime_picked"]).reset_index(drop=True)

def parse_aisle(tok: str):
    try:
        return tok.split("-")[1]  # "A03"
    except Exception:
        return None

def parse_section(tok: str):
    try:
        return tok.split("-")[2]  # "S015"
    except Exception:
        return None

df["aisle"]   = df["bin_id"].apply(parse_aisle)
df["section"] = df["bin_id"].apply(parse_section)

# True order → relevance (higher = earlier)
df["true_rank"]  = df.groupby("order_id")["datetime_picked"].rank(method="first").astype(int) - 1
df["group_size"] = df.groupby("order_id")["order_id"].transform("count")
df["relevance"]  = (df["group_size"] - df["true_rank"]).astype(int)

# -----------------------------
# 2) Time-based split (80% old → train, 20% recent → test)
# -----------------------------
cutoff = df["datetime_picked"].quantile(0.80)
train_df = df[df["datetime_picked"] <= cutoff].copy()
test_df  = df[df["datetime_picked"] >  cutoff].copy()

print(f"[Split] Train ≤ {cutoff:%Y-%m-%d %H:%M:%S}, Test > {cutoff:%Y-%m-%d %H:%M:%S}")
print(f"[Split] Train lines: {len(train_df):,} | Test lines: {len(test_df):,}")

# -----------------------------
# 3) Train-only aggregates (NO leakage)
# -----------------------------
# Average gap between picks for each picker (proxy for speed)
picker_avg_gap = (
    train_df.assign(delta_s=train_df.groupby("picker_id")["datetime_picked"].diff().dt.total_seconds())
            .groupby("picker_id")["delta_s"].mean()
            .rename("avg_picker_gap_s")
)

# Item popularity (count of occurrences in TRAIN only)
item_pop = (
    train_df["item_id"].value_counts()
            .rename_axis("item_id").rename("item_pick_count")
)

def attach_aggs(part: pd.DataFrame) -> pd.DataFrame:
    out = part.merge(picker_avg_gap, on="picker_id", how="left")
    out = out.merge(item_pop,       on="item_id",   how="left")
    # Safe defaults for unseen pickers/items in TEST
    out["avg_picker_gap_s"] = out["avg_picker_gap_s"].fillna(picker_avg_gap.median() if len(picker_avg_gap) else 10.0)
    out["item_pick_count"]  = out["item_pick_count"].fillna(0).astype(int)
    return out

train_df = attach_aggs(train_df)
test_df  = attach_aggs(test_df)

# -----------------------------
# 4) Feature preparation (item-level rows)
#    IMPORTANT: no features derived from the realized in-order timing of each line.
# -----------------------------
FEATURES = ["warehouse_id", "aisle", "section", "item_pick_count", "avg_picker_gap_s"]
TARGET   = "relevance"
GROUP    = "order_id"

def prep_Xyq(part: pd.DataFrame):
    X = part[FEATURES].copy()
    y = part[TARGET].astype(int).values
    q = part[GROUP].astype(str).values
    return X, y, q

X_train, y_train, q_train = prep_Xyq(train_df)
X_test,  y_test,  q_test  = prep_Xyq(test_df)

# -----------------------------
# 5) Encode categoricals on TRAIN, apply to TEST
# -----------------------------
cat_cols = ["warehouse_id", "aisle", "section"]
encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    X_train[c] = le.fit_transform(X_train[c].astype(str))
    # Map unseen labels in test to a fallback (first known class)
    fallback = le.classes_[0]
    X_test[c] = X_test[c].astype(str).map(lambda x: x if x in le.classes_ else fallback)
    X_test[c] = le.transform(X_test[c])
    encoders[c] = le

# -----------------------------
# 6) Build groups (one group per order). LGBMRanker expects group sizes.
#    Make sure row order and group arrangement align.
# -----------------------------
def sort_by_query(X, y, q):
    idx = np.argsort(q, kind="mergesort")
    return X.iloc[idx].reset_index(drop=True), y[idx], q[idx]

X_train, y_train, q_train = sort_by_query(X_train, y_train, q_train)
X_test,  y_test,  q_test  = sort_by_query(X_test,  y_test,  q_test)

def group_sizes(q):
    # counts in the order that groups appear after sorting
    _, counts = np.unique(q, return_counts=True)
    return counts.tolist()

group_train = group_sizes(q_train)
group_test  = group_sizes(q_test)

# -----------------------------
# 7) Train LGBMRanker (LambdaRank)
# -----------------------------
ranker = LGBMRanker(
    objective="lambdarank",
    boosting_type="gbdt",
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=30,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1
)

ranker.fit(
    X_train, y_train,
    group=group_train,
    eval_set=[(X_test, y_test)],
    eval_group=[group_test],
    eval_at=[5, 10]     # report at typical cutoffs (optional)
)

# -----------------------------
# 8) Predict & evaluate per-order: NDCG and Kendall-τ
# -----------------------------
scores = ranker.predict(X_test)

eval_df = X_test.copy()
eval_df["order_id"]  = q_test
eval_df["relevance"] = y_test
eval_df["score"]     = scores

ndcgs, taus = [], []
ndcgs_ge3, taus_ge3 = [], []

for oid, g in eval_df.groupby("order_id", sort=False):
    if len(g) <= 1:
        continue
    y_true = g["relevance"].values
    y_pred = g["score"].values

    nd = ndcg_score([y_true], [y_pred])
    # Kendall-τ over ranks (higher relevance should come earlier)
    # Sort indices by descending true/pred
    tau, _ = kendalltau(np.argsort(-y_true), np.argsort(-y_pred))

    ndcgs.append(nd)
    taus.append(tau)

    if len(g) >= 3:
        ndcgs_ge3.append(nd)
        taus_ge3.append(tau)

def mean_or_nan(arr):
    return float(np.mean(arr)) if len(arr) else float("nan")

print(f"[All test orders]   NDCG: {mean_or_nan(ndcgs):.4f} | Kendall-τ: {mean_or_nan(taus):.4f} | n={len(ndcgs)}")
print(f"[Orders size ≥ 3]   NDCG: {mean_or_nan(ndcgs_ge3):.4f} | Kendall-τ: {mean_or_nan(taus_ge3):.4f} | n={len(ndcgs_ge3)}")

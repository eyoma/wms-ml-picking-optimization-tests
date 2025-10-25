import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ndcg_score
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import kendalltau

# --- Load & parse ---
CSV_PATH = "synthetic_warehouse_picks_sample.csv"
df = pd.read_csv(CSV_PATH)
df["datetime_picked"] = pd.to_datetime(df["datetime_picked"], errors="coerce")
df = df.sort_values(["order_id", "datetime_picked"])

# Parse bin tokens like "W01-A03-S015"
def parse_aisle(tok):
    try: return tok.split("-")[1]
    except: return None

def parse_section(tok):
    try: return tok.split("-")[2]
    except: return None

df["aisle"] = df["bin_id"].apply(parse_aisle)
df["section"] = df["bin_id"].apply(parse_section)

# True sequence → relevance (higher = earlier pick)
df["true_rank"]  = df.groupby("order_id")["datetime_picked"].rank(method="first").astype(int) - 1
df["group_size"] = df.groupby("order_id")["order_id"].transform("count")
df["relevance"]  = (df["group_size"] - df["true_rank"]).astype(int)

# --- Time-based split ---
cutoff = df["datetime_picked"].quantile(0.8)  # 80% old → train
train_df = df[df["datetime_picked"] <= cutoff].copy()
test_df  = df[df["datetime_picked"] > cutoff].copy()

print(f"Training on data before {cutoff:%Y-%m-%d}, testing after.")

# --- Aggregates computed on TRAIN ONLY ---
picker_avg_gap = (
    train_df.assign(delta_s=train_df.groupby("picker_id")["datetime_picked"].diff().dt.total_seconds())
            .groupby("picker_id")["delta_s"].mean()
            .rename("avg_picker_gap_s")
)
item_pop = train_df["item_id"].value_counts().rename_axis("item_id").rename("item_pick_count")

def attach_aggs(part):
    out = part.merge(picker_avg_gap, on="picker_id", how="left")
    out = out.merge(item_pop, on="item_id", how="left")
    out["avg_picker_gap_s"] = out["avg_picker_gap_s"].fillna(picker_avg_gap.median() if len(picker_avg_gap)>0 else 10.0)
    out["item_pick_count"]  = out["item_pick_count"].fillna(0).astype(int)
    return out

train_df = attach_aggs(train_df)
test_df  = attach_aggs(test_df)

# --- Feature prep ---
def prep_features(part):
    X = part[["warehouse_id","aisle","section","item_pick_count","avg_picker_gap_s"]].copy()
    y = part["relevance"].astype(int)
    q = part["order_id"].astype(str)
    return X, y, q

X_train, y_train, q_train = prep_features(train_df)
X_test,  y_test,  q_test  = prep_features(test_df)

# --- Encode categoricals ---
cats = ["warehouse_id","aisle","section"]
encoders = {}
for c in cats:
    le = LabelEncoder()
    X_train[c] = le.fit_transform(X_train[c].astype(str))
    fallback = le.classes_[0]
    X_test[c]  = X_test[c].astype(str).map(lambda x: x if x in le.classes_ else fallback)
    X_test[c]  = le.transform(X_test[c])
    encoders[c] = le

# --- Train regressor ---
gbr = GradientBoostingRegressor(random_state=42, n_estimators=200, learning_rate=0.05, max_depth=3)
gbr.fit(X_train, y_train)

# --- Predict ---
pred = gbr.predict(X_test)

# --- Evaluate per order ---
eval_df = X_test.copy()
eval_df["order_id"] = q_test.values
eval_df["relevance"] = y_test.values
eval_df["score"] = pred

ndcgs, taus = [], []
for oid, g in eval_df.groupby("order_id"):
    if len(g) <= 1: continue
    y_true, y_pred = g["relevance"].values, g["score"].values
    ndcgs.append(ndcg_score([y_true], [y_pred]))
    # Kendall tau rank correlation between true and predicted order
    tau, _ = kendalltau(np.argsort(-y_true), np.argsort(-y_pred))
    taus.append(tau)

print(f"NDCG (mean across test orders): {np.mean(ndcgs):.4f} | "
      f"Kendall-τ (mean): {np.nanmean(taus):.4f}  (orders used: {len(ndcgs)})")

# pip install scikit-learn pandas numpy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ndcg_score
from sklearn.ensemble import GradientBoostingRegressor

CSV_PATH = "synthetic_warehouse_picks_sample.csv"  # adjust path if needed

# --- Load & parse ---
df = pd.read_csv(CSV_PATH)
df["datetime_picked"] = pd.to_datetime(df["datetime_picked"], errors="coerce")
df = df.sort_values(["order_id","datetime_picked"])

# Parse bin tokens like "W01-A03-S015"
def parse_aisle(tok):
    try: return tok.split("-")[1]
    except: return None

def parse_section(tok):
    try: return tok.split("-")[2]
    except: return None

df["aisle"] = df["bin_id"].apply(parse_aisle)
df["section"] = df["bin_id"].apply(parse_section)

# True sequence → create item-level relevance label (higher = earlier in sequence)
df["true_rank"]   = df.groupby("order_id")["datetime_picked"].rank(method="first").astype(int) - 1
df["group_size"]  = df.groupby("order_id")["order_id"].transform("count")
df["relevance"]   = (df["group_size"] - df["true_rank"]).astype(int)  # first picked gets largest relevance

# --- Split by ORDER (no leakage) ---
all_orders = df["order_id"].unique()
train_orders, test_orders = train_test_split(all_orders, test_size=0.2, random_state=42)
train_df = df[df["order_id"].isin(train_orders)].copy()
test_df  = df[df["order_id"].isin(test_orders)].copy()

# --- Aggregates computed on TRAIN ONLY, then joined to both ---
# (Avoid using per-line time deltas within an order; they leak the realized sequence.)
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

# --- Features (one row per order line); keep only info known pre-pick ---
def prep_features(part):
    X = part[["warehouse_id","aisle","section","item_pick_count","avg_picker_gap_s"]].copy()
    y = part["relevance"].astype(int)
    q = part["order_id"].astype(str)
    return X, y, q

X_train, y_train, q_train = prep_features(train_df)
X_test,  y_test,  q_test  = prep_features(test_df)

# --- Encode categoricals (fit on train, apply to test) ---
cats = ["warehouse_id","aisle","section"]
encoders = {}
for c in cats:
    le = LabelEncoder()
    X_train[c] = le.fit_transform(X_train[c].astype(str))
    fallback = le.classes_[0]
    X_test[c]  = X_test[c].astype(str).map(lambda x: x if x in le.classes_ else fallback)
    X_test[c]  = le.transform(X_test[c])
    encoders[c] = le

# --- Train a fast regressor to predict relevance (works fine for ranking eval) ---
gbr = GradientBoostingRegressor(random_state=123, n_estimators=200, max_depth=3, learning_rate=0.05)
gbr.fit(X_train, y_train)

# --- Predict & evaluate NDCG across items per order ---
pred = gbr.predict(X_test)
eval_df = X_test.copy()
eval_df["order_id"] = q_test.values
eval_df["relevance"] = y_test.values
eval_df["score"] = pred

ndcgs = []
for oid, g in eval_df.groupby("order_id"):
    if len(g) <= 1:    # skip degenerate orders
        continue
    ndcgs.append(ndcg_score([g["relevance"].values], [g["score"].values]))

print(f"NDCG (mean across test orders): {np.mean(ndcgs):.4f}  (orders used: {len(ndcgs)})")

# Optional sanity: remove very small orders (<3 lines) which can inflate metrics
ndcgs3 = []
for oid, g in eval_df.groupby("order_id"):
    if len(g) < 3:
        continue
    ndcgs3.append(ndcg_score([g["relevance"].values], [g["score"].values]))
if ndcgs3:
    print(f"NDCG, orders with ≥3 lines: {np.mean(ndcgs3):.4f}  (orders used: {len(ndcgs3)})")
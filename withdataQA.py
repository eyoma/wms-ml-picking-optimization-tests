# Load sample CSV; if not present, regenerate a small synthetic sample, then proceed with QA, stats, and plots.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math, random, re
# import caas_jupyter_tools

CSV_PATH = "./synthetic_warehouse_picks_sample.csv"

def regenerate_if_missing(path):
    if os.path.exists(path):
        return
    # --- Minimal synthetic generator (same schema) ---
    SEED = 42
    N_WAREHOUSES = 2
    AISLES_PER_WH = 12
    SLOTS_PER_AISLE = 25
    N_ITEMS = 800
    N_PICKERS = 60
    N_ORDERS = 1800
    ORDER_SIZE_LOGNORMAL_MEAN = 1.2
    ORDER_SIZE_LOGNORMAL_SIGMA = 0.8
    MAX_LINES_PER_ORDER = 25
    ZIPF_S = 1.15
    SIM_DAYS = 30
    BASE_DATE = datetime(2025, 9, 1, 6, 0, 0)
    PICKS_TARGET_ROWS = 10000
    AISLE_SPACING_M = 2.0
    SLOT_SPACING_M = 1.0
    WALK_SPEED_MPS = 1.2
    PICK_TIME_SEC_MEAN = 10.0
    PICK_TIME_SEC_SIGMA = 0.4
    WAREHOUSE_WEIGHTS = [0.6, 0.4]
    SHIFT_STARTS = [(6, 0), (14, 0)]

    rng = np.random.default_rng(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    def make_bin_id(wh_id, aisle, slot):
        return f"W{wh_id:02d}-A{aisle:02d}-S{slot:03d}"

    warehouses = [f"W{w+1:02d}" for w in range(N_WAREHOUSES)]
    all_bins = []
    bin_coords = {}
    for wi, wh in enumerate(warehouses, start=1):
        for a in range(1, AISLES_PER_WH+1):
            for s in range(1, SLOTS_PER_AISLE+1):
                bid = make_bin_id(wi, a, s)
                all_bins.append((wh, bid, a, s))
                bin_coords[(wh, bid)] = (a, s)

    items = [f"SKU{idx+1:05d}" for idx in range(N_ITEMS)]
    available_bins_by_wh = {wh: [b for (w,b,a,s) in all_bins if w == wh] for wh in warehouses}

    item_bins = {}
    wh_choices = rng.choice(warehouses, size=N_ITEMS, p=(np.array(WAREHOUSE_WEIGHTS)/np.sum(WAREHOUSE_WEIGHTS)))
    for i, sku in enumerate(items):
        wh = wh_choices[i]
        b = random.choice(available_bins_by_wh[wh])
        item_bins[sku] = (wh, b)

    ranks = np.arange(1, N_ITEMS+1)
    zipf_weights = 1.0 / (ranks ** ZIPF_S)
    zipf_weights = zipf_weights / zipf_weights.sum()
    sku_prob = zipf_weights

    pickers = [f"PICKER{p+1:03d}" for p in range(N_PICKERS)]
    picker_experience = rng.uniform(0.85, 1.15, size=N_PICKERS)

    def manhattan_distance(a1, s1, a2, s2):
        return abs(a1 - a2) * AISLE_SPACING_M + abs(s1 - s2) * SLOT_SPACING_M
    def travel_time_sec(a1, s1, a2, s2):
        dist = manhattan_distance(a1, s1, a2, s2)
        return dist / WALK_SPEED_MPS
    def lognormal_seconds(mean, sigma, size=1, factor=1.0):
        mu = math.log(max(mean, 0.1))
        vals = np.random.lognormal(mean=mu, sigma=sigma, size=size) * factor
        return vals

    def aisle_transition_scores(curr_aisle, candidate_coords):
        scores = []
        for (a, s) in candidate_coords:
            d_aisle = abs(a - curr_aisle)
            if d_aisle == 0:
                base = 1.0
            elif d_aisle == 1:
                base = 0.6
            elif d_aisle == 2:
                base = 0.35
            else:
                base = 0.2 / (d_aisle)
            scores.append(base)
        arr = np.array(scores)
        return arr / arr.sum() if arr.sum() > 0 else np.ones(len(scores))/len(scores)

    def generate_order_lines(n_desired):
        size = int(max(1, min(MAX_LINES_PER_ORDER, round(np.random.lognormal(ORDER_SIZE_LOGNORMAL_MEAN, ORDER_SIZE_LOGNORMAL_SIGMA)))))
        size = min(size, n_desired) if n_desired is not None else size
        sku_idx = rng.choice(np.arange(N_ITEMS), size=size, p=sku_prob, replace=True)
        skus = [items[i] for i in sku_idx]
        return skus

    def sample_order_start():
        day_offset = rng.integers(0, SIM_DAYS)
        shift = random.choice(SHIFT_STARTS)
        hour, minute = shift
        jitter_min = rng.integers(0, 180)
        start_dt = BASE_DATE + timedelta(days=int(day_offset), hours=int(hour), minutes=int(minute + jitter_min))
        return start_dt

    rows = []
    order_id_counter = 1
    picks_generated = 0
    while order_id_counter <= N_ORDERS and picks_generated < PICKS_TARGET_ROWS:
        wh = random.choices(warehouses, weights=WAREHOUSE_WEIGHTS, k=1)[0]
        p_idx = rng.integers(0, N_PICKERS)
        picker = pickers[p_idx]
        exp_factor = picker_experience[p_idx]
        skus = generate_order_lines(n_desired=MAX_LINES_PER_ORDER)
        skus = [sku for sku in skus if item_bins[sku][0] == wh]
        if len(skus) == 0:
            continue
        bins_for_lines = [item_bins[sku][1] for sku in skus]
        coords = [bin_coords[(wh, b)] for b in bins_for_lines]

        current = (1, 1)
        start_time = sample_order_start()
        t = start_time
        remaining = list(range(len(bins_for_lines)))

        while remaining:
            candidate_indices = remaining
            candidate_coords = [coords[i] for i in candidate_indices]
            aisle_scores = aisle_transition_scores(current[0], candidate_coords)
            distances = np.array([abs(current[0] - a)*2.0 + abs(current[1] - s)*1.0 for (a, s) in candidate_coords])
            inv = 1.0 / (1.0 + distances)
            combined = 0.65 * aisle_scores + 0.35 * (inv / inv.sum())
            choice_idx = rng.choice(np.arange(len(candidate_indices)), p=combined)
            pick_i = candidate_indices[choice_idx]

            a2, s2 = coords[pick_i]
            travel = (abs(current[0] - a2)*2.0 + abs(current[1] - s2)*1.0) / 1.2
            pick_time = float(lognormal_seconds(10.0, 0.4, size=1, factor=exp_factor)[0])
            t = t + timedelta(seconds=float(travel) + float(pick_time))

            rows.append({
                "order_id": f"ORD{order_id_counter:06d}",
                "item_id": skus[pick_i],
                "datetime_picked": t.isoformat(timespec="seconds"),
                "picker_id": picker,
                "warehouse_id": wh,
                "bin_id": bins_for_lines[pick_i]
            })

            picks_generated += 1
            current = (a2, s2)
            remaining.remove(pick_i)
            if picks_generated >= PICKS_TARGET_ROWS:
                break
        order_id_counter += 1

    pd.DataFrame(rows).to_csv(path, index=False)

# Ensure CSV exists
regenerate_if_missing(CSV_PATH)

# ---------- Load data ----------
df = pd.read_csv(CSV_PATH)
df['datetime_picked'] = pd.to_datetime(df['datetime_picked'], errors='coerce')

# ---------- QA checks ----------
duplicate_rows = int(df.duplicated().sum())
missing_counts = df.isna().sum()

df_sorted = df.sort_values(['order_id','datetime_picked'])
df_sorted['delta_s'] = df_sorted.groupby('order_id')['datetime_picked'].diff().dt.total_seconds()

non_increasing_rows = int((df_sorted['delta_s'] < 0).sum())
very_large_gap_rows = int((df_sorted['delta_s'] > 2*60*60).sum())

# ---------- Summary stats ----------
num_orders = df['order_id'].nunique()
lines_per_order = df.groupby('order_id').size().rename('lines_per_order')
lines_per_order_desc = lines_per_order.describe()

avg_delta_overall = df_sorted['delta_s'].dropna().mean()
avg_delta_per_order = df_sorted.groupby('order_id')['delta_s'].mean().dropna().mean()

agg_per_warehouse = df.groupby('warehouse_id').agg(
    unique_pickers=('picker_id','nunique'),
    unique_bins=('bin_id','nunique'),
    lines=('bin_id','count')
).reset_index()

qa_table = pd.DataFrame({
    'metric': [
        'duplicate_rows',
        'missing_order_id',
        'missing_item_id',
        'missing_datetime_picked',
        'missing_picker_id',
        'missing_warehouse_id',
        'missing_bin_id',
        'non_increasing_rows_within_order',
        'very_large_gap_rows_(>2h)'
    ],
    'value': [
        duplicate_rows,
        int(missing_counts.get('order_id', 0)),
        int(missing_counts.get('item_id', 0)),
        int(missing_counts.get('datetime_picked', 0)),
        int(missing_counts.get('picker_id', 0)),
        int(missing_counts.get('warehouse_id', 0)),
        int(missing_counts.get('bin_id', 0)),
        non_increasing_rows,
        very_large_gap_rows
    ]
})

summary_table = pd.DataFrame({
    'metric': [
        'num_orders',
        'lines_per_order_mean',
        'lines_per_order_std',
        'lines_per_order_min',
        'lines_per_order_25%',
        'lines_per_order_50%',
        'lines_per_order_75%',
        'lines_per_order_max',
        'avg_time_between_picks_overall_seconds',
        'avg_time_between_picks_mean_per_order_seconds'
    ],
    'value': [
        num_orders,
        float(lines_per_order_desc['mean']),
        float(lines_per_order_desc['std']),
        int(lines_per_order_desc['min']),
        float(lines_per_order_desc['25%']),
        float(lines_per_order_desc['50%']),
        float(lines_per_order_desc['75%']),
        int(lines_per_order_desc['max']),
        float(avg_delta_overall) if pd.notna(avg_delta_overall) else np.nan,
        float(avg_delta_per_order) if pd.notna(avg_delta_per_order) else np.nan
    ]
})

# Display QA tables
# caas_jupyter_tools.display_dataframe_to_user("QA Checks", qa_table)
# caas_jupyter_tools.display_dataframe_to_user("Summary Stats", summary_table)
# caas_jupyter_tools.display_dataframe_to_user("Unique Pickers & Bins per Warehouse", agg_per_warehouse)

# ---------- Plot a few order sequences ----------
def parse_aisle_slot(bin_id):
    m = re.search(r"A(\d+)-S(\d+)$", bin_id)
    if m:
        return int(m.group(1)), int(m.group(2))
    return np.nan, np.nan

parsed = df_sorted['bin_id'].apply(parse_aisle_slot)
df_sorted['aisle'] = [p[0] for p in parsed]
df_sorted['slot'] = [p[1] for p in parsed]

lines_per_order_sorted = df_sorted.groupby('order_id').size().rename('lines_per_order')
candidates = lines_per_order_sorted[lines_per_order_sorted >= 6].index.tolist()
sample_orders = candidates[:3] if len(candidates) >= 3 else lines_per_order_sorted.sort_values(ascending=False).index[:3].tolist()

for oid in sample_orders:
    sub = df_sorted[df_sorted['order_id'] == oid].copy()
    # Plot aisle over time
    plt.figure(figsize=(8, 4.5))
    plt.plot(sub['datetime_picked'], sub['aisle'], marker='o')
    plt.title(f"Order {oid} — Aisle over time")
    plt.xlabel("Time")
    plt.ylabel("Aisle")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

    # Plot slot over time
    plt.figure(figsize=(8, 4.5))
    plt.plot(sub['datetime_picked'], sub['slot'], marker='o')
    plt.title(f"Order {oid} — Slot over time")
    plt.xlabel("Time")
    plt.ylabel("Slot")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

CSV_PATH

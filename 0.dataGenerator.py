# Synthetic warehouse picking dataset generator (parameterized) and sample CSV export
# Generates a realistic-ish dataset with columns:
# order_id, item_id, datetime_picked, picker_id, warehouse_id, bin_id
#
# You can tweak parameters in the CONFIG block below and rerun.

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math
import random

# ---------------------------------
# CONFIG (feel free to tweak)
# ---------------------------------
SEED = 42
N_WAREHOUSES = 2
AISLES_PER_WH = 12         # aisles per warehouse
SLOTS_PER_AISLE = 25       # pick faces per aisle
N_ITEMS = 800              # catalog size
N_PICKERS = 60
N_ORDERS = 5000            # number of orders to simulate
ORDER_SIZE_LOGNORMAL_MEAN = 1.2  # controls average lines per order (~exp of this)
ORDER_SIZE_LOGNORMAL_SIGMA = 0.8
MAX_LINES_PER_ORDER = 25
ZIPF_S = 1.15              # skew for item popularity (higher = more long-tail)
SIM_DAYS = 30              # simulate over this many days
BASE_DATE = datetime(2025, 9, 1, 6, 0, 0)  # start of simulation window
PICKS_TARGET_ROWS = 20000  # stop early once we pass this many pick lines
# Timing / geometry
AISLE_SPACING_M = 2.0
SLOT_SPACING_M = 1.0
WALK_SPEED_MPS = 1.2
PICK_TIME_SEC_MEAN = 10.0
PICK_TIME_SEC_SIGMA = 0.4   # lognormal noise
# Probabilities
WAREHOUSE_WEIGHTS = [0.6, 0.4]  # demand mix across warehouses
SHIFT_STARTS = [(6, 0), (14, 0)]  # 6am and 2pm starts
# ---------------------------------

rng = np.random.default_rng(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Build warehouses, aisles, slots -> bins
def make_bin_id(wh_id, aisle, slot):
    return f"W{wh_id:02d}-A{aisle:02d}-S{slot:03d}"

warehouses = [f"W{w+1:02d}" for w in range(N_WAREHOUSES)]
all_bins = []
bin_coords = {}  # (x=aisle_index, y=slot_index) for simple Manhattan distance
for wi, wh in enumerate(warehouses, start=1):
    for a in range(1, AISLES_PER_WH+1):
        for s in range(1, SLOTS_PER_AISLE+1):
            bid = make_bin_id(wi, a, s)
            all_bins.append((wh, bid, a, s))
            bin_coords[(wh, bid)] = (a, s)

# Assign each item to one (warehouse, bin)
items = [f"SKU{idx+1:05d}" for idx in range(N_ITEMS)]
item_bins = {}
# randomly allocate items across warehouses and bins, allowing multiple items per bin (realistic)
available_bins_by_wh = {}
for wh in warehouses:
    available_bins_by_wh[wh] = [b for (w,b,a,s) in all_bins if w == wh]

wh_choices = rng.choice(warehouses, size=N_ITEMS, p=(np.array(WAREHOUSE_WEIGHTS)/np.sum(WAREHOUSE_WEIGHTS)))
for i, sku in enumerate(items):
    wh = wh_choices[i]
    b = random.choice(available_bins_by_wh[wh])
    item_bins[sku] = (wh, b)

# Build a Zipf popularity distribution over SKUs
ranks = np.arange(1, N_ITEMS+1)
zipf_weights = 1.0 / (ranks ** ZIPF_S)
zipf_weights = zipf_weights / zipf_weights.sum()
sku_by_popularity = items  # implicitly ranked by index
# We'll sample SKU indices by this distribution
sku_prob = zipf_weights

# Picker pool with experience levels affecting pick time
pickers = [f"PICKER{p+1:03d}" for p in range(N_PICKERS)]
picker_experience = rng.uniform(0.85, 1.15, size=N_PICKERS)  # <1 faster, >1 slower

def manhattan_distance(a1, s1, a2, s2):
    return abs(a1 - a2) * AISLE_SPACING_M + abs(s1 - s2) * SLOT_SPACING_M

def travel_time_sec(a1, s1, a2, s2):
    dist = manhattan_distance(a1, s1, a2, s2)
    return dist / WALK_SPEED_MPS

def lognormal_seconds(mean, sigma, size=1, factor=1.0):
    # mean is median-ish; use lognormal with log-mean ln(mean)
    mu = math.log(max(mean, 0.1))
    vals = np.random.lognormal(mean=mu, sigma=sigma, size=size) * factor
    return vals

# Simple aisle-level Markov inclination: prefer staying in same aisle or adjacent
def aisle_transition_scores(curr_aisle, candidate_coords):
    scores = []
    for (a, s) in candidate_coords:
        d_aisle = abs(a - curr_aisle)
        # score higher if same aisle, slightly lower if adjacent, else decays
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
    # Draw order size from lognormal, cap at MAX_LINES_PER_ORDER and at least 1
    size = int(max(1, min(MAX_LINES_PER_ORDER, round(np.random.lognormal(ORDER_SIZE_LOGNORMAL_MEAN, ORDER_SIZE_LOGNORMAL_SIGMA)))))
    size = min(size, n_desired) if n_desired is not None else size
    # Sample items by popularity
    sku_idx = rng.choice(np.arange(N_ITEMS), size=size, p=sku_prob, replace=True)
    skus = [items[i] for i in sku_idx]
    return skus

rows = []
order_id_counter = 1
picks_generated = 0

# Pre-generate order start times across SIM_DAYS using two shifts
def sample_order_start():
    day_offset = rng.integers(0, SIM_DAYS)
    shift = random.choice(SHIFT_STARTS)
    hour, minute = shift
    # jitter within 3 hours of shift start
    jitter_min = rng.integers(0, 180)
    start_dt = BASE_DATE + timedelta(days=int(day_offset), hours=int(hour), minutes=int(minute + jitter_min))
    return start_dt

while order_id_counter <= N_ORDERS and picks_generated < PICKS_TARGET_ROWS:
    wh = random.choices(warehouses, weights=WAREHOUSE_WEIGHTS, k=1)[0]
    # choose a picker
    p_idx = rng.integers(0, N_PICKERS)
    picker = pickers[p_idx]
    exp_factor = picker_experience[p_idx]
    # generate order lines
    skus = generate_order_lines(n_desired=MAX_LINES_PER_ORDER)
    # filter to items stored in this warehouse
    skus = [sku for sku in skus if item_bins[sku][0] == wh]
    if len(skus) == 0:
        continue
    # Remaining bins for the order (one line per sku occurrence)
    bins_for_lines = [item_bins[sku][1] for sku in skus]
    coords = [bin_coords[(wh, b)] for b in bins_for_lines]

    # Sequence: start at entrance (aisle=1, slot=1) and pick greedily with aisle-biased Markov scores
    current = (1, 1)
    start_time = sample_order_start()
    t = start_time

    remaining = list(range(len(bins_for_lines)))
    sequence = []
    while remaining:
        candidate_indices = remaining
        candidate_coords = [coords[i] for i in candidate_indices]
        # aisle-biased preference
        aisle_scores = aisle_transition_scores(current[0], candidate_coords)
        # distance penalty (closer is better)
        distances = np.array([manhattan_distance(current[0], current[1], a, s) for (a, s) in candidate_coords])
        # convert distances to a softmax-like preference (smaller distance => higher weight)
        if len(distances) == 0:
            break
        inv = 1.0 / (1.0 + distances)
        # Combine aisle preference and proximity
        combined = 0.65 * aisle_scores + 0.35 * (inv / inv.sum())
        # sample next index by combined preference
        choice_idx = rng.choice(np.arange(len(candidate_indices)), p=combined)
        pick_i = candidate_indices[choice_idx]

        sequence.append(pick_i)
        # compute travel time
        a2, s2 = coords[pick_i]
        travel = travel_time_sec(current[0], current[1], a2, s2)
        # compute pick/handle time (experience-adjusted)
        pick_time = float(lognormal_seconds(PICK_TIME_SEC_MEAN, PICK_TIME_SEC_SIGMA, size=1, factor=exp_factor)[0])

        t = t + timedelta(seconds=travel + pick_time)
        # record
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

df = pd.DataFrame(rows, columns=["order_id","item_id","datetime_picked","picker_id","warehouse_id","bin_id"])

# Save to CSV
out_path = "./synthetic_warehouse_picks_sample.csv"
df.to_csv(out_path, index=False)


out_path

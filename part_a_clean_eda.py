####################################part_a_clean_eda.py###########################################
# part_a_clean_eda.py. Read data/raw_shipments.csv and data/hubs.csv; call your two functions;
# add the target is_delayed (1 when delivery_hours > promised_hours, else 0);
# write the result to data/modelling_table.csv (all columns, one row per unique shipment).

import os
import numpy as np
import pandas as pd
#from src.shipment_prep import clean_shipments, build_features, FEATURES, PROMISED_HOURS

os.makedirs("outputs", exist_ok=True)
raw_ship_df = pd.read_csv("/content/data/raw_shipments.csv")
hubs_df = pd.read_csv("/content/data/hubs.csv")

# before metrics after deduping and before any fillings
dedup = raw_ship_df.drop_duplicates()
dedup["distance_km"].isna().sum()

clean_df , medians = clean_shipments(raw_ship_df)
merged_clean_hubs_df = build_features(clean_df, hubs_df)
merged_clean_hubs_df["is_delayed"] = (merged_clean_hubs_df["delivery_hours"] > merged_clean_hubs_df["promised_hours"]).astype(int)
merged_clean_hubs_df.to_csv("/content/data/modelling_table.csv")

h = merged_clean_hubs_df["delivery_hours"]
q1, q3 = h.quantile(0.25), h.quantile(0.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr


metrics = [
    ("raw_rows", len(raw_ship_df)),
    ("duplicate_rows_removed", len(raw_ship_df) - len(dedup)),
    ("rows_after_dedup", len(dedup)),
    ("missing_distance_km_before", int(dedup["distance_km"].isna().sum())),
    ("missing_parcel_weight_before", int(dedup["parcel_weight_kg"].isna().sum())),
    ("sentinel_weight_rows", int((dedup["parcel_weight_kg"] > 100).sum())),
    ("missing_declared_value_before", int(dedup["declared_value"].isna().sum())),
    ("missing_payment_mode_before", int(dedup["payment_mode"].isna().sum())),
    ("service_type_spellings_before", int(dedup["service_type"].nunique())),
    ("service_type_spellings_after", int(clean_df["service_type"].nunique())),
    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),
    ("rows_after_hub_merge", len(merged_clean_hubs_df)),
    ("nulls_in_features_and_target", int(merged_clean_hubs_df[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum())),
    ("delayed_count", int(merged_clean_hubs_df["is_delayed"].sum())),
    ("delay_rate", merged_clean_hubs_df["is_delayed"].mean()),
    ("hours_q1", q1),
    ("hours_q3", q3),
    ("hours_lower_bound", lo),
    ("hours_upper_bound", hi),
    ("hours_outlier_count", int(((h < lo) | (h > hi)).sum())),
    ("declared_value_mean_inr", merged_clean_hubs_df["declared_value_inr"].mean()),
    ("declared_value_median_inr", merged_clean_hubs_df["declared_value_inr"].median()),
    ("log_declared_value_mean", merged_clean_hubs_df["log_declared_value"].mean()),
    ("log_declared_value_median", merged_clean_hubs_df["log_declared_value"].median()),
]

checks = pd.DataFrame(metrics, columns=["metric", "value"])
checks["value"] = checks["value"].astype(float).round(4)
checks.to_csv("outputs/part_a_checks.csv", index=False)

# delay evidence by group
merged_clean_hubs_df["distance_band"] = pd.cut(merged_clean_hubs_df["distance_km"], bins=[0, 200, 400, 600, 2000],
                                labels=["0-200", "201-400", "401-600", "601+"], right=True)
rows = []
for dim in ["service_type", "hub_tier", "distance_band"]:
    g = merged_clean_hubs_df.groupby(dim, observed=True).agg(shipments=("is_delayed", "size"), delayed=("is_delayed", "sum"),
                                              delay_rate=("is_delayed", "mean"), mean_hours=("delivery_hours", "mean"))
    for grp, r in g.iterrows():
        rows.append([dim, str(grp), int(r["shipments"]), int(r["delayed"]), r["delay_rate"], r["mean_hours"]])
by_group = pd.DataFrame(rows, columns=["dimension", "group", "shipments", "delayed", "delay_rate", "mean_hours"])
by_group[["delay_rate", "mean_hours"]] = by_group[["delay_rate", "mean_hours"]].round(4)
by_group.to_csv("outputs/part_a_delay_by_group.csv", index=False)

# correlations of numeric columns
corr = merged_clean_hubs_df.drop(columns=["distance_band"]).select_dtypes(include=[np.number]).corr().round(4)
corr.to_csv("outputs/part_a_correlations.csv")
print("Part A done:", len(merged_clean_hubs_df), "rows; delayed =", int(merged_clean_hubs_df["is_delayed"].sum()))
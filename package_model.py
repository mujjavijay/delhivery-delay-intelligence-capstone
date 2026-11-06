########################package_model.py#########################################

# 5. Package the model (package_model.py). Read outputs/part_c_final_config.csv, rebuild Part C's stratified training/validation/test split with the same settings, fit the final Pipeline on the training rows, and save with joblib: models/delay_model.pkl; models/delay_threshold.csv (one column threshold); models/feature_order.csv (one column feature, the 17 names in FEATURES order); and models/impute_values.csv (columns column, median — the three medians from clean_shipments). Then reload the pickle with joblib.load, predict probabilities for the 180 test rows with both the reloaded and the in-memory pipeline, and write outputs/part_d_reload_check.csv (metric,value) with test_rows_scored, reload_max_abs_probability_diff and threshold.
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from src.shipment_prep import FEATURES, clean_shipments


os.makedirs("models", exist_ok=True)

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

cfg = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
max_depth = None if str(cfg["max_depth"]) == "unlimited" else int(cfg["max_depth"])

table = pd.read_csv("data/modelling_table.csv")
X, y = table[FEATURES], table["is_delayed"]
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)

model = Pipeline([("selector", SelectKBest(f_classif, k=int(cfg["selector_k"]))),
                  ("model", RandomForestClassifier(random_state=42, n_estimators=int(cfg["n_estimators"]),
                                                   max_depth=max_depth, min_samples_leaf=int(cfg["min_samples_leaf"])))])
model.fit(X_train, y_train)

joblib.dump(model, "/content/models/delay_model.pkl")
pd.DataFrame({"threshold": [float(cfg["threshold"])]}).to_csv("models/delay_threshold.csv", index=False)
pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)
_, medians = clean_shipments(pd.read_csv("data/raw_shipments.csv"))
pd.DataFrame({"column": list(medians), "median": [round(v, 4) for v in medians.values()]}) \
    .to_csv("models/impute_values.csv", index=False)

reloaded = joblib.load("models/delay_model.pkl")
diff = np.abs(reloaded.predict_proba(X_test)[:, 1] - model.predict_proba(X_test)[:, 1]).max()
pd.DataFrame({"metric": ["test_rows_scored", "reload_max_abs_probability_diff", "threshold"],
              "value": [len(X_test), diff, float(cfg["threshold"])]}).round(4) \
    .to_csv("outputs/part_d_reload_check.csv", index=False)
print("packaged; max diff =", diff)
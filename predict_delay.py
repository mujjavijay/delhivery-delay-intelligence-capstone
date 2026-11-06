########################predict_delay.py#########################################

# 6. Scoring script (predict_delay.py). Load data/new_shipments.csv, data/hubs.csv, the four files in models/ and outputs/hub_segments.csv; clean the new parcels with your own clean_shipments, passing the stored medians; attach features with your own build_features; arrange the columns in the order of feature_order.csv; compute delay_probability (positive class, rounded to 4 decimals); set flagged = 1 when the probability is ≥ the stored threshold; join the hub segment; and set action to Escalate to hub manager when the parcel is flagged and its hub's segment is Escalate, Proactive customer call when it is flagged otherwise, and No action when it is not flagged. Write outputs/new_shipment_actions.csv with columns shipment_id, hub_id, segment, delay_probability, flagged, action, one row per new parcel, in file order. Running the script twice must produce byte-identical output.
import joblib
import pandas as pd
from src.shipment_prep import clean_shipments, build_features

new_raw = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
model = joblib.load("models/delay_model.pkl")
threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
imp = pd.read_csv("models/impute_values.csv")
medians = {r["column"]: float(r["median"]) for _, r in imp.iterrows()}
segments = pd.read_csv("outputs/hub_segments.csv")[["hub_id", "segment"]]

clean, _ = clean_shipments(new_raw, medians=medians)
feats = build_features(clean, hubs)
X = feats[feature_order]
assert X.isna().sum().sum() == 0, "missing values left in features"

out = feats[["shipment_id", "hub_id"]].copy()
out["delay_probability"] = model.predict_proba(X)[:, 1].round(4)
out["flagged"] = (out["delay_probability"] >= threshold).astype(int)
out = out.merge(segments, on="hub_id", how="left")
out["action"] = "No action"
out.loc[out["flagged"] == 1, "action"] = "Proactive customer call"
out.loc[(out["flagged"] == 1) & (out["segment"] == "Escalate"), "action"] = "Escalate to hub manager"
out = out[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
out.to_csv("outputs/new_shipment_actions.csv", index=False)
print(out["action"].value_counts().to_string())
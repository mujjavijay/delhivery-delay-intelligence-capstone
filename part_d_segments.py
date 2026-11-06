# ##########################################################part_d_segments.py##############################

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

table = pd.read_csv("data/modelling_table.csv")
#table = pd.read_csv("/content/data/modelling_table.csv")
hubs = pd.read_csv("data/hubs.csv")
#hubs = pd.read_csv("/content/data/hubs.csv")

# 1. Hub profile (part_d_segments.py). From data/modelling_table.csv, group by hub_id and compute shipments (count), mean_delivery_hours, delay_rate (share of delayed parcels), mean_distance_km and mean_n_stops. Write outputs/hub_profile.csv (24 rows, sorted by ascending hub_id; columns hub_id plus those five).
hub_profile = table.groupby("hub_id").agg(
    shipments=("shipment_id", "count"), mean_delivery_hours=("delivery_hours", "mean"),
    delay_rate=("is_delayed", "mean"), mean_distance_km=("distance_km", "mean"),
    mean_n_stops=("n_stops", "mean")).reset_index().sort_values("hub_id").reset_index(drop=True)
hub_profile.round(4).to_csv("outputs/hub_profile.csv", index=False)

# 2. K-Means segmentation. Scale the four profile columns mean_delivery_hours, delay_rate, mean_distance_km, mean_n_stops with StandardScaler (distance-based clustering needs comparable scales) and fit KMeans(n_clusters=3, random_state=42, n_init=10). Fixed order of the hubs: the hub profile is sorted by ascending hub_id (HUB01 first, HUB24 last — the order groupby returns and the order of hub_profile.csv) before it is scaled and clustered, so the 24 hubs always enter StandardScaler and KMeans in that order; the four profile columns may be in any order. K = 3 is fixed by the business — operations runs exactly three playbooks.
COLS = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
Z = StandardScaler().fit_transform(hub_profile[COLS])
km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Z)

# 3. Separation evidence. For K in 2, 3, 4, 5 (each KMeans(n_clusters=K, random_state=42, n_init=10) on the same scaled matrix, hubs in ascending hub_id order) compute the across-to-within distance ratio: the mean Euclidean distance over all unordered hub pairs that sit in different clusters divided by the mean over pairs in the same cluster, measured on the scaled matrix (276 pairs). Write outputs/part_d_separation.csv with columns K, across_over_within, cluster_sizes, where cluster_sizes is the cluster sizes in descending order joined by hyphens (for example 9-8-7).
def across_over_within(Z, labels):
    d = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2))
    iu = np.triu_indices(len(Z), k=1)
    dist, same = d[iu], (labels[:, None] == labels[None, :])[iu]
    return dist[~same].mean() / dist[same].mean()
sep = []
for K in [2, 3, 4, 5]:
    lab = KMeans(n_clusters=K, random_state=42, n_init=10).fit(Z).labels_
    sizes = sorted(np.bincount(lab), reverse=True)
    sep.append({"K": K, "across_over_within": across_over_within(Z, lab),
                "cluster_sizes": "-".join(str(int(s)) for s in sizes)})
pd.DataFrame(sep).round(4).to_csv("outputs/part_d_separation.csv", index=False)  

# 4. Segment names and tier check. Order the three clusters by the mean of their hubs' delay_rate and name them Standard (lowest), Watch (middle) and Escalate (highest). Write outputs/hub_segments.csv (hub_id, segment, hub_city, hub_tier), outputs/segment_profile.csv (segment, hubs, mean_delivery_hours, delay_rate, mean_distance_km, mean_n_stops — means of the hub-level values) and outputs/tier_vs_segment_mismatch.csv (the rows of hub_segments.csv whose segment differs from the one their tier implies: Tier 1 → Standard, Tier 2 → Watch, Tier 3 → Escalate).
hub_profile["cluster"] = km.labels_
order = hub_profile.groupby("cluster")["delay_rate"].mean().sort_values().index.tolist()
names = dict(zip(order, ["Standard", "Watch", "Escalate"]))
hub_profile["segment"] = hub_profile["cluster"].map(names)

seg = hub_profile[["hub_id", "segment"]].merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")
seg.to_csv("outputs/hub_segments.csv", index=False)

seg_prof = (hub_profile.groupby("segment").agg(hubs=("hub_id", "count"), mean_delivery_hours=("mean_delivery_hours", "mean"),
                                           delay_rate=("delay_rate", "mean"), mean_distance_km=("mean_distance_km", "mean"),
                                           mean_n_stops=("mean_n_stops", "mean")).reset_index())
seg_prof["segment"] = pd.Categorical(seg_prof["segment"], ["Standard", "Watch", "Escalate"])
seg_prof = seg_prof.sort_values("segment")
seg_prof.round(4).to_csv("outputs/segment_profile.csv", index=False)

implied = {1: "Standard", 2: "Watch", 3: "Escalate"}
seg[seg["segment"] != seg["hub_tier"].map(implied)].to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)
print(open("outputs/part_d_separation.csv").read())

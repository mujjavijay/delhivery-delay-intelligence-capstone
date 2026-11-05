PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

def clean_shipments(raw, medians=None):
  # 1. drop exact duplicate rows;
    total_dups = raw_ship_df.duplicated().sum()
    print(f"{total_dups}")
    raw_ship_df_dups_dropped = raw_ship_df.drop_duplicates()
    raw_ship_df_dups_dropped.shape

  # 2. service_type: strip leading and trailing spaces and convert to title case, so exactly three spellings remain (Express, Standard, Economy);
    raw_ship_df_dups_dropped["service_type"].unique()
    raw_ship_df_dups_dropped["service_type"] = raw_ship_df_dups_dropped["service_type"].str.strip().str.title()
    raw_ship_df_dups_dropped["service_type"].unique()

  # 3. payment_mode: replace a missing value with the placeholder "Unknown";
    raw_ship_df_dups_dropped["payment_mode"].isna().sum()
    raw_ship_df_dups_dropped["payment_mode"] = raw_ship_df_dups_dropped["payment_mode"].fillna("Unknown")
    raw_ship_df_dups_dropped["payment_mode"].isna().sum()

  # 4. declared_value: remove the rupee sign and the thousands commas, convert to a number, store it in a new column declared_value_inr, and drop the text column;
    raw_ship_df_dups_dropped["declared_value"].unique()
    raw_ship_df_dups_dropped["declared_value_inr"] = raw_ship_df_dups_dropped["declared_value"].str.replace("₹","",regex=False).str.replace(",","",regex=False).astype(float)
    raw_ship_df_dups_dropped = raw_ship_df_dups_dropped.drop(columns = ["declared_value"])

  # 5. dispatch_time: convert to a datetime;
    raw_ship_df_dups_dropped['dispatch_time'] = pd.to_datetime(raw_ship_df_dups_dropped['dispatch_time'])
    raw_ship_df_dups_dropped.info()

  # 6. parcel_weight_kg: any value above 100 is a scanner-fault sentinel (the network accepts parcels up to 40 kg) — treat it as missing;
    (raw_ship_df_dups_dropped['parcel_weight_kg'] > 100).sum()
    raw_ship_df_dups_dropped.loc[raw_ship_df_dups_dropped["parcel_weight_kg"] > 100,"parcel_weight_kg"] = np.nan
    (raw_ship_df_dups_dropped['parcel_weight_kg'] > 100).sum()

  # 7. fill missing distance_km, parcel_weight_kg and declared_value_inr with that column's median. If medians is None, compute the three medians from the non-missing values left after steps 1–6; otherwise use the dictionary passed in. The returned medians is a dictionary with exactly the keys distance_km, parcel_weight_kg, declared_value_inr and float values.
    if medians == None:
      distance_km_median = raw_ship_df_dups_dropped['distance_km'].median()
      print(distance_km_median)
      parcel_weight_kg_median = raw_ship_df_dups_dropped['parcel_weight_kg'].median()
      print(parcel_weight_kg_median)
      declared_value_inr_median = raw_ship_df_dups_dropped['declared_value_inr'].median()
      print(declared_value_inr_median)
      medians = {
          "distance_km":round(distance_km_median,2),
          "parcel_weight_kg":round(parcel_weight_kg_median,2),
          "declared_value_inr":round(declared_value_inr_median,2)
      }
    print(medians)
    raw_ship_df_dups_dropped['distance_km'] = raw_ship_df_dups_dropped['distance_km'].fillna(medians["distance_km"])
    raw_ship_df_dups_dropped['parcel_weight_kg'] = raw_ship_df_dups_dropped['parcel_weight_kg'].fillna(medians["parcel_weight_kg"])
    raw_ship_df_dups_dropped['declared_value_inr'] = raw_ship_df_dups_dropped['declared_value_inr'].fillna(medians["declared_value_inr"])
    print(raw_ship_df_dups_dropped['distance_km'].isna().sum())
    print(raw_ship_df_dups_dropped['parcel_weight_kg'].isna().sum())
    print(raw_ship_df_dups_dropped['declared_value_inr'].isna().sum())

    clean_df = raw_ship_df_dups_dropped

    return (clean_df, medians)



# build_features(clean, hubs) in src/shipment_prep.py.
# It returns a DataFrame with the hub columns attached by a left merge on hub_id (hub_city, region, hub_tier, hub_daily_capacity)
# plus these new columns: promised_hours (from PROMISED_HOURS, integer),
# is_cod (1 when payment_mode is COD, else 0 — Unknown gives 0),
# is_weekend (1 when the dispatch day is Saturday or Sunday),
# dispatch_hour (the hour of the dispatch timestamp as an integer;
# the generated data uses hours 6 to 21), and
# log_declared_value (natural log of declared_value_inr).
# It must not use delivery_hours, so that it also works on parcels that have not been delivered yet.
def build_features(clean, hubs):
  merged_clean_hubs_df = pd.merge(clean_df,hubs_df,on = "hub_id", how = "left")
  merged_clean_hubs_df["promised_hours"]  = merged_clean_hubs_df["service_type"].map(PROMISED_HOURS)
  merged_clean_hubs_df["is_cod"] = (merged_clean_hubs_df["payment_mode"] == "COD").astype(int)
  merged_clean_hubs_df["is_weekend"] = (merged_clean_hubs_df["dispatch_time"].dt.dayofweek >= 5).astype(int)
  merged_clean_hubs_df["dispatch_hour"] = merged_clean_hubs_df["dispatch_time"].dt.hour
  merged_clean_hubs_df["log_declared_value"] = np.log(merged_clean_hubs_df["declared_value_inr"])
  return merged_clean_hubs_df
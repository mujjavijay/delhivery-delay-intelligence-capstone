######################################################part_c_classification.py#####################################################
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
#from src.shipment_prep import FEATURES, NOISE_COLUMNS

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

table = pd.read_csv("data/modelling_table.csv")
#table = pd.read_csv("/content/data/modelling_table.csv")
X, Y = table[FEATURES], table["is_delayed"]

#1. Stratified split. Repeat Part B's two-step split (test_size=0.3 then test_size=0.5, random_state=42 both times) with stratify on the target in both calls. Expected sizes 840 / 180 / 180 with 195 / 42 / 41 delayed parcels.
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42, stratify=Y_temp)
print(len(X_train), len(X_val), len(X_test))

def cm_metrics(y_true, pred):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0)}

#2. Blind baseline and three classifiers, scored on the validation rows with the default 0.5 decision rule. Write outputs/part_c_models.csv with columns model, TN, FP, FN, TP, accuracy, precision, recall, f1 and exactly these rows: blind_all_on_time (predicts "not delayed" for every parcel; write precision as 0); logistic_regression (a Pipeline of StandardScaler then LogisticRegression); decision_tree_depth_<d> (DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42), with d chosen from 2, 3, 4, 5, 6, 8, 10 by highest validation F1, ties → smaller d, and <d> replaced by the chosen value); random_forest_default (RandomForestClassifier(n_estimators=100, random_state=42)). The four confusion-matrix cells in each row must reproduce accuracy, precision, recall and f1.
model_rows = [{"model": "blind_all_on_time", **cm_metrics(Y_val, np.zeros(len(Y_val), dtype=int))}]
logit = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]).fit(X_train, Y_train)
model_rows.append({"model": "logistic_regression", **cm_metrics(Y_val, logit.predict(X_val))})

best_d, best_f1 = None, -1
for d in [2, 3, 4, 5, 6, 8, 10]:
    t = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42).fit(X_train, Y_train)
    f = f1_score(Y_val, t.predict(X_val))
    if f > best_f1 + 1e-12:               # strict: ties keep the smaller depth
        best_d, best_f1 = d, f
tree = DecisionTreeClassifier(criterion="gini", max_depth=best_d, random_state=42).fit(X_train, Y_train)
model_rows.append({"model": f"decision_tree_depth_{best_d}", **cm_metrics(Y_val, tree.predict(X_val))})
rf_default = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, Y_train)
model_rows.append({"model": "random_forest_default", **cm_metrics(Y_val, rf_default.predict(X_val))})
pd.DataFrame(model_rows).round(4).to_csv("outputs/part_c_models.csv", index=False)


#3. Which columns carry signal? Fit a default forest (n_estimators=100, random_state=42) on all 17 columns of the training rows to get feature_importances_; fit SelectKBest(f_classif, k="all") on the training rows to get one F-score per column. Write outputs/part_c_feature_scores.csv (feature, f_score, f_rank, rf_importance, is_noise; f_rank 1 = highest F-score; is_noise is True for the five NOISE_COLUMNS). Then, for k in 5, 7, 10, 12, 17, fit a Pipeline of SelectKBest(f_classif, k=k) followed by RandomForestClassifier(n_estimators=100, random_state=42) on the training rows and write the validation F1 to outputs/part_c_selectk.csv (k, val_f1). From here on k = 7.
sk_all = SelectKBest(f_classif, k="all").fit(X_train, Y_train)
fs = pd.DataFrame({"feature": FEATURES, "f_score": sk_all.scores_,
                   "rf_importance": rf_default.feature_importances_})
fs["f_rank"] = fs["f_score"].rank(ascending=False, method="first").astype(int)
fs["is_noise"] = fs["feature"].isin(NOISE_COLUMNS)
fs = fs[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
fs.round(4).to_csv("outputs/part_c_feature_scores.csv", index=False)
noise_share = fs.loc[fs["is_noise"], "rf_importance"].sum()
print("noise share of RF importance:", round(noise_share, 4))

sel_rows = []
for k in [5, 7, 10, 12, 17]:
    p = Pipeline([("sel", SelectKBest(f_classif, k=k)), ("rf", RandomForestClassifier(n_estimators=100, random_state=42))])
    p.fit(X_train, Y_train)
    sel_rows.append({"k": k, "val_f1": f1_score(Y_val, p.predict(X_val))})
pd.DataFrame(sel_rows).round(4).to_csv("outputs/part_c_selectk.csv", index=False)

#4. Tuning. Fit SelectKBest(f_classif, k=7) on the training rows only and transform the training and validation rows with it. On the transformed training rows run, for RandomForestClassifier(random_state=42), over the grid n_estimators ∈ {50, 100, 150}, max_depth ∈ {4, 6, 8}, min_samples_leaf ∈ {1, 5, 10}: GridSearchCV(..., cv=5) and RandomizedSearchCV(..., n_iter=9, cv=5, random_state=42) (default scoring for both). Write outputs/part_c_tuning.csv with columns search, candidates, cv_folds, total_fits, best_params, best_cv_accuracy, val_f1 and three rows: grid_search, randomized_search, default_forest_same_7_features (the untuned forest n_estimators=100, no depth limit, min_samples_leaf=1 on the same 7 columns; candidates 1, cv_folds 0, total_fits 1, best_params = defaults, best_cv_accuracy left blank). total_fits = candidates × cv_folds for the two searches; val_f1 is each candidate's F1 on the validation rows. The test rows must not be touched in this task.
K = 7
selector = SelectKBest(f_classif, k=K).fit(X_train, Y_train)
Xt_train, Xt_val = selector.transform(X_train), selector.transform(X_val)
grid = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}
gs = GridSearchCV(RandomForestClassifier(random_state=42), grid, cv=5).fit(Xt_train, Y_train)
rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), grid, n_iter=9, cv=5, random_state=42).fit(Xt_train, Y_train)
default7 = RandomForestClassifier(random_state=42).fit(Xt_train, Y_train)

DEFAULT_PARAMS = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}
cands = {
    "grid_search": (gs.best_params_, gs.best_score_, len(gs.cv_results_["params"]), 5, f1_score(Y_val, gs.best_estimator_.predict(Xt_val))),
    "randomized_search": (rs.best_params_, rs.best_score_, len(rs.cv_results_["params"]), 5, f1_score(Y_val, rs.best_estimator_.predict(Xt_val))),
    "default_forest_same_7_features": (DEFAULT_PARAMS, np.nan, 1, 0, f1_score(Y_val, default7.predict(Xt_val))),
}
tune_rows = []
for name, (params, cv_acc, n_cand, folds, vf1) in cands.items():
    tune_rows.append({"search": name, "candidates": n_cand, "cv_folds": folds, "total_fits": (n_cand * folds) if folds else 1,
                      "best_params": json.dumps({k: (None if params[k] is None else int(params[k])) for k in sorted(params)}),
                      "best_cv_accuracy": cv_acc, "val_f1": vf1})
tuning = pd.DataFrame(tune_rows).round(4)
tuning.to_csv("outputs/part_c_tuning.csv", index=False)

#5. Choosing the final configuration. Pick the row of part_c_tuning.csv with the highest val_f1 (ties → default_forest_same_7_features, then grid_search, then randomized_search). The final model is a Pipeline([("selector", SelectKBest(f_classif, k=7)), ("model", RandomForestClassifier(random_state=42, <chosen parameters>))]) fitted on the training rows.
priority = ["default_forest_same_7_features", "grid_search", "randomized_search"]
best_val = tuning["val_f1"].max()
chosen = next(n for n in priority if tuning.loc[tuning["search"] == n, "val_f1"].iloc[0] == best_val)
params = cands[chosen][0]
final_model = Pipeline([("selector", SelectKBest(f_classif, k=K)),
                        ("model", RandomForestClassifier(random_state=42, **params))]).fit(X_train, Y_train)

#6. Alert threshold. Take the final model's validation probabilities (predict_proba, positive class). Build the thresholds as exact decimals — use the literal list [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] or round each value to one decimal (a floating-point range such as np.arange(0.1, 1.0, 0.1) produces values like 0.30000000000000004 that change the result at the boundary). For each threshold t, predict "delayed" when the probability is ≥ t, and write outputs/part_c_thresholds.csv with columns threshold, TN, FP, FN, TP, accuracy, precision, recall, f1. Operations rule: because a missed delayed parcel costs more than an unnecessary call, the alert threshold is the highest t whose validation recall is at least 0.80 (if none reaches 0.80, use 0.1).
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
p_val = final_model.predict_proba(X_val)[:, 1]
th_rows = [{"threshold": t, **cm_metrics(Y_val, (p_val >= t).astype(int))} for t in thresholds]
th = pd.DataFrame(th_rows).round(4)
th.to_csv("outputs/part_c_thresholds.csv", index=False)
ok = th.loc[th["recall"] >= 0.80, "threshold"]
threshold = float(ok.max()) if len(ok) else 0.1

#7. One test evaluation. Score the final model on the test rows at the chosen threshold, exactly once, and write outputs/part_c_final_test.csv (threshold, TN, FP, FN, TP, accuracy, precision, recall, f1). Write outputs/part_c_final_config.csv, one row, columns selector_k, n_estimators, max_depth, min_samples_leaf, threshold, chosen_from (max_depth is unlimited when there is no limit; chosen_from is the name of the winning candidate in part_c_tuning.csv).
p_test = final_model.predict_proba(X_test)[:, 1]
pd.DataFrame([{"threshold": threshold, **cm_metrics(Y_test, (p_test >= threshold).astype(int))}]).round(4) \
    .to_csv("outputs/part_c_final_test.csv", index=False)
pd.DataFrame([{"selector_k": K, "n_estimators": params["n_estimators"],
               "max_depth": "unlimited" if params["max_depth"] is None else params["max_depth"],
               "min_samples_leaf": params["min_samples_leaf"], "threshold": threshold,
               "chosen_from": chosen}]).to_csv("outputs/part_c_final_config.csv", index=False)
print("chosen:", chosen, params, "threshold:", threshold)
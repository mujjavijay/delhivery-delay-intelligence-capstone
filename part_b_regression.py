############################# part_b_regression.py #############################################
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#from src.shipment_prep import FEATURES

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

data = pd.read_csv("/content/data/modelling_table.csv")
X = data[FEATURES]
Y = data["delivery_hours"]

#Split into training, validation and test. Call train_test_split with test_size=0.3, random_state=42 to separate 70% for training from a temporary 30%; call it again on the temporary part with test_size=0.5, random_state=42 to get validation and test. Sizes must be 840 / 180 / 180.
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)
print(len(X_train), len(X_val), len(X_test))
#Scaling without leakage. Fit a StandardScaler on the 840 training rows only, then transform the test rows with that fitted scaler. Write outputs/part_b_checks.csv (metric,value) with train_rows, val_rows, test_rows, train_scaled_max_abs_mean (largest absolute column mean of the scaled training rows) and test_scaled_max_abs_mean (the same for the scaled test rows).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
train_rows = len(X_train)
val_rows = len(X_val)
test_rows = len(X_test)
train_scaled_max_abs_mean = np.max(np.abs(X_train_scaled.mean(axis=0)))
test_scaled_max_abs_mean = np.max(np.abs(X_test_scaled.mean(axis=0)))
print(train_rows, val_rows, test_rows, train_scaled_max_abs_mean, test_scaled_max_abs_mean)
pd.DataFrame({
    "metric": ["train_rows", "val_rows", "test_rows", "train_scaled_max_abs_mean", "test_scaled_max_abs_mean"],
    "value": [train_rows, val_rows, test_rows,
              train_scaled_max_abs_mean, test_scaled_max_abs_mean],
}).round(4).to_csv("outputs/part_b_checks.csv", index=False)

#Baseline. Predict the mean of y_train for every validation row and record its metrics.
baseline_pred = Y_train.mean()
baseline_mae = mean_absolute_error(Y_val, [baseline_pred] * len(Y_val))
baseline_mse = mean_squared_error(Y_val, [baseline_pred] * len(Y_val))
baseline_rmse = np.sqrt(baseline_mse)
baseline_r2 = r2_score(Y_val, [baseline_pred] * len(Y_val))
metrics = pd.DataFrame({
    "model": ["baseline"],
    "split": ["val"],
    "MAE": [baseline_mae],
    "MSE": [baseline_mse],
    "RMSE": [baseline_rmse],
    "R2": [baseline_r2],
})


#Tree-depth diagnostic. For DecisionTreeRegressor(max_depth=d, random_state=42) with d in 1, 2, 3, 4, 6, 8, 12 and no limit, fit on the training rows and write outputs/part_b_tree_depth.csv with columns max_depth (write unlimited for no limit), train_rmse, val_rmse. The depth with the lowest val_rmse (ties → smaller depth) is the decision-tree depth used in task 5.
depth_rows, best_depth, best_val = [], None, np.inf
for d in [1, 2, 3, 4, 6, 8, 12, None]:
    t = DecisionTreeRegressor(max_depth=d, random_state=42).fit(X_train, Y_train)
    tr = np.sqrt(mean_squared_error(Y_train, t.predict(X_train)))
    va = np.sqrt(mean_squared_error(Y_val, t.predict(X_val)))
    depth_rows.append({"max_depth": "unlimited" if d is None else d, "train_rmse": tr, "val_rmse": va})
    if va < best_val - 1e-12:           # strict: ties keep the smaller depth
        best_val, best_depth = va, d
pd.DataFrame(depth_rows).round(4).to_csv("outputs/part_b_tree_depth.csv", index=False)


#Four candidate models, each fitted on the training rows and scored on the validation rows with MAE, MSE, RMSE and R²: LinearRegression inside a Pipeline of ("scaler", StandardScaler()) then ("model", LinearRegression()); the decision tree from task 4; RandomForestRegressor(n_estimators=100, random_state=42); GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42). Write outputs/part_b_metrics.csv with columns model, split, MAE, MSE, RMSE, R2 and rows named mean_baseline, linear_regression, decision_tree, random_forest, gradient_boosting (all split = val), plus the final-test row from task 7. RMSE must equal the square root of MSE (±0.001).
candidates = {

    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}
val_rmse = {}
rows = []
for name, m in candidates.items():
    m.fit(X_train, Y_train)
    pred = m.predict(X_val)
    mae = mean_absolute_error(Y_val, pred)
    mse = mean_squared_error(Y_val, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_val, pred)
    val_rmse[name] = rmse
    rows.append({"model": name, "split": "val", "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2})
    # metrics_candidates = pd.concat([metrics, pd.DataFrame({
    #     "model": [name],
    #     "split": ["val"],
    #     "MAE": [mae],
    #     "MSE": [mse],
    #     "RMSE": [rmse],
    #     "R2": [r2],
    # })], ignore_index=True)
#metrics_candidates.round(4).to_csv("outputs/part_b_metrics.csv", index=False)


#Learning-rate experiment. Fit GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42) for lr in 0.01, 0.1, 1.0 and write outputs/part_b_gb_learning_rate.csv (learning_rate, val_rmse, val_r2).
lr_rows = []
for lr in [0.01, 0.1, 1.0]:
    g = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42).fit(X_train, Y_train)
    va = np.sqrt(mean_squared_error(Y_val, g.predict(X_val)))
    vr = r2_score(Y_val, g.predict(X_val))
    lr_rows.append({"learning_rate": lr, "val_rmse": va, "val_r2": vr})
    pd.DataFrame(lr_rows).round(4).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

#Cross-validated comparison and final model. Rank the four candidates by validation RMSE. For the two best, run 5-fold cross-validation with cross_val_score on the 1,020 training-plus-validation rows, using one KFold(n_splits=5, shuffle=True, random_state=42) object shared by both models (default scoring, which is R²). Write outputs/part_b_cv.csv (model, mean_r2, std_r2). The final model is the candidate with the lowest validation RMSE; score it once on the test rows (already fitted on the training rows) and add the row model = <its name>, split = test to part_b_metrics.csv.
ranking = sorted(val_rmse, key=val_rmse.get)
print(val_rmse)

top2 = ranking[:2]
X_tv, Y_tv = pd.concat([X_train, X_val]), pd.concat([Y_train, Y_val])
cv_rows = []
for name in top2:
     cv = cross_val_score(candidates[name], X_tv, Y_tv, cv=KFold(n_splits=5, shuffle=True, random_state=42))
     cv_rows.append({"model": name, "mean_r2": cv.mean(), "std_r2": cv.std()})
pd.DataFrame(cv_rows).round(4).to_csv("outputs/part_b_cv.csv", index=False)

final_name = ranking[0]

mse_final = mean_squared_error(Y_test, candidates[final_name].predict(X_test))
r2_final = r2_score(Y_test, candidates[final_name].predict(X_test))
mae_final = mean_absolute_error(Y_test, candidates[final_name].predict(X_test))
rsme_final = np.sqrt(mse)


rows.append({"model": final_name, "split": "test","MAE": mae_final,"MSE": mse_final,"RMSE": rsme_final,"R2": r2_final })
pd.DataFrame(rows)[["model", "split", "MAE", "MSE", "RMSE", "R2"]].round(4).to_csv("outputs/part_b_metrics.csv", index=False)
print("best depth:", best_depth, "| ranking:", ranking, "| final:", final_name)
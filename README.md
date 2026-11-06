# delhivery-delay-intelligence-capstone

# Data findings
Total raw data had 1224 rows. exact duplicates were 24. After removing duplicated the remaining records are 1200. 
Total Delayed are 278 which is 23.16% of total orders

(i) which service type is delayed most often and at what rate;
Answer: Express service type is  most often delayed. 119 out of 243 at a rate of 0.4897

(ii) how delay rate changes from Tier 1 to Tier 3; 
Answer: Tier 1 has 0.0400 delay rate, Tier 2 has 0.2150 and Tier 3 has 0.4400 delay rate.

(iii) how it changes across distance bands; 
0-200	has	0.0219
201-400	has	0.1368
401-600	has	0.4364
601+	has	0.8523

(iv) the three columns most strongly correlated with delivery_hours — rank the columns of part_a_correlations.csv by absolute correlation, exclude delivery_hours itself, and quote each signed value to four decimals; and
Answer: promised_hours at 0.8372 then distance_km at 0.4013 then n_stops at 0.3247

(v) one sentence on why declared_value_inr was log-transformed (compare its mean and median before and after, using the filled column).
since the declared_value_inr has huge values in them it will skew the results in more favor towards it.
('declared_value_mean_inr', np.float64(1533.0541666666666)),
 ('declared_value_median_inr', 1121.5),
 ('log_declared_value_mean', np.float64(7.027692684489747)),
 ('log_declared_value_median', 7.022422353960263)]
 After log the mean and median are close.

# Model results (regression)
Split: 840 training, 180 validation, 180 test rows. The scaler was fitted on training rows only: the scaled training columns have a largest absolute mean of about 4e-16 (shown as 0.0 after rounding to 4 places), while the scaled test rows give 0.2187 (part_b_checks.csv), which proves the scaler did not see the test rows.

Baseline. Predicting the training mean for every validation parcel gives R² of -0.0212 (RMSE 18.8552 hours). An R² of about zero means it explains none of the variation in delivery time; every real model must beat this.
Winner on validation. Linear regression, with validation RMSE 4.8912 hours (R² 0.9313). Gradient boosting was almost identical (RMSE 4.9165, R² 0.9306), then random forest (5.5095, R² 0.9128) and the decision tree (6.2796, R² 0.8867).
Bias and variance (part_b_tree_depth.csv). Depth 1 is high bias: train RMSE 11.6334 and validation RMSE 12.1324 are both large and close (gap 0.4990), and depth 2 is similar (10.4114 versus 9.3687). Depths 12 and unlimited are high variance: unlimited depth fits the training set perfectly (train RMSE 0.0) but validation RMSE is 7.8368, a gap of 7.8368 hours; depth 12 has a gap of 5.9927. Validation error is lowest at depth 6 (RMSE 6.2796, gap 1.5682), which is the depth used for the decision tree. Depth 6 is 5.8528 hours better than depth 1.
Learning rate (part_b_gb_learning_rate.csv). 0.1 is best (validation R² 0.9306). With 0.01 (R² 0.7127) each of the 100 trees corrects only 1% of the error, so the ensemble is too slow and underfits. With 1.0 (R² 0.8440) each tree's correction is applied in full, so the ensemble overshoots and chases noise.
Cross-validation (part_b_cv.csv, same KFold(5, shuffle=True, random_state=42) for both). Gradient boosting: mean R² 0.9353 (std 0.0045). Linear regression: mean R² 0.9261 (std 0.0048). The cross-validated ranking disagrees with the validation ranking, because on validation linear regression was better by only 0.0253 hours of RMSE. The five-fold estimate uses all 1,020 rows and is far less affected by which 180 rows happened to be in validation, and the data contains interactions a straight line cannot represent (the sale window adds 12 extra hours to Express parcels and rain adds 8 extra hours on lanes over 400 km). I would ship gradient boosting.
Test result. By the pre-stated rule (lowest validation RMSE), the final model scored once on the test rows is linear regression: RMSE 4.5235 hours, MAE 3.5671 hours, R² 0.9493. I did not re-score gradient boosting on the test rows, so the test set stays untouched by the shipping decision.

## Model results (classification)

Stratified split gives 840 / 180 / 180 rows with 195 / 42 / 41 delayed parcels.

- **Blind baseline.** Predicting "not delayed" for every parcel has validation accuracy 0.7667 (TN 138, FN 42) but recall 0: it catches none of the 42 delayed parcels, so it is useless for the operations goal even though the accuracy looks decent. The trained models have accuracy 0.8833 to 0.9056 (`part_c_models.csv`) with F1 of 0.8046 (logistic regression), 0.7470 (decision tree, depth 6) and 0.7848 (default forest).
- **Noise columns.** The five bookkeeping columns carry 0.2244 of the default forest's total importance, even though they carry no signal; the forest spreads splits over them. By F-score all five fall outside the top 7 (ranks 8, 12, 14, 16 and 17) and each has an F-score below 10 (highest 4.5847), while the lowest F-score among the top seven is 36.7281 (`part_c_feature_scores.csv`). The top seven are `distance_km`, `hub_tier`, `hub_daily_capacity`, `n_stops`, `promised_hours`, `rain_flag` and `sale_event_flag`.
- **Selecting k (`part_c_selectk.csv`).** Validation F1 is 0.6988 for k=5, 0.8333 for k=7, 0.7857 for k=10, 0.8049 for k=12 and 0.7848 for all 17. k=7 is best: k=5 drops real signal columns, and larger k adds noise columns back.
- **Tuning (`part_c_tuning.csv`).** Grid search (27 candidates, 135 fits, best cross-validated accuracy 0.8952) reached validation F1 0.7750; randomized search (9 candidates, 45 fits, 0.8905) reached 0.7848; the untuned forest on the same 7 columns reached 0.8333. So tuning did not beat the untuned forest on validation. The cross-validated accuracies differ by only 0.0047, and on a 180-row validation set with 42 delayed parcels one or two parcels move F1 by more than that, so these differences are within noise; the lesson is that tuning on a tiny validation set can chase luck, and that a default forest on well-chosen features is already close to the best this data supports. The final configuration is the default forest (`n_estimators` 100, no depth limit, `min_samples_leaf` 1) on k=7 features.
- **Alert threshold (`part_c_thresholds.csv`).** The operations rule needs validation recall of at least 0.80. Threshold 0.6 gives recall 0.7619 (too low); threshold 0.5 gives recall 0.8333 with precision 0.7955 (TP 35, FP 9, FN 7); lowering to 0.4 raises recall to 0.8810 but precision falls to 0.7400, and at 0.1 precision is only 0.4824. The highest threshold meeting the rule is **0.5**.
- **Test result (`part_c_final_test.csv`, scored once).** At threshold 0.5: TN 125, FP 14, FN 10, TP 31; accuracy 0.8667, precision 0.6889, recall 0.7561, F1 0.7209. Test recall (0.7561) is lower than validation recall (0.8333) and below the 0.80 target. With only 41 delayed parcels in the test set, 31 caught instead of the 34 that would give 0.83 is a small absolute difference, so it is partly sampling noise, but I do not claim the 0.80 recall target is met on unseen data.

## Operations recommendation

**Escalate segment.** K-Means on the hub profiles (`hub_segments.csv`, `segment_profile.csv`) puts nine hubs in Escalate, with a segment delay rate of 0.4600, mean delivery 64.6504 hours, mean distance 470.4757 km and 84.3444 stops: Nagpur (0.46), Kochi (0.36), Patna (0.46), Ranchi (0.38), Guwahati (0.62), Raipur (0.48), Bhubaneswar (0.40), Siliguri (0.44) and Agartala (0.54). Watch has 8 hubs (delay rate 0.1550) and Standard 7 hubs (0.0257).

**Tier labels that mislead** (`tier_vs_segment_mismatch.csv`):
- Kolkata (Tier 1) behaves like a Watch hub, so the Tier 1 label overstates its reliability.
- Nagpur (Tier 2) and Kochi (Tier 2) behave like Escalate hubs, so the Tier 2 label understates their risk.
- Dehradun (Tier 3) behaves like a Watch hub, so the Tier 3 label is too pessimistic.
Operations should assign playbooks from the data-driven segment and treat the tier label as a static attribute only.

**Alert threshold.** Flag a parcel when its delay probability is at least 0.5. On the test rows, about 7 of every 10 flagged parcels are really delayed (precision 0.6889) and about 8 of every 10 delayed parcels are caught (recall 0.7561, that is 7.6 of 10); the validation recall was 0.8333, so plan for roughly 75 to 83 percent.

**New parcels.** Of the 48 new parcels in `new_shipment_actions.csv`, 8 are flagged; 6 of those sit in Escalate hubs and need a hub-manager escalation, and the other 2 get a proactive customer call. The remaining 40 need no action.

**Why K = 3.** The across-to-within ratio (`part_d_separation.csv`) is 2.894 for K=2, 5.3928 for K=3, 6.3047 for K=4 and 6.8866 for K=5. The largest jump comes from 2 to 3 (+2.4988); after that it adds only +0.9119 and +0.5819. K=4 has cluster sizes 8-7-5-4 and K=5 has 8-7-4-3-2, so each contains a cluster of fewer than 6 hubs, too small to carry its own playbook, and the business runs exactly three playbooks. K=3 gives sizes 9-8-7 with clearly separated delay rates.

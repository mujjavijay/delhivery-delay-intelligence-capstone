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

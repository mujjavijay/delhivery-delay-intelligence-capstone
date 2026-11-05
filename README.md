# delhivery-delay-intelligence-capstone

# Data findings
Total raw data had 1224 rows. exact duplicates were 24. After removing duplicated the remaining records are 1200. 
Total Delayed are 278 which is 23.16% of total orders

(i) which service type is delayed most often and at what rate;/n
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

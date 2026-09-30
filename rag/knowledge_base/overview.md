# Overview

This dashboard models seasonal groundwater depth to water level (DTWL, in mbgl)
for rajasthan district using a supervised machine-learning workflow.

Four seasons are analysed independently: Pre-Monsoon, Monsoon, Post-Monsoon,
and Non-Monsoon.

The pipeline stages are:
1. Workbook loading and validation
2. Derived feature engineering (Temp_Range, Avg_SM, SM_Gradient, Veg_Moisture,
   GW_Stress, Water_Balance, Veg_Health, Season_Sin, Season_Cos)
3. Lag, rolling, and trend feature creation
4. Training-only feature selection (mRMR or mutual information)
5. Multi-model training with temporal train/validation/test split
6. Bias-corrected held-out evaluation
7. Diagnostics, sensitivity, forecast, wavelet analysis
8. A second, separate deep-learning / physics-informed pass (PINN, reservoir
   computing, SARIMA, and LSTM), shown on its own tab

On seasons with too little data for the split after step 3, the pipeline
retries by imputing incomplete lag/rolling/trend values within each village
instead of dropping those rows — see the troubleshooting doc's "small
dataset handling" entry.

The target variable is "DTWL (mbgl)" — depth to water level in metres below
ground level. Lower DTWL means a shallower, healthier water table.
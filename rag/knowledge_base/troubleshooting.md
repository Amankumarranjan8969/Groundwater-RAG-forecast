# Troubleshooting

### "Not enough data" message
Each season needs enough years to build lag features and support a temporal
train/validation/test split — at least eight distinct years overall, and
after lag creation, at least 8 training rows, 2 validation rows, and 2 test
rows for that season. If you hit this message, either upload more years of
data or move the training/validation split earlier in the sidebar. On
genuinely small datasets the pipeline also tries to rescue rows on its own
first — see "Small dataset handling" below.

### Small dataset handling
When the strict pass leaves too few rows, the pipeline automatically
retries once: instead of dropping every row with an incomplete lag,
rolling, or trend value, it imputes just those engineered columns
(forward-filled within each village, then filled with the column median) so
there's enough data to train on. Raw measurements and the target itself are
never touched — only the derived lag/rolling/trend columns. A notice
appears on the Data quality tab whenever this kicks in for a season, and it
only ever kicks in when the strict, notebook-faithful pass wasn't enough on
its own.

### Only some models ran
CatBoost, XGBoost, LightGBM, Cubist, M5, and the LSTM in the deep-learning
tab are all optional — if a package isn't installed, that one model is
skipped silently and everything else keeps running. Check the "Model
availability" expander on the relevant tab to see what's active. To add
one, install it, e.g. `pip install catboost xgboost lightgbm cubist m5py
torch`.

### Wavelet tab says "at least 16 observations"
The continuous wavelet transform needs at least 16 aggregated quarterly
points, which works out to roughly 4 years of data across all four
seasons. For the significance testing to mean much, aim for 20+ years.

### Taylor diagram is empty
The Taylor grid needs held-out predictions for each season — if a season
was excluded from the run, its panel will say so. Re-run with that season
selected in the sidebar. The same applies to the smaller Taylor diagram on
the deep-learning tab, scoped to PINN / ESN / SARIMA / LSTM.

### The chat says "I don't know"
That's intentional — I only answer from the methodology docs and the
current run's live numbers, so if nothing relevant turns up I'll say so
instead of guessing. Try rephrasing with a keyword like "R²", "Taylor",
"wavelet", "PINN", or a season name.

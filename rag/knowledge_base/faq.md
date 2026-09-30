# Conversation guide

### Comparing two models
Give me two model names and a season and I'll compare them on R², RMSE, MAE,
KGE, and NSE using the current run's numbers. Works across either model
family — the six headline models, or PINN / reservoir computing / SARIMA /
LSTM in the deep-learning tab.

### Explaining things simply
DTWL is just how deep the water table sits, in metres below ground —
smaller numbers mean a shallower, healthier water table. Each season gets
its own model because the drivers behind groundwater change differ by
season: rainfall lag dominates in Monsoon, while ET and soil-moisture
recession dominate in Non-Monsoon. R² is how much of the variation in DTWL
the model explains (1.0 is perfect); RMSE is the typical error, in metres.
Everything else on the dashboard is really just a way of checking those two
numbers from a different angle.

### Summarising the current run
Ask me to "summarise the run" and I'll pull straight from the live session:
how big the dataset is and which seasons are present, the best model per
season, its bias-corrected test R² and RMSE, and the features that were
selected for it.

### Where to start looking
Start with the overview bar chart at the top for a quick read on which
model won each season, then the model performance table for R²/RMSE/MAE,
the diagnostics tab for the observed-vs-predicted scatter, the Taylor
diagram for variance and correlation in one glance, and the wavelet tab if
you're curious about periodicities in the aggregated series.

### Explaining a term simply
Just name the term — "explain R² simply" works fine — and I'll drop the
jargon and use a plain analogy instead of the formal definition.

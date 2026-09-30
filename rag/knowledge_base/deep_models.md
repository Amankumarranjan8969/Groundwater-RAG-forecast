# Deep learning & physics-informed models

A second, separate model family lives in its own "12 · Deep learning &
physics-informed" tab. It trains on the same temporal split and the same
selected features as the six headline models, and is scored the same way
(bias-corrected test R², RMSE, MAE, KGE, NSE) — but it's kept in its own
results, predictions, and Taylor diagram so it never changes the six-model
views elsewhere in the dashboard. This whole family can be switched on or
off from the sidebar before running the analysis.

### PINN
PINN stands for Physics-Informed NN. It's a small neural network whose
training loss adds a soft water-balance penalty — built from Rainfall minus
ET — on top of the usual prediction error, nudging predictions toward
physically sensible recharge and discharge behaviour rather than just
fitting the numbers. When those columns aren't part of a season's selected
features, the physics term switches off automatically and it behaves like a
plain regularised neural network.

### Reservoir computing
Reservoir Computing (ESN) is an echo-state network: a large, fixed, random
recurrent reservoir whose only trained part is the final linear read-out.
It rides each row's own lag history — lag4 through lag1 through the present
value — through the reservoir before reading out a prediction.

### SARIMA
SARIMA (SARIMAX) is a classical autoregressive model with exogenous
drivers, included as a statistical time-series baseline next to the
machine-learning models above it.

### LSTM
LSTM (Deep Learning) is a single-layer LSTM over the same lag history as
the reservoir model, trained end-to-end with PyTorch. It only appears when
the optional `torch` package is installed — otherwise it shows as "Not
installed" in the model availability table, the same way CatBoost or
XGBoost would.

### Where to find it
Its metrics table, diagnostic scatter plot, and its own Taylor diagram all
live on tab 12, right after the wavelet analysis.

# Models

The six headline models are:
- Cubist (rule-based regression)
- Random Forest
- XGBoost
- CatBoost
- AdaBoost + CART
- Extra Trees

The dashboard also trains Linear Regression, Ridge, Bayesian Ridge, SVR (RBF),
and a Gaussian Process with an RBF kernel.

Optional packages (CatBoost, XGBoost, LightGBM, Cubist, m5py) are included only
when installed. The scikit-learn models always run.

Random Forest and Extra Trees typically perform best on this dataset. Cubist
often wins on Monsoon because its rule-based partitions capture the sharp
recharge peak.

A separate model family — PINN, reservoir computing (ESN), SARIMA, and LSTM
— trains alongside these six but is scored and shown on its own "12 · Deep
learning" tab rather than mixed into this comparison. See the deep-learning
KB entry for what each of those does.
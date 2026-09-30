# Groundwater ML Dashboard

An upload-driven Streamlit dashboard built from the seasonal Ajmer groundwater notebook. It follows the notebook workflow for every selected season: data validation, engineered features, lag/rolling/trend variables, training-only feature selection, model comparison, diagnostics, feature sensitivity, scenario forecasting, and Excel export. It also includes cross-season metric heatmaps, residual distributions, journal-style Taylor diagrams, rainfall/DTWL timelines, ACF/PACF and ADF diagnostics, rainfall cross-correlation, interactive 3D groundwater surfaces, and a second, separate deep-learning / physics-informed model family (PINN, reservoir computing, SARIMA, and LSTM) on its own tab. A built-in chat assistant, grounded in the methodology docs and the current run's live numbers, can answer questions about any of it.

## Run it

Open PowerShell in this folder and run:

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

If `py` is not available, use the Python executable installed on your computer in its place. Streamlit opens the dashboard in your browser (usually at `http://localhost:8501`).

## Upload format

Upload an `.xlsx` file containing one or more recognised sheets:

- `Monsoon`
- `Pre_Monsoon`
- `Post_Monsoon`
- `Non_Monsoon`

The dashboard requires these fields in each sheet: `VILLAGE`, `YEAR`, `DTWL (mbgl)`, `Tmax`, `Tmin`, `SM_10cm`, `SM_40cm`, `SM_100cm`, `NDVI`, `NDMI`, `NDBI`, `ET`, and `Rainfall`.

Other numeric variables in your file, such as terrain and remote-sensing fields, are kept as potential predictors. The original metadata columns (`STATE_UT`, `DISTRICT`, `BLOCK`, `LATITUDE`, and `LONGITUDE`) are excluded in the same way as the notebook.

## The two model families

**Six headline models** (tabs 3–11): Cubist, Random Forest, XGBoost, CatBoost, AdaBoost+CART, and Extra Trees, plus scikit-learn baselines. These drive the main Taylor diagrams, cross-season views, sensitivity analysis, and scenario forecasts.

**Deep learning & physics-informed models** (tab 12, toggle in the sidebar): a second, independent family trained on the same split and features.
- **PINN** — a small neural network with a soft water-balance physics prior in its loss.
- **Reservoir Computing (ESN)** — an echo-state network riding each row's own lag history through a fixed random reservoir, with only the read-out layer trained.
- **SARIMA (SARIMAX)** — a classical autoregressive baseline with exogenous drivers.
- **LSTM** — a PyTorch LSTM over the same lag history. Needs the optional `torch` package; skipped gracefully (like CatBoost/XGBoost) if it isn't installed.

This family never touches the six-model Taylor diagrams or cross-season views — it has its own metrics, predictions, and Taylor diagram on tab 12.

## Notes

- Default time boundaries match the notebook: train through 2012, validate through 2014, and test on later years. Change these in the sidebar for another time range.
- CatBoost, XGBoost, LightGBM, mRMR, Cubist, and M5 are treated safely. The first four are installed by `requirements.txt`; Cubist and M5 are added only if a compatible local package is already available. Core scikit-learn models always remain available. The same applies to `torch` for the LSTM model above.
- Forecasts are clearly labelled as scenarios: future rainfall, soil moisture, vegetation, and other non-target variables are held at the latest measured state, while DTWL-history features update recursively.
- **Small datasets**: if a season doesn't have enough complete rows for the train/validation/test split, the pipeline automatically retries by imputing incomplete lag/rolling/trend values within each village (forward-fill, then median) instead of dropping those rows — raw measurements and the target itself are never touched. A notice appears on the Data quality tab whenever this happens, and it only kicks in when the strict, notebook-faithful pass wasn't enough on its own.
- The chat assistant (bottom of the page) answers from the methodology docs in `rag/knowledge_base/` and, once you've run an analysis, the live numbers from your session — including the deep-learning tab and whether the low-data rescue above kicked in for a season. It optionally calls Gemini or Claude if `GEMINI_API_KEY` or `ANTHROPIC_API_KEY` is set (see `rag/llm_provider.py`); otherwise it composes answers locally.


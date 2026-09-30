# Metrics

### R²
The proportion of variance in DTWL explained by the model. 1.0 is perfect,
0.0 means the model is no better than predicting the mean, and negative
means the model is doing worse than that.

### RMSE
Root mean squared error, in mbgl. Lower is better. Penalises large errors
more heavily than small ones because they're squared before averaging.

### MAE
Mean absolute error, in mbgl. Lower is better, and unlike RMSE it weighs
every error equally regardless of size.

### Pearson r
The linear correlation between observed and predicted DTWL — how closely
the two move together, independent of any constant offset or scale.

### KGE
Kling–Gupta efficiency. Combines correlation, bias ratio, and variability
ratio into a single score; 1.0 is perfect. It's a stricter, more balanced
check than R² because it penalises a model that's well correlated but
biased or under/over-dispersed.

### NSE
Nash–Sutcliffe efficiency. 1.0 is perfect, 0.0 means the model is no better
than always predicting the mean observed value — conceptually close to R²
but computed slightly differently.

### Bias
Mean(prediction) minus mean(observation). Positive means the model
over-predicts DTWL, i.e. it under-estimates how much water is actually
available, since a larger DTWL number means a deeper, less healthy water
table.

### Bias-corrected test metrics
The dashboard's test-set predictions shifted by the training-mean residual
before scoring — this removes any constant offset the model picked up
during training so the test-set score reflects the model's shape-fitting
ability, not a simple systematic bias.

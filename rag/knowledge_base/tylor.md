# Taylor diagram

A Taylor diagram summarises three statistics in one polar plot:
- Angular position: arccos(correlation) — closer to the x-axis is higher r.
- Radial distance: σ_model / σ_observed — 1.0 means the model reproduces the
  observed variability.
- Dashed arcs: centred RMSD around the observed reference point.

The observed reference is the star at (1.0, 0). A model that sits exactly on the
star has r = 1, σ ratio = 1, and centred RMSD = 0.

Reading tips:
- A model far from the reference with a large σ ratio over-disperses predictions.
- A model near the reference but at high angle correlates poorly.
- In this dashboard, each season gets its own panel in a 2×2 grid.
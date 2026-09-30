# Wavelet analysis

The wavelet tab runs a continuous wavelet transform (CWT) on the aggregated
seasonal DTWL series (median across villages, quarterly timestamped).

- Mother wavelet: Morlet with ω₀ = 6 by default. Alternatives: Paul, DOG.
- Y-axis is log₂(period in years).
- Coloured contours show normalised power (log₂ scale).
- The hatched region is the cone of influence — periods there are edge-affected
  and should be interpreted with caution.
- Black contour lines mark the 95 % significance level against a red-noise (AR1)
  background, following Torrence & Compo (1998).
- The global spectrum on the right shows time-averaged power vs period.

Interpretation:
- Peaks near 2–7 years often correspond to ENSO.
- Peaks near 3–7 years can also reflect the Indian Ocean Dipole (IOD).
- Peaks near 11 years may reflect the solar cycle.
- Peaks beyond ~10 years in a 30-year record are usually inside the COI and
  should not be over-interpreted.
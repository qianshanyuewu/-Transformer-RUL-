# Chapter 2 Rebuild Notes

## Scope

- Dataset: XJTU-SY bearing accelerated degradation dataset
- Data source used now: vibration only
- Temperature: not used
- Vibration channels: horizontal only in the frozen thesis protocol

## Current method

1. Read raw vibration files and use the horizontal channel as the unified analysis target.
2. Apply `db4` wavelet denoising with level `1` and `soft` thresholding.
3. Extract the frozen set of `13` time-domain candidate features.
4. Smooth each feature sequence with Savitzky-Golay filtering.
5. Use `monotonicity + time-correlation` only as an auxiliary ranking score on all bearings.
6. Freeze the modeling feature set to `12` cumulative time-domain features by removing `mean_abs`.
7. Build cumulative transformed features.
8. Construct fused health indicator from the fixed `12` features.
9. Detect FPT with the `3σ` rule.

## Current defaults

- Wavelet: `db4`
- Wavelet level: `1`
- Wavelet mode: `soft`
- Score weights: `0.5 / 0.5`
- Candidate feature count: `13`
- Fixed modeling feature count: `12`
- FPT baseline window: `10`
- FPT sigma: `3.0`
- FPT consecutive points: `3`

## Current status

- The Chapter 2 report has been regenerated under the frozen protocol.
- The current Chapter 2 figures and document outputs are already based on the unified `horizontal + 13 candidates + fixed 12` pipeline.
- FPT is kept as a Chapter 2 degradation-stage result, but it is no longer the default Chapter 3 window start.

## Current interpretation

- The Chapter 2 path is now based on real vibration data only, without any unsupported temperature narrative.
- The candidate pool is intentionally limited to `13` time-domain features to match the Chapter 3 modeling path.
- Trend scoring is retained, but only as a pre-ranking step.
- The final modeling feature set is fixed rather than dynamically selected per condition.

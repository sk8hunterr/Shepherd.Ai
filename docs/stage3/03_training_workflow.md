# Stage 3 Training Workflow

The Stage 3 training workflow is:

1. Download or load real Kepler light curves.
2. Download or load the official KOI cumulative catalog.
3. Normalize and window the light curves.
4. Match targets to the KOI catalog.
5. Compute expected transit times from period and transit epoch values.
6. Label windows using real transit-event timing.
7. Skip ambiguous edge windows.
8. Save the prepared dataset to `data/prepared/`.
9. Train multiple simple models on the same dataset split.
10. Compare the model metrics.
11. Save the best model to disk.
12. Save the latest run outputs and append an experiment log entry.

The key idea in Stage 3 is that dataset preparation and model training are now separate concepts, even though they still run in one command. The dataset itself is now reusable, which makes scaling much easier later.

# Stage 2 Training Workflow

The Stage 2 training workflow was:

1. Load a larger set of real Kepler light curves.
2. Clean and normalize the flux values.
3. Split the light curves into many more windows than Stage 1.
4. Continue using synthetic transit injection for labels.
5. Convert each window into simple transit-shape features.
6. Train the baseline model.
7. Evaluate the model.
8. Save the best model to disk.
9. Use that saved model inside the app for future testing.

The key difference from Stage 1 was scale. Stage 2 used more real Kepler data, created more training examples, and saved the learned model so it could be reused later.

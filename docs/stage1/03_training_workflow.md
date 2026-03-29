# Stage 1 Training Workflow

The Stage 1 training workflow was:

1. Load a small sample of Kepler light curves.
2. Remove invalid values from the data.
3. Normalize the brightness values so different stars could be compared more fairly.
4. Split each long light curve into many smaller fixed-length windows.
5. Create labels by injecting fake transit-like dips into some windows.
6. Convert the windows into simple numerical features.
7. Split the dataset into training and test sets.
8. Train a logistic regression model.
9. Evaluate the model using accuracy, precision, recall, F1 score, and confusion matrix.
10. Save plots and a run summary.

The key idea in Stage 1 was that the model was learning from synthetic labels, not real planet-candidate labels. That made the problem simpler and helped establish a stable first baseline.

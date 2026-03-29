# Stage 1 Plain-English Walkthrough

Stage 1 was the first working version of the project. The goal was to build a simple machine learning system that could look at star brightness data and learn the difference between transit-like signals and non-transit signals.

At this stage, the project used Kepler light-curve data as the starting point. A light curve is just a record of how a star's brightness changes over time. The code loaded a small number of Kepler targets, cleaned the data, normalized the brightness values, and split each long light curve into many smaller fixed-size windows.

Because real labels are hard to build at the beginning of a research project, Stage 1 used synthetic labeling. That means some windows were left unchanged and labeled as non-transit examples, while other windows had a simple fake transit dip added and were labeled as transit examples. This gave the model a clean training problem without needing a full astronomy catalog system yet.

The model used in Stage 1 was logistic regression. Instead of feeding the entire raw signal directly into the model, the code extracted simple features that describe the shape of each window. These features included things like the minimum flux value, the spread of the flux values, and how long low points continued in a row. The project then trained the model, measured accuracy, precision, recall, F1 score, and confusion matrix, and saved plots and summaries.

By the end of Stage 1, the project had become a full end-to-end prototype. It could load data, preprocess it, create labels, train a baseline model, evaluate results, and save outputs. That gave the project a stable starting point for all later improvements.

# Stage 3 Plain-English Walkthrough

Stage 3 is the point where the project became much more scientifically realistic. The biggest change was that the labels stopped coming from fake injected transit dips and started coming from the official Kepler KOI cumulative catalog.

In Stage 3, the project still downloaded and processed real Kepler light curves, but now it also loaded the real Kepler candidate catalog. The code matched the Kepler target names in the light curves to the host stars in the catalog, used the catalog's orbital period and transit epoch values to estimate when transits should occur, and labeled windows based on whether they overlapped those real candidate transit events.

Stage 3 also became more careful about labels. Windows that only partially overlapped a transit in an ambiguous way were skipped instead of being forced into positive or negative classes. That improved the quality of the training set.

At this stage, the project compared two models on the same Stage 3 dataset: logistic regression and random forest. Random forest performed better, so it became the best saved Stage 3 model. Stage 3 also added a reusable prepared dataset file and a simpler output organization with a latest run folder, an archive folder, and an experiment log.

By the end of Stage 3, the project had become a real-label baseline training system, not just a synthetic demonstration.

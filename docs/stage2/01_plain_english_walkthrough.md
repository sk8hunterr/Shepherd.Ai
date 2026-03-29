# Stage 2 Plain-English Walkthrough

Stage 2 took the Stage 1 prototype and made it larger and more realistic. The main goal was to train on more real Kepler light-curve data instead of just a very small sample.

At this stage, the project expanded the number of Kepler targets, allowed more light curves per target, and created more windows per light curve. The preprocessing and feature extraction logic stayed mostly the same, because the focus of Stage 2 was not to redesign the pipeline. The focus was to scale it.

The labels in Stage 2 were still synthetic, which means fake transit dips were still being inserted into some windows. However, the underlying light curves were now much more strongly based on real Kepler data. That meant the model was learning transit-like patterns against real telescope noise and star variability, which was much more useful than the first tiny proof of concept.

Stage 2 also added saved model support. Instead of the trained model disappearing when the program ended, the project now wrote the trained model to disk. That made the project easier to reuse and was an important step toward a real application. Stage 2 also added the first local app prototype, which later became Shepherd.Ai.

By the end of Stage 2, the project had a stronger real-data baseline, a saved model workflow, and a more realistic training scale than Stage 1.

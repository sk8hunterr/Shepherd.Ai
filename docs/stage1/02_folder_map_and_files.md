# Stage 1 Folder Map And Important Files

At Stage 1, the most important files were:

- `main.py`
  - The entry point that started the pipeline.
- `src/exoplanet_research/config.py`
  - Stored project settings such as targets, window length, and output locations.
- `src/exoplanet_research/data_loader.py`
  - Loaded Kepler light curves or used a fallback synthetic source if download failed.
- `src/exoplanet_research/preprocessing.py`
  - Normalized flux values and split the light curves into windows.
- `src/exoplanet_research/labeling.py`
  - Created synthetic transit and non-transit labels.
- `src/exoplanet_research/modeling.py`
  - Trained the logistic regression baseline.
- `src/exoplanet_research/evaluation.py`
  - Computed metrics and confusion matrix values.
- `src/exoplanet_research/visualization.py`
  - Saved example plots and result plots.
- `src/exoplanet_research/pipeline.py`
  - Connected all the steps together in order.
- `requirements.txt`
  - Listed the Python packages needed to run the project.
- `README.md`
  - Explained the project and how to run it.

The most important folders at Stage 1 were:

- `data/`
  - Held downloaded or cached light-curve data.
- `outputs/`
  - Held plots, summaries, and metrics from the run.
- `src/`
  - Held the project code.

Stage 1 was mainly about building the core pipeline files that made the project run from beginning to end.

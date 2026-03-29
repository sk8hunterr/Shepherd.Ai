# Stage 3 Folder Map And Important Files

Stage 3 kept the core files from Stage 1 and Stage 2, and added several new important files:

- `src/exoplanet_research/catalog_loader.py`
  - Downloads and caches the official Kepler KOI cumulative catalog.
- `src/exoplanet_research/real_labeling.py`
  - Uses real KOI timing information to label windows.
- `src/exoplanet_research/dataset_builder.py`
  - Builds and saves reusable prepared datasets.
- `src/exoplanet_research/output_manager.py`
  - Manages the simpler output structure.

The most important Stage 3 folders are:

- `data/prepared/`
  - Stores reusable prepared datasets and metadata.
- `models/`
  - Stores the current best trained model, including the Stage 3 model.
- `outputs/latest_run/`
  - Stores the newest run results.
- `outputs/archive/`
  - Stores older archived run results.
- `outputs/experiment_log.csv`
  - Stores a simple experiment history.

Stage 3 is the stage where the project files became more like a real research workflow and less like one single monolithic script.

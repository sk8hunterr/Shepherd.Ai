# Stage 2 Folder Map And Important Files

Stage 2 kept all the important Stage 1 files and added a few more important pieces:

- `app.py`
  - The first local GUI for uploading light curves and screening them with the model.
- `src/exoplanet_research/model_io.py`
  - Saved and loaded trained models from disk.
- `src/exoplanet_research/training_service.py`
  - Prepared a model for the app and reused saved models when available.
- `src/exoplanet_research/inference.py`
  - Loaded uploaded CSV files and ran the model on them.

The most important folders at Stage 2 were:

- `models/`
  - Stored saved trained model files.
- `data/cache/`
  - Stored more downloaded Kepler light curves.
- `outputs/`
  - Held more training summaries and figures.
- `data/samples/`
  - Held sample CSV files for app testing.

Stage 2 was the stage where the project started to look like both a training system and a user-facing screening tool.

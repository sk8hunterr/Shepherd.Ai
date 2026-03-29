# Exoplanet Transit Baseline Project

This project is a beginner-friendly summer research prototype for detecting likely exoplanet transit signals in Kepler light curve data.

The current project has two main parts:

- a baseline machine learning pipeline for training and evaluation
- a first local GUI prototype for screening uploaded light curves

## What The Project Does

The first version is intentionally simple:

- It tries to load a sample of real Kepler light curves.
- It cleans and normalizes the flux values.
- It cuts each light curve into fixed-size windows.
- It creates labels by injecting simple synthetic transit dips into some windows.
- It extracts simple transit-shape features and trains a baseline logistic regression classifier.
- It evaluates the classifier with accuracy, precision, recall, F1 score, and a confusion matrix.
- It saves plots so you can inspect both the data and the model behavior.
- It includes a GUI where you can upload a CSV light curve and get a transit-like screening result.

## Why This First Version Is Simple

The hardest part of a real exoplanet project is getting reliable labels. For a minimum viable prototype, this project uses real Kepler-style light curves as background data and synthetic transit injection for labels.

Assumption:
This means the first model is learning to detect simple transit-like dips in realistic light-curve noise, not yet learning from a fully curated Kepler candidate catalog.

This is a good research starting point because it is:

- realistic enough to use astronomy data
- simple enough to understand
- easy to improve later

## Folder Structure

```text
Research Project/
|-- app.py                      # Streamlit GUI for uploaded light-curve screening
|-- data/
|   |-- cache/                  # Cached Kepler downloads when available
|   |-- external/               # Future external or user-provided datasets
|   |-- prepared/               # Reusable prepared training datasets
|   `-- samples/                # Small sample CSV files for testing the app
|-- docs/
|   |-- research_notes/         # Notes, ideas, and future planning
|   `-- stage_summaries/        # Written summaries for each project stage
|-- models/                     # Saved trained models
|-- outputs/                    # Saved plots and evaluation results
|   |-- latest_run/             # Most recent training run results
|   |-- archive/                # Older archived runs and legacy results
|   `-- experiment_log.csv      # Lightweight run history
|-- src/
|   |-- exoplanet_research/
|   |   |-- __init__.py         # Package marker
|   |   |-- config.py           # Central project settings
|   |   |-- data_loader.py      # Real Kepler loading + fallback data
|   |   |-- preprocessing.py    # Cleaning, normalization, and windowing
|   |   |-- labeling.py         # Synthetic transit injection and labels
|   |   |-- modeling.py         # Baseline logistic regression model
|   |   |-- evaluation.py       # Metrics and confusion matrix saving
|   |   |-- inference.py        # App-side CSV loading and screening logic
|   |   |-- training_service.py # Prepares a model for the GUI
|   |   |-- visualization.py    # Figures for light curves and results
|   |   `-- pipeline.py         # End-to-end training and evaluation flow
|-- main.py                     # Script entry point
|-- requirements.txt            # Python dependencies
`-- README.md                   # Project explanation and instructions
```

## Installation

Create and activate a virtual environment in VS Code or PowerShell, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## How To Run The Training Pipeline

Run the small MVP pipeline with:

```powershell
python main.py
```

Run the larger Stage 2 data pipeline with:

```powershell
python main.py --stage stage2
```

If you want the run to stop unless real Kepler data is available, use:

```powershell
python main.py --stage stage2 --real-only
```

Run the Stage 3 expanded KOI-labeled Kepler pipeline with:

```powershell
python main.py --stage stage3 --real-only
```

Run the Stage 4 newer TESS/TOI pipeline with:

```powershell
python main.py --stage stage4 --real-only
```

After the run finishes, check the `outputs/` folder for:

- `latest_run/run_summary.txt`
- `latest_run/metrics.json`
- `latest_run/model_comparison.json`
- `latest_run/confusion_matrix.png`
- `latest_run/prediction_scores.png`

The training pipeline now also saves a reusable prepared dataset in `data/prepared/`.

## How To Run The GUI App

Run the local GUI with:

```powershell
streamlit run app.py
```

The app lets you:

- upload a CSV with `time` and `flux` columns
- preview and plot the uploaded data
- screen the light curve with the current baseline model
- inspect window-level transit probability scores

Sample file for testing:

- `data/samples/sample_light_curve.csv`

Important note:
The current app is a research screener, not a final exoplanet-confirmation tool. It can flag transit-like patterns for review, but it should not be presented as proving that an exoplanet exists.

## Deployment

The project can be deployed as a public Streamlit app.

Recommended path:

- push the project to GitHub
- include the saved model files in `models/`
- deploy `app.py` using Streamlit Community Cloud

Deployment notes:

- the hosted app is designed to load saved model files from disk
- it should not rely on retraining at startup
- if a required model file is missing, the app will show a deployment error instead of silently trying to retrain

See [DEPLOYMENT.md](C:\Users\belvo\OneDrive\Desktop\Research Project\DEPLOYMENT.md) for a step-by-step guide.

## Workflow Explanation

### 1. Load Data

The pipeline first tries to download Kepler light curves using the `lightkurve` package.

If that download fails, the code falls back to synthetic light curves so the project still runs locally.

### 2. Preprocess The Data

Each light curve is normalized around its median value so that small dips are easier to compare. Then the time series is split into overlapping windows of equal length.

Equal-size windows are important because machine learning models expect a consistent input shape.

### 3. Create Labels

Some windows are left unchanged and labeled `0` for non-transit.

Other windows receive a simple box-shaped dip and are labeled `1` for transit.

This gives us a clean supervised-learning dataset for the first prototype.

### 4. Train A Baseline Model

The baseline model is logistic regression on a small set of hand-crafted features from each window.

Why logistic regression?

- it is simple
- it is fast
- it is easy to explain
- it gives a clear baseline before trying more advanced models

The features are also beginner-friendly. They summarize things like:

- the minimum flux in the window
- the average and spread of the flux values
- how deep the lowest dip is
- how many points stay unusually low in a row

### 5. Evaluate The Model

The project reports:

- accuracy
- precision
- recall
- F1 score
- confusion matrix

These metrics matter because exoplanet detection is not only about being correct overall. It is also about not missing transit-like signals.

### 6. Use The GUI

The Streamlit app uses the current baseline model to screen uploaded light curves.

You provide a CSV file with `time` and `flux`, and the app:

- loads your data
- plots the light curve
- splits it into windows
- computes transit-like scores for each window
- gives a simple screening result

## Minimum Viable Prototype Design Choices

The first version chooses the simplest good options:

- small sample of data instead of a huge archive
- window-based classification instead of a full detection pipeline
- synthetic transit labels instead of a difficult catalog-matching pipeline
- logistic regression instead of a neural network
- a local Streamlit app instead of a polished desktop application

This keeps the project stable and understandable.

## Staged Plan For Scaling Up

### Stage 1: Current MVP

- Load a small Kepler sample
- Normalize and window the light curves
- Inject synthetic transits
- Train a simple classifier

### Stage 2: Larger Real Dataset

- Download more Kepler targets
- Cache more light curves in `data/cache/`
- Increase the number of windows
- Retrain the same baseline on more real data
- Compare logistic regression with random forest

In the code, Stage 2 now means:

- a larger built-in target list
- more light curves per target
- more windows per light curve
- the same simple baseline model so only the data scale changes

### Stage 3: Better Real Labels

- Use a curated Kepler object catalog
- Match known planet candidates to transit times
- Label windows using real candidate events instead of injected ones

Stage 3 is now started in the codebase. The new workflow:

- downloads or loads the official Kepler KOI cumulative catalog
- matches catalog rows to selected Kepler targets
- computes expected transit times from period and epoch values
- labels windows as positive when they overlap real candidate transit events
- skips ambiguous edge windows that only partially overlap a transit
- compares logistic regression and random forest on the same Stage 3 split

### Stage 4: Newer TESS/TOI Training

- Load newer TESS QLP light curves
- Prefer Sector 94 and later
- Use the official TOI table for real labels
- Compare the same baseline models on newer mission data

### Stage 5: Better Models And Better App

- Try 1D convolutional neural networks
- Try anomaly detection
- Add cross-validation
- Tune hyperparameters
- Save the trained model to disk
- Package the GUI into a more desktop-style app later

## Suggested VS Code Workflow

- Open the project folder in VS Code
- Create a virtual environment
- Install dependencies
- Run `python main.py`
- Run `python main.py --stage stage2` when you are ready for more data
- Run `streamlit run app.py` when you want to use the GUI
- Inspect the `outputs/` figures and metrics
- Change one setting at a time in `src/exoplanet_research/config.py`

## Possible Next Improvements

- replace synthetic labels with known Kepler candidate labels
- compare logistic regression to random forest
- use folded light curves around known periods
- add ROC-AUC and precision-recall curves
- save the trained model to disk
- create a notebook for research exploration
- package the Streamlit prototype into a desktop-style app later

## Short Next Steps

1. Install the dependencies and run `python main.py` or `python main.py --stage stage2`.
2. Run `streamlit run app.py`.
3. Upload a CSV with `time` and `flux` columns.
4. Use the app as a transit-like signal screener while we continue improving the model later.

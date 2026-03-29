# Shepherd.Ai Deployment Guide

This project can be deployed most easily as a public Streamlit app.

## Recommended Platform

- Streamlit Community Cloud

This is the simplest option because the project already uses `app.py` as a Streamlit entry point.

## Before You Deploy

Make sure these files are present in the repository:

- `app.py`
- `requirements.txt`
- `src/`
- `.streamlit/config.toml`
- `models/baseline_stage2_model.joblib`
- `models/baseline_stage3_model.joblib`
- `models/baseline_stage4_model.joblib`

Important:

- The deployed app is designed to load saved model files from `models/`.
- It should not depend on retraining in the cloud.
- If a model file is missing, the app will show a clear deployment error.

## Local Check Before Publishing

From the project folder, run:

```powershell
streamlit run app.py
```

Confirm that:

- the app opens
- the model selector works
- the app loads without trying to retrain
- uploads still analyze correctly

## GitHub Steps

1. Create a GitHub repository.
2. Push this project to the repository.
3. Make sure the `models/` folder is included.

## Streamlit Community Cloud Steps

1. Go to Streamlit Community Cloud.
2. Sign in with GitHub.
3. Choose the Shepherd.Ai repository.
4. Set the main file path to:

```text
app.py
```

5. Deploy the app.

## Updating The Live App

After deployment, maintenance is simple:

1. Make changes locally.
2. Test with `streamlit run app.py`.
3. Commit and push to GitHub.
4. Streamlit will update the live app from the new commit.

## Recommended First Public Framing

This app should be described as:

- a research prototype
- a transit-signal screening tool
- not an exoplanet confirmation tool

## Optional Future Improvements

- add a custom domain
- add an About page
- add a public disclaimer section
- add a lightweight sample-file downloader in the app


# Fake Social Media ID Detector (Demo)

This is a demo Streamlit app that predicts whether a social media profile looks fake.  
**Important:** This is trained on a synthetic dataset for demonstration only — do NOT use for real moderation without proper data and evaluation.

Files:
- `app.py` - Streamlit app
- `fake_id_detector_artifact.pkl` - saved model pipeline artifacts (vectorizer, scaler, classifier)
- `synthetic_fake_social_dataset.csv` - generated synthetic dataset (sample)
- `requirements.txt` - Python dependencies

To deploy on Streamlit Cloud:
1. Create a new GitHub repo and copy these files to the repository root.
2. On Streamlit Cloud, create a new app, connect the repo and branch, and set the main file to `app.py`.
3. The app will install dependencies from `requirements.txt` and should run.

Dataset and model were generated on: 2025-12-03T16:14:35.142038 UTC

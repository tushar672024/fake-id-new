
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack

st.set_page_config(page_title="Fake Social Media ID Detector", layout="centered")

st.title("Fake Social Media ID Detector")
st.markdown("Enter profile details and the model will predict probability that the account is fake (higher -> more likely fake).")

# Load artifacts
@st.cache_data
def load_artifacts():
    artifact = joblib.load("fake_id_detector_artifact.pkl")
    return artifact

artifact = load_artifacts()
vect = artifact["text_vectorizer"]
scaler = artifact["numeric_scaler"]
num_cols = artifact["numeric_columns"]
clf = artifact["classifier"]

with st.form("input_form"):
    username = st.text_input("Username", value="john.doe")
    bio = st.text_area("Bio", value="Software developer. Coffee lover.")
    followers = st.number_input("Followers count", min_value=0, value=150, step=1)
    following = st.number_input("Following count", min_value=0, value=200, step=1)
    posts = st.number_input("Posts count", min_value=0, value=120, step=1)
    account_age_days = st.number_input("Account age (days)", min_value=1, value=800, step=1)
    has_profile_pic = st.radio("Has profile picture?", ["Yes","No"])
    verified = st.radio("Verified?", ["No","Yes"])
    submitted = st.form_submit_button("Check")

if submitted:
    is_pic = 1 if has_profile_pic=="Yes" else 0
    is_verified = 1 if verified=="Yes" else 0
    digits_in_username = sum(c.isdigit() for c in username)
    username_len = len(username)
    numeric_ratio = digits_in_username / max(1, username_len)
    special_chars = sum(not c.isalnum() for c in username)
    bio_len = len(bio)
    contains_link = 1 if ("http" in bio or "bit.ly" in bio or "tiny" in bio) else 0
    contains_offer = 1 if any(x in bio.lower() for x in ["offer","free","giveaway","buy","cheap","click"]) else 0
    contains_money = 1 if any(x in bio.lower() for x in ["$", "earn", "$$$", "profit"]) else 0

    num_feat = [followers, following, posts, account_age_days, is_pic, is_verified,
                digits_in_username, username_len, numeric_ratio, special_chars, bio_len,
                contains_link, contains_offer, contains_money]
    num_df = pd.DataFrame([num_feat], columns=num_cols)
    num_scaled = scaler.transform(num_df)

    text_input = (username + " " + bio)
    text_vec = vect.transform([text_input])

    X = hstack([text_vec, num_scaled])
    proba = clf.predict_proba(X)[:,1][0]
    label = "Fake" if proba>0.5 else "Likely Real"

    st.metric("Prediction", label, delta=f"{proba*100:.2f}% probability of being fake")
    st.write("Probability score (0-1):", float(proba))

    # Simple explainability: feature importances for numeric columns
    import numpy as np
    try:
        importances = clf.feature_importances_
        # note: first chunk of importances correspond to text vector features (we skip), last correspond to numeric cols
        num_imp = importances[-len(num_cols):]
        imp_df = pd.DataFrame({"feature": num_cols, "importance": num_imp})
        imp_df = imp_df.sort_values("importance", ascending=False)
        st.write("Top numeric feature importances (approx):")
        st.table(imp_df.style.format({"importance":"{:.4f}"}))
    except Exception as e:
        st.write("Feature importance not available:", e)

st.markdown("---")
st.markdown("**Notes:** This is a synthetic-model demo. Real deployments must use real labeled data, robust features (image analysis, behaviour over time, network patterns), and careful privacy/compliance checks.")
st.markdown("You can download the sample synthetic dataset below.")
try:
    df = pd.read_csv("synthetic_fake_social_dataset.csv")
    st.download_button("Download synthetic dataset CSV", data=df.to_csv(index=False).encode('utf-8'), file_name="synthetic_fake_social_dataset.csv", mime="text/csv")
except Exception as e:
    st.write("Dataset not found:", e)

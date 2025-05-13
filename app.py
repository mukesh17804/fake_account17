
import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("model.pkl")

st.title("🚨 Fake Account Detector")

followers = st.number_input("Followers", min_value=0)
following = st.number_input("Following", min_value=0)
is_verified = st.selectbox("Is Verified?", [0, 1])
posts = st.number_input("Number of Posts", min_value=0)
likes_per_post = st.number_input("Average Likes per Post", min_value=0)
has_profile_picture = st.selectbox("Has Profile Picture?", [0, 1])
comments_per_post = st.number_input("Average Comments per Post", min_value=0)
suspicious_words = st.selectbox("Suspicious Words in Bio/Username?", [0, 1])

if st.button("Predict"):
    features = np.array([[followers, following, is_verified, posts, likes_per_post,
                          has_profile_picture, comments_per_post, suspicious_words]])
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.error("⚠️ This might be a Fake Account!")
    else:
        st.success("✅ This appears to be a Genuine Account.")

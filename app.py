import streamlit as st

st.set_page_config(
    page_title="Telugu Music Genre Classifier",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="collapsed"
)

import numpy as np
import librosa
import joblib

# Load your trained model
model = joblib.load('telugu_genre_classifier.pkl')

# Feature extraction function
def extract_features(file):
    y, sr = librosa.load(file, duration=30)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_centroid_mean = float(np.mean(spec_centroid))

    features = [float(tempo), spec_centroid_mean]
    features.extend([float(val) for val in mfcc_mean])
    features.extend([float(val) for val in chroma_mean])

    return np.array(features).reshape(1, -1)

# ---------------- UI ----------------

st.title("🎶 Telugu Music Genre Classifier")
st.write("Upload a **Telugu song (MP3/WAV)** to predict its genre using Machine Learning.")

uploaded_file = st.file_uploader("Choose a Telugu song to analyze", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav', start_time=0)
    if st.button("Predict Genre 🎵"):
        with st.spinner('Analyzing the song and predicting...'):
            try:
                features = extract_features(uploaded_file)
                prediction = model.predict(features)[0]
                st.success(f"✅ Predicted Genre: **{prediction}**")
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Machine Learning.")

# ---- Remove deploy button and 3-dot menu ----
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

import os
import numpy as np
import pandas as pd
import librosa
import joblib

# Load trained model
model = joblib.load('telugu_genre_classifier.pkl')

# Function to extract features for a single file
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)
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

# File to predict
file_to_predict = input("Enter the path to the Telugu song file (wav/mp3): ")

# Extract features
features = extract_features(file_to_predict)

# Predict
predicted_genre = model.predict(features)[0]
print(f"\n✅ Predicted Genre: {predicted_genre}")

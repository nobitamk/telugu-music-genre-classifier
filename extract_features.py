import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm

# Paths
AUDIO_FOLDER = 'audio_files'
LABELS_FILE = os.path.join(AUDIO_FOLDER, 'labels.csv')
OUTPUT_FOLDER = 'data'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load labels
labels_df = pd.read_csv(LABELS_FILE)

# Prepare data storage
features_list = []

print("Extracting features, please wait...")

for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
    filename = row['filename']
    genre = row['genre']
    filepath = os.path.join(AUDIO_FOLDER, filename)
    try:
        y, sr = librosa.load(filepath, duration=30)  # Load first 30 sec
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1).tolist()
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1).tolist()
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_centroid_mean = float(np.mean(spec_centroid))

        feature_row = [filename, float(tempo), spec_centroid_mean]
        feature_row.extend([float(val) for val in mfcc_mean])
        feature_row.extend([float(val) for val in chroma_mean])
        feature_row.append(genre)

        features_list.append(feature_row)

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Column names
columns = ['filename', 'tempo', 'spec_centroid'] + \
          [f'mfcc_{i}' for i in range(1, 14)] + \
          [f'chroma_{i}' for i in range(1, 13)] + \
          ['genre']

# Save to CSV
features_df = pd.DataFrame(features_list, columns=columns)
features_df.to_csv(os.path.join(OUTPUT_FOLDER, 'features.csv'), index=False)

print("✅ Feature extraction complete! Saved to data/features.csv")

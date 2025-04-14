import streamlit as st
import soundfile as sf
import tempfile
import torch.nn as nn
import torch
import os
import librosa
import numpy as np

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean


class GenreClassifier(nn.Module):
    def __init__(self, input_size, num_classes=10):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes)  # final output (logits)
        )

    def forward(self, x):
        return self.classifier(x)

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

def predict_genre(file_path):
    features = extract_features(file_path)
    model = GenreClassifier(input_size=len(features), num_classes=10)
    model.load_state_dict(torch.load("model/model.pth"))
    model.eval()

    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        predictions = model(x)
        predicted_idx = predictions.argmax().item()

    return GENRES[predicted_idx]


st.set_page_config(page_title="Music Genre Classifier 🎵", layout="centered")
st.title("🎤 Record Your Music and Predict the Genre")

# Create a recording UI
st.markdown("### Step 1: Record Audio or Upload File")

audio_file = st.file_uploader("Upload a WAV file", type=["wav"])

# Placeholder for prediction result
prediction = None

if audio_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name
        st.audio(tmp_path, format="audio/wav")
        
        # Predict genre
        if st.button("🎧 Predict Genre"):
            prediction = predict_genre(tmp_path)
            st.success(f"🎶 Predicted Genre: **{prediction}**")

st.markdown("---")
st.markdown("Built with ❤️ to learn Deep Learning.")

# Imports the Streamlit library for creating interactive web applications.
import streamlit as st
# Imports the SoundFile library for reading and writing audio files.
import soundfile as sf
# Imports the tempfile module for creating temporary files.
import tempfile
import predict as pr
import torch.nn as nn
import torch
import os
import librosa
import numpy as np


# Set the page configuration for the Streamlit app.
st.set_page_config(page_title="Music Genre Classifier 🎵", layout="centered")
# Set the title of the Streamlit application.
st.title("🎤 Record Your Music and Predict the Genre")

# Create a section in the UI for recording or uploading audio.
st.markdown("### Step 1: Record Audio or Upload File")

# Create a file uploader widget that accepts WAV files.
audio_file = st.file_uploader("Upload a WAV file", type=["wav"])

# Placeholder variable to store the prediction result.
prediction = None

# Check if an audio file has been uploaded.
if audio_file:
    # Create a temporary file to store the uploaded audio data.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        # Write the content of the uploaded audio file to the temporary file.
        tmp.write(audio_file.read())
        # Get the name (path) of the temporary file.
        tmp_path = tmp.name
        # Display an audio player in the Streamlit app for the uploaded audio.
        st.audio(tmp_path, format="audio/wav")

        # Create a button in the Streamlit app to trigger genre prediction.
        if st.button("🎧 Predict Genre"):
            # Call the predict_genre function with the path to the uploaded audio file.
            prediction = pr.predict_genre(tmp_path)
            # Display the predicted genre in a success message.
            st.success(f"🎶 Predicted Genre: **{prediction}**")

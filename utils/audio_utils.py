import librosa
import numpy as np

def extract_features(file_path):
    """
    Extracts Mel-Frequency Cepstral Coefficients (MFCCs) from an audio file.

    Args:
        file_path (str): The path to the audio file.

    Returns:
        numpy.ndarray: A 1D array containing the mean of the MFCCs over time.
    """
    
    # Load the audio file using librosa.
    # 'file_path' is the path to a .wav (or any compatible) audio file.
    # 'sr=None' ensures librosa uses the file’s original sampling rate rather than resampling it to the default (22050 Hz).
    #
    # Output:
    # - y: a 1D NumPy array containing the raw audio waveform (amplitude values over time).
    #      len(y) = sr * duration (in seconds)
    # - sr: the sampling rate (samples per second), which defines how many values exist per second of audio.
    y, sr = librosa.load(file_path, sr=None)  

    # Compute the Mel-Frequency Cepstral Coefficients (MFCCs) from the raw audio waveform.
    # MFCCs are a compact, perceptually motivated representation of the short-term power spectrum of a sound.
    # Key Parameters:
    # - y: the audio time series (waveform)
    # - sr: the sampling rate
    # - n_mfcc=13: number of MFCC coefficients to extract per frame
    #
    # Output:
    # - mfcc: a 2D NumPy array of shape (n_mfcc, T), where:
    #   - n_mfcc: number of coefficients (vertical axis, 13 rows)
    #   - T: number of time frames (horizontal axis, e.g., ~427 for a 10-second clip with 512 hop)
    # Each column of this matrix represents a "snapshot" of the sound's frequency characteristics at that time.
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Transpose the MFCC matrix and calculate the mean of each coefficient across all time frames.
    mfcc_mean = np.mean(mfcc.T, axis=0)

    return mfcc_mean
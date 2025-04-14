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
    
    # Load the audio file using librosa, keeping the original sampling rate.
    y, sr = librosa.load(file_path, sr=None)  

    # Compute the MFCCs with 13 coefficients.
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Transpose the MFCC matrix and calculate the mean of each coefficient across all time frames.
    mfcc_mean = np.mean(mfcc.T, axis=0)

    return mfcc_mean
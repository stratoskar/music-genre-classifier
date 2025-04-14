import torch
import librosa
import numpy as np
import torch.nn as nn

# --------- Neural Network Model ---------
class GenreClassifier(nn.Module):
    # Define the neural network model for genre classification.
    def __init__(self, input_size, num_classes=10):
        # Initialize the layers of the neural network.
        super().__init__()
        # A sequential container for the layers of the classifier.
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

            # Fully connected layer to the number of classes.
            nn.Linear(128, num_classes)
        )

    # Define the forward pass of the neural network.
    def forward(self, x):
        return self.classifier(x)

# A list of the music genre labels that the model is trained to predict.
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']  

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

def predict_genre(file_path):
    """
    This function is used to predict the music genre of a given audio file using a trained model.

    Args:
        file_path (str): The path to the audio file.

    Returns:
        str: The predicted music genre.
    """

    # Extract the audio features from the input file.
    features = extract_features(file_path)  

    # Initialize the genre classification model with the correct input size (number of features) and number of output classes (genres).
    model = GenreClassifier(input_size=len(features), num_classes=10)  
    
    # Load the trained weights and biases into the model from the 'model.pth' file.
    model.load_state_dict(torch.load("model/model.pth"))  
    
    # Set the model to evaluation mode, 
    # which disables dropout and batch normalization's training behavior.
    model.eval()  

    # Disable gradient calculations
    with torch.no_grad():  
        # Convert the NumPy array to a PyTorch tensor with a batch dimension of 1.
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  
        
        # Pass the input tensor through the trained model to get the prediction scores for each genre.
        predictions = model(x) 
        
        # Find the index of the genre with the highest prediction score.
        predicted_idx = predictions.argmax().item()  

    # Return the genre label corresponding to the predicted index.
    return GENRES[predicted_idx]

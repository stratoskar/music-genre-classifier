import os
import numpy as np
from utils.audio_utils import extract_features

# Define a list of the music genre labels.
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

def prepare_dataset():
    # Initialize an empty list to store the extracted audio features.
    features = []
    # Initialize an empty list to store the corresponding genre labels (as numerical indices).
    labels = []

    # Iterate through each genre in the 'GENRES' list along with its index.
    for idx, genre in enumerate(GENRES):
        # Construct the path to the folder containing audio files for the current genre.
        genre_folder = f"data/raw/{genre}"
        # Iterates through each file in the specified genre folder.
        for file in os.listdir(genre_folder):
            # Check if the file ends with the ".wav" extension, indicating an audio file.
            if file.endswith(".wav"):
                # Construct the full path to the audio file.
                path = os.path.join(genre_folder, file)

                try:
                    # Call the 'extract_features' function to get the MFCC features from the audio file.
                    mfcc = extract_features(path)

                    # Append the extracted MFCC features to the 'features' list.
                    features.append(mfcc)

                    # Append the numerical index of the current genre to the 'labels' list.
                    labels.append(idx)

                except Exception as e:
                    # Print an error message indicating the problematic file and the error.
                    print(f"Error with {file}: {e}")

    # Create the "data/processed" directory if it doesn't already exist.
    os.makedirs("data/processed", exist_ok=True)

    # Save the extracted features and labels as a NumPy compressed array (.npz file).
    np.savez("data/processed/features.npz", X=np.array(features), y=np.array(labels))

    # Print a message indicating that the dataset preparation is complete and the file has been saved.
    print("Dataset prepared and saved.")
    
if __name__ == "__main__":
    # Call the 'prepare_dataset' function to start the dataset preparation process.
    prepare_dataset()
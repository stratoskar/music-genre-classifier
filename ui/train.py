import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import time
import wandb # For experiment tracking and logging.
import os
import yaml # For reading configuration files in YAML format.

# --------- Dataset Class ---------
class GenreDataset(Dataset):
    # Define a custom Dataset class to load and manage the music genre data.
    def __init__(self, features, labels):
        # Initialize the dataset with features and labels.
        self.X = torch.tensor(features, dtype=torch.float32)  # Convert the input features to a PyTorch float tensor.
        # Convert the input labels to a PyTorch long tensor.
        self.y = torch.tensor(labels, dtype=torch.long)

    # Return the total number of samples in the dataset.
    def __len__(self):
        return len(self.y)

    # Retrieve a single sample (features and label) from the dataset at the given index.
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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

# --------- Load Config ---------
def load_config():
    # Load training configuration from a YAML file.
    try:
        # Open and read the YAML configuration file.
        with open("config/config.yaml") as f:
            # Safely load the YAML content into a Python dictionary.
            return yaml.safe_load(f)
    # Handle the case where the configuration file is not found.
    except FileNotFoundError:
        print("Config file not found. Using default values.")
        return {
            "batch_size": 32,
            "lr": 0.001,
            "epochs": 120
        }

# --------- Training Function ---------
def train():
    # Main function to train the genre classification model.
    try:
        # Load the training configuration.
        config = load_config()
        # Initialize Weights & Biases for experiment tracking.
        wandb.init(project="music-genre", config=config)
        
        # Access the configuration values from WandB.
        config = wandb.config

        print("Config loaded:")
        print(config)

        # Path to the processed feature data.
        data_path = "data/processed/features.npz"
        
        # Check if the dataset file exists.
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")

        print("Loading dataset...")
        
        # Load the data from the .npz file.
        data = np.load(data_path)
        
        # Extract features (X) and labels (y) from the loaded data.
        X, y = data["X"], data["y"]
        print(f"Dataset loaded: {X.shape[0]} samples with {X.shape[1]} features")

        # Split the data into training and validation sets.
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Data split into training and validation sets.")

        # Create a DataLoader for the training set.
        train_loader = DataLoader(GenreDataset(X_train, y_train), batch_size=config.batch_size, shuffle=True)
        
        # Create a DataLoader for the validation set.
        val_loader = DataLoader(GenreDataset(X_val, y_val), batch_size=config.batch_size)

        # Initialize the GenreClassifier model with the correct input size.
        model = GenreClassifier(input_size=X.shape[1])
        
        # Initialize the Adam optimizer.
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        
        # Define the loss function for multi-class classification.
        criterion = nn.CrossEntropyLoss()

        print("Training started...\n")
        # Records the starting time of the training.
        start_time = time.time()

        # Iterate through the specified number of training epochs.
        for epoch in range(config.epochs):
            # Set the model to training mode.
            model.train()
            # Initialize the total loss for the epoch.
            total_loss = 0

            # Iterate through batches of training data.
            for x_batch, y_batch in train_loader:
                # Clear the gradients from the previous iteration.
                optimizer.zero_grad()
                # Perform a forward pass through the model.
                predictions = model(x_batch)
                # Calculate the loss between predictions and true labels.
                loss = criterion(predictions, y_batch)
                # Perform backpropagation to calculate gradients.
                loss.backward()
                # Update the model's weights based on the gradients.
                optimizer.step()
                # Accumulate the loss for the epoch.
                total_loss += loss.item()

            # Set the model to evaluation mode.
            model.eval()
            # Initialize the count of correct predictions on the validation set.
            correct = 0
            # Initialize the total number of samples in the validation set.
            total = 0
            # Disable gradient calculations during validation.
            with torch.no_grad():
                # Iterate through batches of validation data.
                for x_val, y_val in val_loader:
                    # Perform a forward pass on the validation batch.
                    preds = model(x_val)
                    # Get the predicted class labels.
                    _, predicted = torch.max(preds, 1)
                    # Update the total number of samples.
                    total += y_val.size(0)
                    # Count the number of correct predictions.
                    correct += (predicted == y_val).sum().item()

            # Calculate the validation accuracy.
            acc = correct / total
            print(f"📈 Epoch {epoch+1}/{config.epochs} - Loss: {total_loss:.4f} | Val Acc: {acc:.4f}")
            # Logs the epoch number, loss, and validation accuracy to WandB.
            wandb.log({"epoch": epoch + 1, "loss": total_loss, "val_acc": acc})

        # Calculate the total training time.
        elapsed = time.time() - start_time
        print(f"\n✅ Training completed in {elapsed:.2f} seconds.")

        # Create the 'model' directory if it doesn't exist.
        os.makedirs("model", exist_ok=True)
        # Save the trained model's state dictionary.
        torch.save(model.state_dict(), "model/model.pth")
        # Save the model checkpoint to WandB.
        wandb.save("model/model.pth")
        print("💾 Model saved to model/model.pth")

    # Handles any exceptions that occur during the training process.
    except Exception as e:
        print(f"❌ Error during training: {e}")

# Calls the training function to start the training process.
train()
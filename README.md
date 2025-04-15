
# 🎵 Music Genre Classification with Deep Learning

## 📌 Repository Description

This repository contains a complete machine learning pipeline for **Music Genre Classification**, from live audio recording or file upload to genre prediction using a deep neural network trained on extracted audio features.

Developed with **PyTorch**, integrated with **Weights & Biases** for experiment tracking, and built around a modular project structure for clarity and extendability.

> The project has been crafted as a learning experience in deep learning, data pipelines, and model deployment — all assisted and co-developed with **ChatGPT**. It is ideal for those who are beginning their journey in AI and music data analysis.

---

## 🧠 Project Overview

**Goal**: Predict the genre of a music clip (e.g., Classical, Rock, Pop, Jazz, Country, etc.) using a deep learning model trained on a preprocessed dataset of audio features.

**Core Features:**
- Live recording or song upload via a Streamlit interface.
- Feature extraction using `librosa` and custom signal processing.
- A PyTorch-based classifier architecture.
- Configurable training using `config.yaml`.
- Experiment tracking with Weights & Biases.
- End-to-end reproducible pipeline.

---

## 🎼 Dataset

The project uses the **GTZAN Genre Collection** dataset, a well-known benchmark for music genre classification tasks.

- **Dataset Name**: GTZAN Genre Collection
- **Content**: 1000 audio files, each 30 seconds long, divided into 10 genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock).
- **Source**: [Marsyas](http://marsyas.info/downloads/datasets.html)
- **Citation**:

> [1] G. Tzanetakis and P. Cook, “Musical genre classification of audio signals,” *IEEE Transactions on Speech and Audio Processing*, vol. 10, no. 5, pp. 293–302, 2002.

Please make sure to download the dataset manually and place it under the `data/raw` directory.

---

## 🏗️ Installation

### 1. Clone the repository:

```bash
git clone https://github.com/yourusername/music-genre-classifier.git
cd music-genre-classifier
```

### 2. Set up a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # or use `venv\Scripts\activate` on Windows
```

### 3. Install dependencies:

```bash
pip install -r requirements.txt
```

### 4. Set up Weights & Biases (Optional):

```bash
wandb login
```

### 5. Download the dataset:

Download the GTZAN dataset from the Marsyas project [here](http://marsyas.info/downloads/datasets.html), unzip it, and place it under `data/raw/`.

---

## ▶️ Usage

### To train the model:

```bash
python model/train.py
```

### To launch the UI:

```bash
streamlit run ui/streamlit_app.py
```

You can either record live audio or upload an existing file for genre prediction.

---

## 🧪 Deep Learning Technologies Used

This project employs modern deep learning technologies and best practices:

- **PyTorch**: for model definition, training loops, and inference.
- **Weights & Biases**: for experiment tracking and hyperparameter logging.
- **Scikit-learn**: for data splitting and evaluation utilities.
- **Librosa**: for audio signal processing and feature extraction (MFCCs).
- **Streamlit**: for building an interactive, user-friendly interface.

The model architecture includes:
- Fully connected layers with ReLU activation
- Dropout for regularization
- CrossEntropy loss for multi-class classification
- Adam optimizer

---

## 📁 Project Structure

```
music-genre-classifier/
│
├── data/
│   ├── raw/               # Place GTZAN dataset here
│   └── processed/         # Saved features and labels (npz)
│
├── model/
│   ├── train.py           # Model architecture and training script
│   └── predict.py         # Inference functions
│
├── utils/
│   └── audio_utils.py     # Feature extraction utilities
│
├── ui/
│   ├── record_ui.py       # Optional CLI-based recorder
│   └── streamlit_app.py   # Streamlit user interface
│
├── config/
│   └── config.yaml        # Configurable parameters (batch size, lr, epochs, etc.)
│
└── requirements.txt       # Project dependencies
```

---

## 👤 Author

**Efstratios Karkanis**  
Email: [stratoskarkanis2@gmail.com](mailto:stratoskarkanis2@gmail.com)  
Field: Deep Learning | Signal Processing | Music AI

This project was developed with the assistance of **ChatGPT** for educational purposes, aiming to demystify deep learning for curious minds.

---

## 🚀 Stay Curious!

> “The important thing is not to stop questioning. Curiosity has its own reason for existing.” — Albert Einstein

Feel free to fork, star, or contribute to this project. Let’s learn and build together! 🎧

---

## 📚 References

[1] G. Tzanetakis and P. Cook, “Musical genre classification of audio signals,” *IEEE Transactions on Speech and Audio Processing*, vol. 10, no. 5, pp. 293–302, 2002. [Online]. Available: http://marsyas.info/downloads/datasets.html

---


# 🎵 Music Genre Classification with Deep Learning

## 📌 Repository Description

This repository contains a complete machine learning pipeline for **Music Genre Classification**, from file upload to genre prediction using a deep neural network trained on extracted audio features.

Developed with **PyTorch**, integrated with **Weights & Biases** for experiment tracking, and built around a modular project structure for clarity and extendability.

> The project has been crafted as a learning experience in deep learning, data pipelines, and model deployment — all assisted and co-developed with **ChatGPT**.

---

## 🧠 Project Overview

**Goal**: Predict the genre of a music clip (e.g., Classical, Rock, Pop, Jazz, Country, etc.) using a deep learning model trained on a preprocessed dataset of audio features.

**Core Features:**
- Song upload via a Streamlit interface.
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

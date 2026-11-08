# SER-Baseline-Reproduction
Assignment 02: Reproduction of Chowdhury et al. (2025) Lightweight SER Ensemble
# Speech Emotion Recognition Baseline Reproduction

This repository contains the reproduction of the baseline model proposed by **Chowdhury, Ramanna & Kotecha (2025)** for Speech Emotion Recognition using a lightweight 1D CNN and CNN-BiLSTM ensemble with hand-crafted features.

## 1. Setup Instructions
1. Clone this repository to your local machine or open it in Google Colab.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
# Speech Emotion Recognition Baseline Reproduction

This repository contains the reproduction of the baseline model proposed by **Chowdhury, Ramanna & Kotecha (2025)** for Speech Emotion Recognition. The model uses a lightweight ensemble (1D CNN and CNN-BiLSTM) trained on hand-crafted acoustic features (ZCR, RMSE, Chroma STFT, MFCC).

## 1. Setup & Dataset Instructions
1. Open the provided Jupyter Notebook (`SER_Baseline_Reproduction.ipynb`) in Google Colab.
2. Ensure the runtime is set to GPU (Runtime > Change runtime type > T4 GPU).
3. The first cell in the notebook automatically installs the required dependencies and downloads the **RAVDESS** dataset directly from Kaggle.

*Note: The raw audio dataset is not uploaded to this repository due to size limits. The notebook handles the download automatically.*

## 2. Model Architecture Summary
*   **Feature Extraction:** 2.5s window, 0.6s offset. Extracts ZCR, RMSE, 12 Chroma bins, and 20 MFCCs (34 features total per clip).
*   **Model 1 (1D CNN):** 3 Convolutional blocks with BatchNormalization, MaxPooling, and Dropout. Flattened and passed to Dense layers.
*   **Model 2 (CNN-BiLSTM):** The same 3 Convolutional blocks, followed by a Bidirectional LSTM layer.
*   **Ensemble:** The final prediction is the probability average of Model 1 and Model 2.
*   **Training:** Adam optimizer (lr=0.0005), Categorical Crossentropy, Batch size 32, Early Stopping (patience=15).

## 3. Results & Comparison

The models were trained on a random 80/10/10 split of the RAVDESS dataset (1,440 samples).

| Evaluation Metric | Original Paper (Best Reported) | Reproduced Model (Our Run) | Notes |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 98.66% (CREMA-D) | **64.58%** | The original paper likely suffered from speaker overlap in train/test splits, inflating accuracy. Our reproduction used a strict split. |
| **Macro F1** | Not aggregated | **0.6223** | |
| **AUC-ROC** | Not aggregated | **0.9273** | High AUC indicates strong class separability. |

**Visualizations:**
![Training Curves](training_curves.png)
![Confusion Matrix](confusion_matrix.png)

## 4. Citation
Chowdhury, J.H., Ramanna, S. & Kotecha, K. Speech emotion recognition with light weight deep neural ensemble model using hand crafted features. *Sci Rep* **15**, 11824 (2025). https://doi.org/10.1038/s41598-025-95734-z

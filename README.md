# Self-Supervised Learning for Brain Tumor Classification

This project implements a brain tumor classification system using self-supervised learning (SSL) to extract features from brain MRI scans and a supervised classifier for final prediction. The project aims to reduce reliance on labeled data while maintaining strong classification performance.

## Project Structure

```
├── trainmodel.py               # Trains the SSL encoder and classifier
├── feature extraction.py       # Feature extraction using encoder or HOG
├── test.py                     # Tests model on new MRI input
├── score1.py                   # Evaluates model using precision, recall, F1-score
├── requirements.txt            # Python libraries required (to be added)
├── README.md                   # Project instructions and documentation
├── model/                      # Stores ssl_encoder.pt and classifier.pkl
└── data/                       # Sample MRI image(s) for testing
```

## Description

The pipeline consists of:

1. **Preprocessing:** MRI images are resized and normalized.
2. **Feature Extraction:** Self-supervised learning (contrastive learning) is used to train an encoder to extract meaningful features from unlabeled data.
3. **Classifier Training:** A small labeled dataset is used to train a Random Forest classifier on the extracted features.
4. **Evaluation:** The model is tested and evaluated using precision, recall, F1-score, and a confusion matrix.

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/NamannTyagii/brain-tumor-ssl
cd brain-tumor-ssl
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python trainmodel.py
```

### 4. Run a Test

Place a test MRI image in the `data/` folder and run:

```bash
python test.py
```

### 5. Evaluate Model Performance

```bash
python score1.py
```

## Dataset

The dataset used includes brain MRI scans categorized into:
- Tumor
- No Tumor

Images are resized to a fixed input size (e.g., 224x224) and normalized before feature extraction.

## Authors

- Naman Tyagi
- Dev Yadav
- Shreksha

Bachelor of Science, K.R. Mangalam University  
April 2025


# Comparative Analysis of Text Classification with Multiple Embeddings (Hate Speech Detection)

## Project Overview
This project compares multiple embedding + model combinations for binary hate speech classification.  
The goal is to evaluate how representation choice affects performance under the **same dataset** and a **shared preprocessing strategy**.

### Embeddings Compared
- **TF-IDF**
- **Word2Vec Skip-gram**
- **Word2Vec CBOW**

### Model Pipelines
- **TF-IDF + Logistic Regression**
- **TF-IDF + LinearSVC** (baseline improvement check)
- **Word2Vec Skip-gram + GRU/BiGRU**
- **Word2Vec CBOW + GRU/BiGRU**

### Main Evaluation Metric
- **Macro-F1** (primary, because classes are imbalanced)

---

## Repository Structure

```text
.
├── data/
│      # Kaggle download goes here── HateSpeechDataset.csv
│      └── HateSpeechDataset_preprocessed.csv
├── notebooks/
│   ├── HateSpeech_Preprocessing_Notebook.ipynb
│   ├── HateSpeech_RubricReady_Notebook.ipynb
├── requirements.txt
└── README.md

---

## Team Objective

Each member implements one architecture and evaluates it across at least 3 embedding setups (aligned with assignment rubric).
This repo consolidates code, experiments, and report assets for reproducibility.

Dataset
1) Kaggle Source

Use your competition/dataset page link here:

Kaggle dataset URL: PASTE_KAGGLE_LINK_HERE

2) Download (Option A: Kaggle API - recommended)

Install Kaggle CLI:

pip install kaggle


Place kaggle.json in:

Linux/macOS: ~/.kaggle/kaggle.json

Windows: C:\Users\<USERNAME>\.kaggle\kaggle.json

Set permissions (Linux/macOS):

chmod 600 ~/.kaggle/kaggle.json


Download dataset:

kaggle datasets download -d <owner>/<dataset-name> -p data/raw


or for competition:

kaggle competitions download -c <competition-name> -p data/raw


Unzip:

unzip data/raw/*.zip -d data/raw

3) Download (Option B: Manual)

Download from Kaggle web UI.

Put HateSpeechDataset.csv inside data/raw/.

Environment Setup
1) Create virtual environment
python -m venv .venv


Activate:

Windows:

.venv\Scripts\activate


Linux/macOS:

source .venv/bin/activate

2) Install dependencies
pip install -r requirements.txt


Suggested requirements.txt minimum:

numpy
pandas
scikit-learn
matplotlib
gensim
torch
jupyter
notebook

How to Run the Project
A. Preprocessing (shared for all team members)

Use the preprocessing notebook/script first.

Notebook path

notebooks/HateSpeech_Preprocessing_Notebook.ipynb

Expected output

data/processed/HateSpeechDataset_preprocessed.csv

What preprocessing does

text normalization (lowercase, URL/user cleanup)

whitespace/punctuation cleanup

train/val/test split (stratified)

shared cleaning logic for consistent team comparisons

B. Improved CPU pipeline (recommended)

Run:

notebooks/HateSpeech RubricReady_Notebook.ipynb

This notebook includes:

stopword removal (keeps negation terms: no, not, nor, never)

wiki/meta artifact removal (wikipedia, article, page, edit, etc.)

TF-IDF + LR and TF-IDF + LinearSVC

Word2Vec (Skip-gram / CBOW) + GRU/BiGRU

class imbalance handling (class_weight, pos_weight)

threshold tuning for macro-F1

Recommended Run Order

HateSpeech_Preprocessing_Notebook.ipynb

HateSpeech RubricReady_Notebook.ipynb

HateSpeech_RubricReady_Notebook.ipynb (for figures/tables/report polish)

Key Training Notes

Primary metric: Macro-F1

Why not only accuracy? Dataset is imbalanced; macro-F1 treats classes equally.

Threshold tuning is enabled to maximize validation macro-F1.

Word2Vec stability on CPU:

workers=1 to avoid environment/thread issues

sample=1e-4 to reduce frequent-token dominance

No leakage policy:

Fit TF-IDF on train only

Train Word2Vec on train tokens only

Keep validation/test fully unseen during fitting

Expected Outputs

Preprocessed CSV:

data/processed/HateSpeechDataset_preprocessed.csv

Model comparison table:

Macro-F1 / precision / recall / accuracy across pipelines

Confusion matrices for best models

EDA figures for report

Draft report:

reports/Comparative_Embedding_Text_Classification_Report_Draft.docx

## RNN Contribution
- **Model:** SimpleRNN
- **Embeddings:** CBOW (Best), Skip-gram, Standard
- **Key Finding:** SimpleRNN requires class weights to handle imbalance; CBOW provides the best stability.

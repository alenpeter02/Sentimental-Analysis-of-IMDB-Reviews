## 📁 Project 1: IMDb Movie Review Sentiment Analysis

### Overview
Binary classification project to predict whether IMDb movie reviews are **positive** or **negative** using various machine learning and deep learning approaches.

### Dataset
- **Source**: IMDB Dataset (50,000 reviews)
- **Classes**: Balanced distribution (24,884 positive, 24,696 negative)
- **Features**: Review text and sentiment labels

### Preprocessing
- HTML tag removal, punctuation and special character cleaning
- Tokenization and lowercasing
- Stopword removal, stemming, and lemmatization
- Duplicate removal

### Feature Engineering
- **Text Features**: TF-IDF with bigrams (top 5,000 words)
- **Numeric Features**: Word count, sentence count, average word/sentence length
- **Insight**: Numeric features showed low importance compared to textual content

### Models & Performance

| Model | Accuracy | ROC-AUC | F1-Score | Key Observations |
|-------|----------|---------|----------|------------------|
| Random Forest | 85% | 0.93 | 0.85 | Good positive sentiment detection (Recall=0.91) |
| Logistic Regression | **89%** | **0.96** | **0.89** | **Best overall performance**, balanced precision/recall |
| Naive Bayes | 86% | 0.93 | 0.86 | Balanced performance across classes |
| Bi-LSTM | 86.5% | 0.94 | 0.86 | Excellent balance, bidirectional processing |

### Recommendation
**Logistic Regression** is recommended for this task due to:
- Highest accuracy, F1-score, and ROC-AUC
- Balanced performance for both positive and negative sentiments
- Lowest misclassification rates in confusion matrix


# DistilBERT Sentiment Analysis for IMDb Reviews

A transformer-based sentiment analysis project that fine-tunes DistilBERT on IMDb movie reviews to classify them as positive or negative, with full MLflow tracking and Gradio deployment.

## Project Overview

This project implements a state-of-the-art sentiment analysis system using Hugging Face's DistilBERT model fine-tuned on the IMDb movie review dataset. The solution includes:

- **Fine-tuning** of pre-trained DistilBERT model
- **Comprehensive evaluation** with multiple metrics
- **MLflow experiment tracking** for reproducibility
- **Gradio web interface** for real-time predictions
- **Model deployment** pipeline

## Dataset

- **Source**: IMDb Dataset (50,000 reviews)
- **Classes**: Balanced (25,000 positive, 25,000 negative)
- **Features**: 
  - `review`: Text content of the movie review
  - `sentiment`: Target label (positive/negative)

### Sample Data
```python
# Raw dataset info
Columns: review, sentiment
Class distribution: Perfectly balanced
The file size was reduced for the ease of training

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 87.3% |
| F1-Score | 87.5% |
| ROC-AUC | 94.0% |

## Tech Stack

- **Transformers** (Hugging Face)
- **MLflow** for experiment tracking
- **Gradio** for web deployment
- **PyTorch** backend

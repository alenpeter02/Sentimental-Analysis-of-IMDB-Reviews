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

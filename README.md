
# 🛡️ PromptShield

A machine-learning-based prompt injection detection system.

## Models

- DeBERTa v1 — primary detection model.
- TF-IDF + Logistic Regression v1 — classical ML baseline.

## Features

- Prompt injection scanning.
- Attack score and threshold-based classification.
- Comparison between transformer and classical ML models.
- Evaluation metrics from a custom red-team benchmark.

## Architecture

- Python
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- Streamlit
- Hugging Face Hub

## Model Performance

### DeBERTa v1

- Test accuracy: 96.18%
- Test precision: 93.33%
- Test recall: 98.59%
- Test F1: 95.89%
- Red-team attack recall: 93.45%

### TF-IDF v1

- Red-team attack recall: 79.76%
- Red-team benign false-positive rate: 60%

The red-team benchmark is a custom challenge set and should not be
interpreted as a standardized measure of real-world security.

## Limitations

PromptShield is an experimental classifier. It can produce false
positives and false negatives. It is not a replacement for proper
LLM security controls, input isolation, or authorization checks.

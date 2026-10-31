
import os
import torch
import joblib
import streamlit as st

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification


HF_REPO_ID = "sadhaka19745/promptshield-models"

DEBERTA_THRESHOLD = 0.17
TFIDF_THRESHOLD = 0.30

MAX_LENGTH = 256


@st.cache_resource
def download_models():
    token = st.secrets.get("HF_TOKEN", None)

    if not token:
        raise RuntimeError(
            "HF_TOKEN is missing. Configure it in Streamlit Secrets."
        )

    local_dir = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="model",
        token=token
    )

    return local_dir


@st.cache_resource
def load_deberta():
    model_dir = download_models()
    deberta_dir = os.path.join(model_dir, "deberta")

    tokenizer = AutoTokenizer.from_pretrained(deberta_dir)

    model = AutoModelForSequenceClassification.from_pretrained(
        deberta_dir
    )

    model.eval()
    device = torch.device("cpu")
    model.to(device)

    return tokenizer, model, device


@st.cache_resource
def load_tfidf():
    model_dir = download_models()

    model_path = os.path.join(
        model_dir,
        "tfidf_logistic_regression.joblib"
    )

    return joblib.load(model_path)


def predict_deberta(text):
    tokenizer, model, device = load_deberta()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    attack_score = probabilities[1].item()
    prediction = "Attack" if attack_score >= DEBERTA_THRESHOLD else "Benign"

    return {
        "prediction": prediction,
        "attack_score": attack_score,
        "threshold": DEBERTA_THRESHOLD
    }


def predict_tfidf(text):
    model = load_tfidf()

    if not hasattr(model, "predict_proba"):
        raise RuntimeError(
            "The saved TF-IDF artifact does not expose predict_proba. "
            "Check whether the complete sklearn Pipeline was saved."
        )

    if hasattr(model, "classes_"):
        classes = list(model.classes_)
    elif hasattr(model, "named_steps"):
        classes = list(model.classes_)
    else:
        raise RuntimeError("Cannot identify TF-IDF model classes.")

    if 1 not in classes:
        raise RuntimeError("Attack label 1 is missing from TF-IDF classes.")

    attack_index = classes.index(1)

    attack_score = float(model.predict_proba([text])[0][attack_index])

    prediction = (
        "Attack"
        if attack_score >= TFIDF_THRESHOLD
        else "Benign"
    )

    return {
        "prediction": prediction,
        "attack_score": attack_score,
        "threshold": TFIDF_THRESHOLD
    }
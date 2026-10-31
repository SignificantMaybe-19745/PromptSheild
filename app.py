
import streamlit as st
import pandas as pd

from detector import predict_deberta, predict_tfidf


st.set_page_config(
    page_title="PromptShield",
    page_icon="🛡️",
    layout="wide"
)

# ---------- Styling ----------

st.markdown("""
<style>
    .main {
        background-color: #0b1020;
    }
    .block-container {
        padding-top: 2rem;
        max-width: 1150px;
    }
    .hero {
        padding: 1.5rem;
        border: 1px solid #28344d;
        border-radius: 16px;
        background: linear-gradient(120deg, #151e35, #101827);
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        margin-bottom: 0.3rem;
    }
    .muted {
        color: #9caac0;
    }
    div[data-testid="stMetric"] {
        background: #151e35;
        border: 1px solid #28344d;
        padding: 15px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ---------- Sidebar ----------

st.sidebar.title("🛡️ PromptShield")
st.sidebar.caption("Prompt Injection Detection System")

page = st.sidebar.radio(
    "Navigation",
    ["Prompt Scanner", "Model Comparison", "About"]
)

st.sidebar.divider()
st.sidebar.caption("Primary model: DeBERTa v1")
st.sidebar.caption("Baseline: TF-IDF + Logistic Regression v1")


# ---------- Header ----------

st.markdown("""
<div class="hero">
    <h1>🛡️ PromptShield</h1>
    <p class="muted">
        Detect malicious prompt injection attempts using machine learning.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------- Scanner ----------

if page == "Prompt Scanner":

    st.subheader("Prompt Scanner")

    st.write(
        "Enter a prompt to analyse it for potential prompt injection."
    )

    prompt = st.text_area(
        "Input text",
        height=190,
        placeholder="Paste a prompt here..."
    )

    if st.button("Analyse Prompt", type="primary", use_container_width=True):

        if not prompt.strip():
            st.warning("Please enter some text first.")

        else:
            try:
                with st.spinner("Analysing with DeBERTa v1..."):
                    result = predict_deberta(prompt)

                st.divider()
                st.subheader("Detection Result")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Classification",
                        result["prediction"]
                    )

                with col2:
                    st.metric(
                        "Attack score",
                        f'{result["attack_score"] * 100:.2f}%'
                    )

                st.progress(
                    min(max(result["attack_score"], 0.0), 1.0)
                )

                st.caption(
                    f'Decision threshold: {result["threshold"]:.2f}'
                )

                if result["prediction"] == "Attack":
                    st.error(
                        "Potential prompt injection detected. "
                        "Treat this input as suspicious."
                    )
                else:
                    st.success(
                        "No prompt injection detected by the model."
                    )

                st.info(
                    "This is a machine-learning prediction, not a "
                    "guarantee that the input is safe."
                )

                with st.expander("Compare with TF-IDF baseline"):
                    with st.spinner("Running baseline model..."):
                        baseline = predict_tfidf(prompt)

                    st.write(
                        f'**Prediction:** {baseline["prediction"]}'
                    )
                    st.write(
                        f'**Attack score:** '
                        f'{baseline["attack_score"] * 100:.2f}%'
                    )
                    st.caption(
                        f'Threshold: {baseline["threshold"]:.2f}'
                    )

            except Exception as e:
                st.error(f"Prediction failed: {e}")


# ---------- Model Comparison ----------

elif page == "Model Comparison":

    st.subheader("Model Comparison")

    st.write(
        "Performance comparison using the held-out test set "
        "and custom red-team benchmark."
    )

    test_data = pd.DataFrame({
        "Model": ["TF-IDF v1", "DeBERTa v1"],
        "Test F1": [0.8608, 0.9589],
        "Test Recall": [0.8310, 0.9859],
        "Red-team Attack Recall": [0.7976, 0.9345],
        "Red-team Benign FPR": [0.6000, 0.2923]
    })

    st.dataframe(
        test_data.style.format({
            "Test F1": "{:.2%}",
            "Test Recall": "{:.2%}",
            "Red-team Attack Recall": "{:.2%}",
            "Red-team Benign FPR": "{:.2%}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Test F1 comparison")
    st.bar_chart(
        test_data.set_index("Model")[["Test F1"]]
    )

    st.subheader("Red-team benchmark")
    st.bar_chart(
        test_data.set_index("Model")[
            ["Red-team Attack Recall", "Red-team Benign FPR"]
        ]
    )

    st.caption(
        "The red-team benchmark is a custom challenge set, "
        "not an independent standardized benchmark."
    )


# ---------- About ----------

elif page == "About":

    st.subheader("About PromptShield")

    st.markdown("""
    PromptShield is a machine-learning-based prompt injection
    detection system.

    **Models**
    - DeBERTa v1: primary detection model.
    - TF-IDF + Logistic Regression v1: classical ML baseline.

    **Dataset**
    - Combined prompt injection datasets.
    - Custom red-team benchmark covering 17 attack categories.

    **Detection approach**
    - Text tokenization and transformer classification.
    - TF-IDF feature extraction and logistic regression.
    - Threshold-based binary classification.

    **Limitations**
    - Detection is not guaranteed.
    - Long inputs may be truncated at 256 tokens.
    - False positives and false negatives are possible.
    - This tool should not be used as the sole security boundary
      for an LLM application.
    """)

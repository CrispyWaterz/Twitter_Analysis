# app.py — Twitter Sentiment Analyser with CRISP-DM Documentation

import os
import re
import sys

import joblib
import nltk
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_SEQUENCE_LENGTH = 100
SENTIMENT_LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}
SENTIMENT_EMOJI  = {0: "➖", 1: "🟰", 2: "➕"}
SENTIMENT_COLOR  = {0: "#e74c3c", 1: "#f39c12", 2: "#2ecc71"}

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(BASE_DIR, "model_weights.npy")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer_labeled.joblib")

st.set_page_config(page_title="Twitter Sentiment Analyser", page_icon="💬", layout="centered")

@st.cache_resource(show_spinner="Downloading NLTK data...")
def load_nltk():
    for pkg in ("stopwords", "wordnet", "punkt", "punkt_tab"):
        nltk.download(pkg, quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    return set(stopwords.words("english")), WordNetLemmatizer()

stop_words, lemmatizer = load_nltk()

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens
               if w not in stop_words and len(w) > 1]
    return " ".join(tokens)

@st.cache_resource(show_spinner="Loading model and tokenizer...")
def load_model_and_tokenizer():
    errors = []
    for path, label in [(MODEL_PATH, "model_weights.npy"),
                        (TOKENIZER_PATH, "tokenizer_labeled.joblib")]:
        if not os.path.exists(path):
            errors.append(f"❌ `{label}` not found. Files: `{os.listdir(BASE_DIR)}`")
    if errors:
        return None, None, errors
    try:
        weights = np.load(MODEL_PATH, allow_pickle=True)
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=25430, output_dim=100,
                                      input_length=100, name="embedding_layer"),
            tf.keras.layers.Conv1D(64, 5, activation="relu", name="conv1d_layer"),
            tf.keras.layers.Dropout(0.2, name="dropout_cnn"),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(96), name="bidirectional_lstm_layer"),
            tf.keras.layers.Dropout(0.3, name="dropout_lstm"),
            tf.keras.layers.Dense(3, activation="softmax", name="output_layer"),
        ])
        model.build(input_shape=(None, 100))
        model.set_weights(weights)
    except Exception as exc:
        return None, None, [f"❌ Model failed to load: `{exc}`"]
    try:
        tokenizer = joblib.load(TOKENIZER_PATH)
    except Exception as exc:
        return None, None, [f"❌ Tokenizer failed to load: `{exc}`"]
    return model, tokenizer, []

def predict(texts, model, tokenizer):
    processed = [preprocess_text(t) for t in texts]
    sequences = tokenizer.texts_to_sequences(processed)
    padded    = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH,
                              padding="post", truncating="post")
    probs     = model.predict(padded, verbose=0)
    results   = []
    for i, prob_row in enumerate(probs):
        label_id   = int(np.argmax(prob_row))
        confidence = float(prob_row[label_id]) * 100
        results.append({
            "original_text":  texts[i],
            "processed_text": processed[i],
            "sentiment":      SENTIMENT_LABELS[label_id],
            "emoji":          SENTIMENT_EMOJI[label_id],
            "confidence":     confidence,
            "label_id":       label_id,
        })
    return results

# ── Load ──────────────────────────────────────────────────────────────────────
model, tokenizer, load_errors = load_model_and_tokenizer()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("Twitter Sentiment Analyser")
st.caption("Powered by a pre-trained LSTM-CNN hybrid model.")

if load_errors:
    for err in load_errors:
        st.error(err)
    st.stop()

tab_single, tab_batch, tab_crisp = st.tabs(["Single Text", "Batch CSV", "CRISP-DM"])

# ── Tab 1: Single Text ────────────────────────────────────────────────────────
with tab_single:
    st.subheader("Predict sentiment for a single text")
    user_text = st.text_area("Enter text:", height=120,
                             placeholder="e.g. I love this product, it works perfectly!")
    if st.button("Analyse", type="primary"):
        if not user_text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Analysing..."):
                results = predict([user_text], model, tokenizer)
            r = results[0]
            if not r["processed_text"]:
                st.warning("Text was empty after preprocessing.")
            else:
                color = SENTIMENT_COLOR[r["label_id"]]
                st.markdown(f"<h2 style='color:{color}'>{r['emoji']} {r['sentiment']}</h2>",
                            unsafe_allow_html=True)
                st.metric("Confidence", f"{r['confidence']:.1f}%")
                with st.expander("Processed text"):
                    st.code(r["processed_text"])

# ── Tab 2: Batch CSV ──────────────────────────────────────────────────────────
with tab_batch:
    st.subheader("Predict sentiment for a CSV file")
    st.info("Upload a CSV with a column named **`full_text`**.")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            st.stop()
        st.dataframe(df.head())
        if "full_text" not in df.columns:
            st.error(f"No `full_text` column found. Columns: `{list(df.columns)}`")
        else:
            df["full_text"] = df["full_text"].astype(str)
            df_valid = df[df["full_text"].str.strip().str.lower() != "nan"].copy()
            if df_valid.empty:
                st.warning("No valid rows found.")
            else:
                with st.spinner(f"Analysing {len(df_valid):,} rows..."):
                    results = predict(df_valid["full_text"].tolist(), model, tokenizer)
                df_valid["processed_text"]     = [r["processed_text"] for r in results]
                df_valid["predicted_sentiment"] = [r["sentiment"]      for r in results]
                df_valid["confidence_%"]        = [round(r["confidence"], 2) for r in results]
                st.success(f"Done! Predicted {len(df_valid):,} rows.")
                st.bar_chart(df_valid["predicted_sentiment"].value_counts())
                st.dataframe(df_valid[["full_text", "predicted_sentiment", "confidence_%"]].head(20))
                st.download_button("Download results as CSV",
                                   data=df_valid.to_csv(index=False).encode("utf-8"),
                                   file_name="predicted_sentiments.csv", mime="text/csv")

# ── Tab 3: CRISP-DM ───────────────────────────────────────────────────────────
with tab_crisp:
    st.header("📋 CRISP-DM Methodology")
    st.caption("Cross-Industry Standard Process for Data Mining — step-by-step documentation of this project.")

    # Step 1
    with st.expander("1️⃣ Business Understanding", expanded=True):
        st.markdown("""
**Objective**
Understand public sentiment toward the *blue checkmark* policy and *Community Notes* feature on platform X (formerly Twitter).

**Research Questions**
- What is the overall sentiment distribution of tweets about the blue checkmark and Community Notes?
- How does public opinion shift over time?
- Can a machine learning model accurately classify tweet sentiment into Negative, Neutral, or Positive?

**Success Criteria**
A deployed sentiment analysis model with acceptable accuracy, accessible via a web interface for real-time and batch prediction.
        """)

    # Step 2
    with st.expander("2️⃣ Data Understanding"):
        st.markdown("""
**Data Sources**

| Source | Description | Size |
|---|---|---|
| `twitter_training.csv` (Kaggle) | Pre-labeled tweets — Negative, Neutral, Positive, Irrelevant | ~74,000 rows |
| `Scrape (1–14).csv` | Tweets scraped from platform X using keyword *"blue checkmark"* | ~12,195 rows (10,084 after cleaning) |

**Key Fields**
- `full_text` — raw tweet content
- `sentiment` — ground truth label (labeled data only)
- `created_at` — tweet timestamp
- `lang` — language of the tweet (filtered to English only)

**Initial Observations**
- 94 duplicate rows removed from scraped data
- 2,018 non-English tweets filtered out
- `location` and `username` columns dropped (not relevant to analysis)
        """)

    # Step 3
    with st.expander("3️⃣ Data Preparation"):
        st.markdown("""
**Text Preprocessing Pipeline**

Applied to both labeled training data and scraped tweets:

1. **Lowercasing** — convert all text to lowercase
2. **URL removal** — strip `http://`, `https://`, `www.` links
3. **Mention removal** — remove `@username` patterns
4. **Hashtag removal** — remove `#hashtag` patterns
5. **Special character removal** — keep only letters and spaces
6. **Tokenization** — split text into individual words
7. **Stopword removal** — remove common English stopwords (NLTK)
8. **Lemmatization** — reduce words to their base form (WordNetLemmatizer)

**Tokenization & Padding**
- Tokenizer fitted on labeled training data (`Tokenizer(num_words=20000, oov_token="<unk>")`)
- Sequences padded/truncated to fixed length of **100 tokens**
- Same tokenizer applied to scraped data for consistency

**Label Mapping**

| Original Label | Numeric Label |
|---|---|
| Negative | 0 |
| Neutral / Irrelevant / null | 1 |
| Positive | 2 |

**Train/Test Split** — 80% training, 20% testing (`random_state=42`, stratified)
        """)

    # Step 4
    with st.expander("4️⃣ Modeling"):
        st.markdown("""
**Architecture — LSTM-CNN Hybrid Model**

| Layer | Configuration |
|---|---|
| Embedding | `input_dim=25,430`, `output_dim=100`, `input_length=100` |
| Conv1D | `filters=64`, `kernel_size=5`, `activation=relu` |
| Dropout (CNN) | `rate=0.4` |
| Bidirectional LSTM | `units=96`, `return_sequences=False` |
| Dropout (LSTM) | `rate=0.5` |
| Dense (Output) | `units=3`, `activation=softmax` |

**Rationale**
- CNN layers extract local n-gram features from tweet text
- Bidirectional LSTM captures sequential dependencies in both directions
- Combined architecture leverages strengths of both approaches

**Hyperparameter Tuning — Keras Tuner (RandomSearch)**

| Hyperparameter | Search Space | Optimal Value |
|---|---|---|
| Conv1D filters | 64, 128, 192, 256 | **64** |
| Bidirectional LSTM units | 32, 64, 96, 128 | **96** |
| Dropout rate (CNN) | 0.2 – 0.5 | **0.4** |
| Dropout rate (LSTM) | 0.3 – 0.6 | **0.5** |
| Learning rate (Adam) | 0.01, 0.001, 0.0001 | **0.001** |

- `max_trials=5`, `executions_per_trial=1`
- Objective: maximize `val_accuracy`
- Training: `epochs=10`, `batch_size=32`, `validation_split=0.2`
        """)

    # Step 5
    with st.expander("5️⃣ Evaluation"):
        st.markdown("""
**Evaluation Metrics**
- **Loss function:** Sparse Categorical Crossentropy
- **Primary metric:** Accuracy on held-out test set (20% of labeled data)
- **Confusion matrix** generated to assess per-class performance across Negative, Neutral, and Positive classes

**Model Performance**
The best model from Keras Tuner hyperparameter search was selected based on validation accuracy and evaluated on the test set.
        """)

    # Step 6
    with st.expander("6️⃣ Deployment"):
        st.markdown("""
**Deployment Stack**

| Component | Technology |
|---|---|
| Web framework | Streamlit |
| Hosting | Streamlit Cloud |
| Model weights | NumPy `.npy` format (version-independent) |
| Tokenizer | Joblib `.joblib` format |
| Source control | GitHub (`CrispyWaterz/Twitter_Analysis`) |

**Features**
- **Single Text** tab — real-time sentiment prediction for any input text
- **Batch CSV** tab — upload a CSV with a `full_text` column for bulk prediction with downloadable results
- **CRISP-DM** tab — this documentation page

**Preprocessing at Inference**
Identical pipeline to training: lowercase → remove URLs/mentions/hashtags → remove special chars → tokenize → remove stopwords → lemmatize → pad to 100 tokens.
        """)

st.divider()
st.caption(f"TF {tf.__version__} | Python {sys.version.split()[0]}")

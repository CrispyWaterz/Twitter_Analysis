# app.py — Production-ready Streamlit Sentiment Analysis App
# Model and tokenizer are committed directly to the repo.

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
SENTIMENT_EMOJI  = {0: "😠", 1: "😐", 2: "😊"}
SENTIMENT_COLOR  = {0: "#e74c3c", 1: "#f39c12", 2: "#2ecc71"}

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(BASE_DIR, "hybrid_sentiment_model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer_labeled.joblib")

st.set_page_config(page_title="Sentiment Analyser", page_icon="🐦", layout="centered")

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
    for path, label in [(MODEL_PATH, "hybrid_sentiment_model.h5"),
                        (TOKENIZER_PATH, "tokenizer_labeled.joblib")]:
        if not os.path.exists(path):
            errors.append(f"❌ `{label}` not found. Files present: `{os.listdir(BASE_DIR)}`")
    if errors:
        return None, None, errors
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
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

model, tokenizer, load_errors = load_model_and_tokenizer()

st.title("Twitter Sentiment Analyser")
st.caption("Powered by a pre-trained LSTM-CNN hybrid model.")

if load_errors:
    for err in load_errors:
        st.error(err)
    st.stop()

tab_single, tab_batch = st.tabs(["Single Text", "Batch CSV"])

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

st.divider()
st.caption(f"TF {tf.__version__} | Python {sys.version.split()[0]}")

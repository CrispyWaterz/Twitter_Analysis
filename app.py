# app.py — Production-ready Streamlit Sentiment Analysis App
# Model and tokenizer are downloaded from Google Drive on first startup
# and cached locally so subsequent restarts skip the download.
#
# To update the files, change the FILE_IDS below and redeploy.

import os
import re
import sys

import gdown
import joblib
import nltk
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── Constants (must match values used during training) ──────────────────────
MAX_SEQUENCE_LENGTH = 100
SENTIMENT_LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}
SENTIMENT_EMOJI  = {0: "😠", 1: "😐", 2: "😊"}
SENTIMENT_COLOR  = {0: "#e74c3c", 1: "#f39c12", 2: "#2ecc71"}

# ── Google Drive file IDs ────────────────────────────────────────────────────
# Share each file in Drive with "Anyone with the link → Viewer" or it will
# return an HTML permission-error page instead of the actual file.
MODEL_GDRIVE_ID     = "1Ix6TNefsFk0f31JfpjeNrsBCx_ffu9YK"
TOKENIZER_GDRIVE_ID = "1GDukP-GtIJIrX5BShSolAz2xYRqtypEI"

# Local paths where the files are cached after the first download.
# Streamlit Cloud gives each deployment ephemeral disk space; the cache
# persists for the lifetime of the running container (~hours / until restart).
CACHE_DIR      = "/tmp/sentiment_model_cache"
MODEL_PATH     = os.path.join(CACHE_DIR, "hybrid_sentiment_model.keras")
TOKENIZER_PATH = os.path.join(CACHE_DIR, "tokenizer_labeled.joblib")

os.makedirs(CACHE_DIR, exist_ok=True)


def download_from_gdrive(file_id: str, dest_path: str, label: str) -> list[str]:
    """Download a file from Google Drive using gdown. Returns a list of errors."""
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, dest_path, quiet=False, fuzzy=True)
        if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
            return [
                f"❌ Downloaded `{label}` appears to be empty. "
                "Make sure the file is shared as **Anyone with the link → Viewer** in Google Drive."
            ]
    except Exception as exc:
        return [f"❌ Failed to download `{label}` from Google Drive: `{exc}`"]
    return []

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analyser",
    page_icon="🐦",
    layout="centered",
)

# ── NLTK setup (cached so it only runs once per session) ────────────────────
@st.cache_resource(show_spinner="Downloading NLTK data…")
def load_nltk():
    for pkg in ("stopwords", "wordnet", "punkt", "punkt_tab"):
        nltk.download(pkg, quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    return set(stopwords.words("english")), WordNetLemmatizer()

stop_words, lemmatizer = load_nltk()

# ── Text preprocessing (identical to training pipeline) ─────────────────────
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w not in stop_words and len(w) > 1
    ]
    return " ".join(tokens)

# ── Model & tokenizer loader (cached) ───────────────────────────────────────
@st.cache_resource(show_spinner="Downloading & loading model… (first run only, may take a minute)")
def load_model_and_tokenizer():
    errors = []

    # Download model if not already cached
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        errors += download_from_gdrive(MODEL_GDRIVE_ID, MODEL_PATH, "model")

    # Download tokenizer if not already cached
    if not os.path.exists(TOKENIZER_PATH) or os.path.getsize(TOKENIZER_PATH) == 0:
        errors += download_from_gdrive(TOKENIZER_GDRIVE_ID, TOKENIZER_PATH, "tokenizer")

    if errors:
        return None, None, errors

    # Load model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as exc:
        return None, None, [
            f"❌ Model downloaded but failed to load: `{exc}`\n\n"
            "This usually means the `.keras` file is corrupted or the wrong format. "
            "Re-export it with `model.save('hybrid_sentiment_model.keras')` in Colab."
        ]

    # Load tokenizer
    try:
        tokenizer = joblib.load(TOKENIZER_PATH)
    except Exception as exc:
        return None, None, [
            f"❌ Tokenizer downloaded but failed to load: `{exc}`\n\n"
            "Re-export it with `joblib.dump(tokenizer_labeled, 'tokenizer_labeled.joblib')` in Colab."
        ]

    return model, tokenizer, []

# ── Prediction helper ────────────────────────────────────────────────────────
def predict(texts: list[str], model, tokenizer) -> list[dict]:
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

# ── Load resources ───────────────────────────────────────────────────────────
model, tokenizer, load_errors = load_model_and_tokenizer()

# ── UI ───────────────────────────────────────────────────────────────────────
st.title("🧠 Sentiment Analyser")
st.caption("Powered by a pre-trained LSTM-CNN hybrid model.")

if load_errors:
    for err in load_errors:
        st.error(err)
    st.info(
        "**Troubleshooting checklist:**\n"
        "1. Open each file in Google Drive → Share → change to **Anyone with the link → Viewer**.\n"
        "2. Confirm the File IDs in `app.py` match the URLs of your Drive files.\n"
        "3. Make sure `gdown` is in your `requirements.txt`.\n"
        "4. Check the Streamlit Cloud logs for the full traceback."
    )
    st.stop()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_single, tab_batch = st.tabs(["✏️ Single Text", "📄 Batch CSV"])

# ── Tab 1 · Single prediction ────────────────────────────────────────────────
with tab_single:
    st.subheader("Predict sentiment for a single text")
    user_text = st.text_area(
        "Enter text:",
        placeholder="e.g. I love this product, it works perfectly!",
        height=120,
    )

    if st.button("Analyse", type="primary"):
        if not user_text.strip():
            st.warning("Please enter some text before clicking Analyse.")
        else:
            with st.spinner("Analysing…"):
                results = predict([user_text], model, tokenizer)
            r = results[0]

            if not r["processed_text"]:
                st.warning(
                    "The text was empty after preprocessing (only stopwords / "
                    "symbols). Try a longer, more descriptive sentence."
                )
            else:
                color = SENTIMENT_COLOR[r["label_id"]]
                st.markdown(
                    f"<h2 style='color:{color}'>"
                    f"{r['emoji']} {r['sentiment']}"
                    f"</h2>",
                    unsafe_allow_html=True,
                )
                st.metric("Confidence", f"{r['confidence']:.1f}%")
                with st.expander("Processed text (after cleaning)"):
                    st.code(r["processed_text"])

# ── Tab 2 · Batch prediction ─────────────────────────────────────────────────
with tab_batch:
    st.subheader("Predict sentiment for a CSV file")
    st.info(
        "Upload a CSV that contains a column named **`full_text`**. "
        "Results will be available for download."
    )

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")
            st.stop()

        st.write("**Preview (first 5 rows):**")
        st.dataframe(df.head())

        if "full_text" not in df.columns:
            st.error(
                "The uploaded CSV must contain a column named **`full_text`**. "
                f"Columns found: `{list(df.columns)}`"
            )
        else:
            df = df.copy()
            df["full_text"] = df["full_text"].astype(str)

            # Drop rows where full_text is NaN / empty
            mask_valid = df["full_text"].str.strip().str.lower() != "nan"
            df_valid   = df[mask_valid].copy()

            if df_valid.empty:
                st.warning("No valid rows found in `full_text` column.")
            else:
                with st.spinner(f"Analysing {len(df_valid):,} rows…"):
                    results = predict(df_valid["full_text"].tolist(), model, tokenizer)

                df_valid["processed_text"]      = [r["processed_text"] for r in results]
                df_valid["predicted_sentiment"]  = [r["sentiment"]      for r in results]
                df_valid["confidence_%"]         = [round(r["confidence"], 2) for r in results]

                st.success(f"Done! Predicted {len(df_valid):,} rows.")

                # Distribution chart
                counts = df_valid["predicted_sentiment"].value_counts()
                st.subheader("Sentiment distribution")
                st.bar_chart(counts)

                # Preview table
                st.subheader("Results preview (first 20 rows)")
                st.dataframe(
                    df_valid[["full_text", "predicted_sentiment", "confidence_%"]].head(20)
                )

                # Download
                csv_bytes = df_valid.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download full results as CSV",
                    data=csv_bytes,
                    file_name="predicted_sentiments.csv",
                    mime="text/csv",
                )

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"Model: `{MODEL_PATH}` · Tokenizer: `{TOKENIZER_PATH}` · "
    f"TensorFlow {tf.__version__} · Python {sys.version.split()[0]}"
)

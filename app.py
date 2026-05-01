import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import joblib

# ==============================
# CONFIG
# ==============================
MAX_SEQUENCE_LENGTH = 100

# ==============================
# LOAD NLTK
# ==============================
@st.cache_resource
def load_nltk():
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    return set(stopwords.words('english')), WordNetLemmatizer()

stop_words, lemmatizer = load_nltk()

# ==============================
# PREPROCESS
# ==============================
def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    tokens = nltk.word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w not in stop_words and len(w) > 1
    ]

    return " ".join(tokens)

# ==============================
# LOAD MODEL + TOKENIZER
# ==============================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("hybrid_sentiment_model.keras")
    tokenizer = joblib.load("tokenizer_labeled.joblib")
    return model, tokenizer

model, tokenizer = load_model()

label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# ==============================
# UI
# ==============================
st.title("Sentiment Analysis (LSTM-CNN)")
st.write("Predict sentiment from text")

# ==============================
# SINGLE INPUT
# ==============================
text = st.text_area("Input text:")

if st.button("Predict"):
    if text:
        processed = preprocess_text(text)

        seq = tokenizer.texts_to_sequences([processed])
        padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)

        pred = model.predict(padded)
        label = np.argmax(pred)
        confidence = pred[0][label] * 100

        st.success(f"{label_map[label]} ({confidence:.2f}%)")
        st.write("Processed:", processed)
    else:
        st.warning("Masukkan teks dulu")

# ==============================
# CSV UPLOAD
# ==============================
st.subheader("Batch Prediction")

file = st.file_uploader("Upload CSV (must have 'full_text')")

if file:
    df = pd.read_csv(file)

    if "full_text" not in df.columns:
        st.error("CSV harus punya kolom 'full_text'")
    else:
        df["processed"] = df["full_text"].apply(preprocess_text)

        df = df[df["processed"] != ""]

        seq = tokenizer.texts_to_sequences(df["processed"])
        padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)

        pred = model.predict(padded)
        df["sentiment"] = np.argmax(pred, axis=1)
        df["sentiment"] = df["sentiment"].map(label_map)

        st.dataframe(df.head())

        st.bar_chart(df["sentiment"].value_counts())

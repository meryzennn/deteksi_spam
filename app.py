import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# Load and preprocess data
data = pd.read_csv("data_spam.csv", encoding="latin-1")
data = data[["class", "message"]]

# Split data and train model
x = np.array(data["message"])
y = np.array(data["class"])
cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)

detector = MultinomialNB()
detector.fit(X_train, y_train)

# Custom CSS for enhanced styling with abstract background
st.markdown("""
    <style>
        /* Abstract Background */
        .stApp {
            background: linear-gradient(120deg, #e0c3fc 0%, #8ec5fc 100%);
            background-size: cover;
            padding: 20px;
        }

        /* Main Title */
        .main-title {
            font-size: 3em;
            color: #000000;
            font-weight: bold;
            text-align: center;
            margin-bottom: 10px;
        }

        /* Instructions */
        .instructions {
            font-size: 1.2em;
            color: #fff;
            text-align: center;
            margin-bottom: 20px;
        }

        /* Prediction Text */
        .prediction-text {
            font-size: 1.5em;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
            margin-top: 20px;
            padding: 10px;
            border-radius: 10px;
            background-color: #e6ffed;
        }

        /* Warning Text */
        .warning {
            font-size: 1.5em;
            font-weight: bold;
            color: #FF6347;
            text-align: center;
            margin-top: 20px;
            padding: 10px;
            border-radius: 10px;
            background-color: #ffe6e6;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit app with enhanced UI
st.markdown("<div class='main-title'>🚨 Instagram Spam Detection 🚨</div>", unsafe_allow_html=True)
st.write("<div class='instructions'>Enter text below to detect if it's spam or not</div>", unsafe_allow_html=True)

input_text = st.text_input("Enter Text:")

if input_text:
    data = cv.transform([input_text]).toarray()
    prediction = detector.predict(data)

    if prediction[0] == 'spam':
        st.markdown("<div class='warning'>⚠️ Detected as Spam</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='prediction-text'>✅ Not Spam</div>", unsafe_allow_html=True) 
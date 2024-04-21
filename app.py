import streamlit as st
import pickle
import joblib
from sklearn.feature_extraction.text import CountVectorizer
st.title("Airline Sentiment Analysis")
st.image(r"C:\Users\ravin\Downloads\Innomatics-Logo1 (1).webp")
if st.button("Text Analysis"):
    st.image(r"C:\Users\ravin\Downloads\wordcloud.png")
if st.button("Sentiment analysis"):
    st.image(r"C:\Users\ravin\OneDrive\Pictures\Screenshots\Screenshot 2024-04-21 193049.png")

text = st.text_input("Enter text...")
if st.button("Predict Sentiment"):
    vectorizer = joblib.load(r"D:\innomatics\ML\vecsent.pkl")
    text_vectorized = vectorizer.transform([text])
    model = joblib.load(r"D:\innomatics\ML\rf2.pkl")
    prediction = model.predict(text_vectorized)[0]
    st.title(prediction)

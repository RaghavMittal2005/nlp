import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

word_index=imdb.get_word_index()
reverse_index = {value: key for key, value in word_index.items()}

model=load_model('simple_rnn_imdb.h5')

def decode_review(sent):
    decoded_review = " ".join([reverse_index.get(i-3, "?") for i in sent])
    return decoded_review
def preprocess(text):
    text = text.lower().split()
    encoded=[word_index.get(word,2)+3 for word in text]
    pad=sequence.pad_sequences([encoded],maxlen=500)
    return pad

def predict_sentiment(review):
    preproc=preprocess(review)
    pred=model.predict(preproc)
    sentiment='Positive' if pred[0][0]>0.5 else 'Negative'
    return sentiment,pred[0][0]

st.title("IMDB Review Analysis")
st.write("Enter a review to classify")

user_input=st.text_area('Movie Review')

if st.button('Classify'):
    sentiment,pred=predict_sentiment(user_input)
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Confidence: {pred:.2f}")
else:
    st.write("Please click the button to classify the review")
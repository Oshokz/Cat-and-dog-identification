# import required packages
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle # to load model and tokenizer
import numpy as np

#load the trained model
model = load_model(r"C:\Users\User\streamlit_sample\nlp_model.h5")

# load the saved tokenizer used during traning
with open(r"C:\Users\User\streamlit_sample\tokenizer.pkl", "rb") as tk:
          tokenizer = pickle.load(tk)
# rb because it was saved as binary mode

#Define the function to preprocess the user text input 
def preprocess_text(text):
    #Tokenize the text
    tokens = tokenizer.texts_to_sequences([text])

    #pad the sequences to a fixed length:
    padded_tokens = pad_sequences(tokens, maxlen = 100) 
    return padded_tokens[0]

#create the title of the app
st.title("Sentiment Analysis App")

#Create a text input widget for user input
user_input = st.text_area("Enter text for sentiment analysis", " ")

# create a button to trigger the sentiment analysis
if st.button("Predict Sentiment"):
    # preprocess the user input
    processed_input = preprocess_text(user_input)
    
    # Make prediction using the loaded model
    prediction = model.predict(np.array([processed_input]))
    st.write(prediction)
    sentiment = "Negative" if prediction[0][0] > 0.5 else "Positive"
    
    # Display the sentiment
    st.write(f" ### Sentiment: {sentiment}")




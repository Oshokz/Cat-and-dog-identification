# importing required packages
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Create the title for the App
st.title("Dog-Cat Image Classification")
st.write("Upload an image of a cat or a dog, and we'll predict which is it")

# Create a file uploader
uploaded_file = st.file_uploader("Upload an image..", type = ["jpg", "jpeg", "png"])

#check if the image is uploaded
if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")
    st.write("")

    # Preprocess the image
    img = np.array(image)
    # Resize the image
    img = tf.image.resize(img, (64, 64))
    # Normalize the image
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Change the batch dimension to (1, 64, 64, 3) for 1 image
    st.write(f"Image shape: {img.shape}")

    # Load the trained model (for classifying "Cat" or "Dog")
    model = load_model(r"C:\Users\User\streamlit_sample\vgg_model.h5")

    # Make predictions
    prediction = model.predict(img)
    label = "Cat" if prediction[0][0] > 0.5 else "Dog"
    
    # Display the prediction
    st.write(f"## Prediction: {label}")
    

      



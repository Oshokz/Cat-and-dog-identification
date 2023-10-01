# importing required packages
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Create the title for the App
st.title("Covid X-ray Prediction")
st.write("### Examine your X-ray image here to determine if it indicates the presence of COVID-19.")

# Create a file uploader
uploaded_file = st.file_uploader("Upload an image..", type = ["jpg", "jpeg", "png"])

#check if the image is uploaded
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")
    st.write("")

    # Preprocess the image
    img = np.array(image)
    # Resize the image
    img = tf.image.resize(img, (64, 64))
    # Normalize the image
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Change the batch dimension to (1, 64, 64, 3)  

    # Load the trained model
    model = load_model(r"C:\Users\User\streamlit_sample\covid_vgg_model.h5")

    # Make predictions
    prediction = model.predict(img)
    
    # Determine the label based on numerical thresholds
    class_index = np.argmax(prediction)
    if class_index == 0:
        label = "Covid"
    elif class_index == 1:
        label = "Normal"
    else:
        label = "Viral Pneumonia"
    
    st.write(f"Prediction: {label}")

    # Display the prediction
    st.write(f"## Predicted Image: {label}")

    









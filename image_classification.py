import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# Load model saved in native Keras format
model = tf.keras.models.load_model("fashion_mnist_cnn_model.keras")

# Class labels for Fashion MNIST dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# App title and instructions
st.title("👗 Fashion Item Classifier")
st.write("Upload an image of a fashion item (28x28 pixels or larger), and the model will predict its class.")

# Image uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open uploaded image and convert to grayscale
    image = Image.open(uploaded_file).convert("L")

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize image to 28x28 using Lanczos resampling
    image_resized = ImageOps.fit(image, (28, 28), Image.Resampling.LANCZOS)

    # Convert image to numpy array and normalize pixel values
    img_array = np.array(image_resized) / 255.0

    # Reshape array to match model input shape: (1, 28, 28, 1)
    img_array = img_array.reshape(1, 28, 28, 1)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)

    # Show the prediction result
    st.success(f"Prediction: **{class_names[predicted_label]}**")

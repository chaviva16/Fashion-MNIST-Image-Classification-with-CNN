# 👗 Fashion MNIST Image Classifier

This project is a **Convolutional Neural Network (CNN)** based image classification system trained on the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).
It uses **TensorFlow/Keras** for model training and **Streamlit** to deploy an interactive web app that predicts the type of fashion item from an uploaded image.

---

## 🚀 Features

- 📷 Upload grayscale fashion images
- 🧠 CNN model trained on Fashion MNIST
- 🖼️ Real-time predictions via Streamlit
- 🎯 Achieves over 90% accuracy on test data

---

## 🧠 Model Architecture

- `Conv2D(32)` → `MaxPooling2D`
- `Conv2D(64)` → `MaxPooling2D`
- `Flatten` → `Dense(128)` → `Dense(10)`

Trained on 60,000 training images, validated on 10,000 test images.

---

## 🏷️ Class Labels

['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



---

## 🛠️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/chaviva16/Fashion-MNIST-Image-Classification-with-CNN.git
   cd image_classsification

2.  Create a virtual environment (optional but recommended):
 conda create -n tf-env python=3.9
conda activate tf-env

3. Install dependencies:
pip install -r requirements.txt

4.Launch the Streamlit app:
streamlit run image_classification.py


## 📁 Project Structure
├── fashion_mnist_cnn_model.keras      # Trained CNN model
├── image_classification.py            # Streamlit app script
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies

## 🧪 Sample Predictions
Upload a 28x28 grayscale or color image of a fashion item, and the app will display its predicted class label.

## 📦 Dependencies
tensorflow

streamlit

numpy

Run pip install -r requirements.txt to install everything you need.

## 📌 License
This project is licensed under the MIT License – feel free to use, modify, and share it.

## 🙋‍♀️ Author
Developed by Chaviva

Feel free to connect on GitHub






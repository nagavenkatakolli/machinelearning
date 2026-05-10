from src import data_loader as data_loader
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps, ImageOps
import cv2


def preprocess_data():
    # Load the MNIST dataset
    (x_train , y_train),(x_test , y_test) = data_loader.load_mnist_data()

    # Normalize the pixel values to be between 0 and 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Reshape the data to add a channel dimension (since MNIST images are grayscale)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    return (x_train, y_train), (x_test, y_test)

# Preprocess the uploaded image for prediction

def preprocess_uploaded_image(uploaded_file):

    # Open image and convert to grayscale
    image = Image.open(uploaded_file).convert("L")

    # Auto-adjust contrast slightly
    image = ImageOps.autocontrast(image)

    # Resize to MNIST size
    image = image.resize((28, 28))

    # Convert to numpy array
    image = np.array(image)

    # Invert if background is light
    if image.mean() > 127:
        image = 255 - image

    # Normalize pixel values
    image = image.astype("float32") / 255.0

    # Reshape for CNN input
    image = image.reshape(1, 28, 28, 1)

    return image
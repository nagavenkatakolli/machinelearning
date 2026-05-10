import tensorflow as tf
import numpy as np
import os

# Load the MNIST dataset Avalibale in Keras, The data is already split in to training and test sets
def load_mnist_data():

    return tf.keras.datasets.mnist.load_data()


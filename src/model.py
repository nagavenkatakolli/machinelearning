import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential  


def create_model():

# Define a simple CNN model for MNIST classification
    # This is how the model architecture looks like:
    # Conv2D  kernel_size=(3, 3) activation='relu' 
    # -> MaxPooling2D 
    # -> Conv2D kernel_size=(3, 3) activation='relu'
    # -> MaxPooling2D
    # -> Flatten - > Dense 128 activation='relu'    
    # -> Dense 10 activation='softmax'
    # The input shape is (28, 28, 1) since MNIST images are grayscale and have a size of 28x28 pixels.
    # The output layer has 10 units corresponding to the 10 classes (digits 0-9) and uses softmax activation for multi-class classification.
    # The model is compiled with the Adam optimizer, sparse categorical crossentropy loss (since the labels are integers), and accuracy as a metric.
    # The model architecture is simple and suitable for the MNIST dataset, 
    # which is a common benchmark for image classification tasks. 
    # It should achieve good performance on this dataset after training.

    model = Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.summary()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    
    return model
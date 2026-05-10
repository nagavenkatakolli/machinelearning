import src.data_loader as data_loader
import tensorflow as tf
import src.preprocessing as preprocessing
import src.model as model
import src.utils as utils


def train_model():
    # Load the MNIST dataset
    (x_train , y_train),(x_test , y_test) = data_loader.load_mnist_data()

    # Print all 4 arrays shapes
    print("x_train shape: ", x_train.shape)
    print("y_train shape: ", y_train.shape)
    print("x_test shape: ", x_test.shape)
    print("y_test shape: ", y_test.shape)

    # Normalize the pixel values to be between 0 and 1
    (x_train , y_train),(x_test , y_test) = preprocessing.preprocess_data()

    print("After preprocessing:")
    # Print all 4 arrays shapes
    print("x_train shape: ", x_train.shape)
    print("y_train shape: ", y_train.shape)
    print("x_test shape: ", x_test.shape)   
    print("y_test shape: ", y_test.shape)

    # train the model for 5 epochs and validate on the test set
    cnn_model = model.create_model()
    cnn_model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))     

    # Save the model in the specified path
    utils.save_model(cnn_model, model_dir="model", model_name="mnist_cnn_model.keras"  )
    utils.save_training_plot(cnn_model.history, model_dir="model")
    # Evaluate the model on the test set and print the test accuracy
    test_loss, test_acc = cnn_model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")

   
    

if __name__ == "__main__":
    train_model()
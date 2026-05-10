import os
import matplotlib.pyplot as plt


# Function to create a directory if it does not exist and save the model in the specified path
def save_model(model, model_dir, model_name):
    # Create the directory if it does not exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save the model in the specified path
    model_path = os.path.join(model_dir, model_name)
    model.save(model_path)
    print(f"Model saved at: {model_path}")

# save accuracy and model plots in the same directory
# save accuracy plot
# save model architecture plot

def save_training_plot(history, model_dir="model"):
    PLOT_PATH = os.path.join(model_dir, "training_plot.png")

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()
    
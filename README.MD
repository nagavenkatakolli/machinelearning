MNIST Digit Classification with CNN

A deep learning web application for handwritten digit recognition using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

The application allows users to upload an image of a handwritten digit (0–9) and predicts the digit along with confidence scores for all classes.

Features
CNN model built with TensorFlow/Keras
Trained on the MNIST handwritten digit dataset
Streamlit web application
Upload custom digit images
Real-time prediction
Confidence score display
Probability table and visualization
Image preprocessing pipeline
Project Structure
mnist_project/
│
├── model/
│   └── mnist_cnn_model.keras
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   └── utils.py
│
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
Technologies Used
Python
TensorFlow / Keras
Streamlit
NumPy
Pandas
PIL (Pillow)
CNN Architecture

The model uses a Convolutional Neural Network (CNN) for image classification.

Typical architecture:

Input Layer (28x28x1)
    → Conv2D
    → MaxPooling2D
    → Conv2D
    → MaxPooling2D
    → Flatten
    → Dense
    → Output Layer (10 classes)
Dataset

The model is trained using the MNIST dataset:

60,000 training images
10,000 testing images
Grayscale handwritten digits (0–9)
Image size: 28×28 pixels
Installation

Clone the repository:

git clone https://github.com/nagavenkatakolli/machinelearning.git

cd machinelearning

Create a virtual environment:

    python -m venv .venv
    source .venv/bin/activate

Install dependencies:

    pip install -r requirements.txt
Train the Model

Run:

python train_model.py

The trained model will be saved in:

    model/mnist_cnn_model.keras

Run the Streamlit App

Start the web application:

    streamlit run app.py

Open in browser:

http://localhost:8501
Image Preprocessing

Uploaded images are preprocessed before prediction: Below are the steps how image preprocessed before prediction

    Convert image to grayscale
    Resize to 28×28
    Auto contrast adjustment
    Normalize pixel values
    Reshape for CNN input
    Example Workflow
    Upload a handwritten digit image
    Image is preprocessed
    CNN model predicts the digit
    Application displays:
    Predicted digit
    Confidence score
    Probability scores for all digits
    Limitations

This model is trained only on the MNIST dataset.

Predictions work best when:

    digit is large and centered
    background is plain
    image is tightly cropped
    digit has strong contrast

Performance may decrease for:

    noisy backgrounds
    notebook lines
    shadows
    textured surfaces
    very small digits
    Future Improvements
    Better preprocessing pipeline
    Automatic digit segmentation
    Support for multiple digits
    Data augmentation
    Higher accuracy CNN architectures


Screenshots:

Example:

![alt text](image-1.png)
![alt text](image-2.png)
![alt text](image-3.png)

Limitations:
As mentioned if the background is noisy or not centered there is limitation 

![alt text](image-4.png)
![alt text](image-5.png)

Author

Nagavenkata Kolli

License

This project is for educational and academic purposes.
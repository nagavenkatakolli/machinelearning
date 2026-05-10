import streamlit as st
from src.preprocessing import preprocess_uploaded_image
import tensorflow as tf
import pandas as pd
import numpy as np


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model/mnist_cnn_model.keras"
    )


def main():

    st.set_page_config(
        page_title="MNIST Digit Classifier",
        layout="wide"
    )

    st.title("MNIST Digit Classification with CNN")

    st.info(
    "For best results, upload a clear, tightly cropped image with one large dark digit on a plain light background. "
    "Avoid shadows, notebook lines, textured backgrounds, and very small digits."
    )

    model = load_model()

    uploaded_file = st.file_uploader(
        "Upload an image of a digit (0-9)",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:

        # Preprocess uploaded image
        image = preprocess_uploaded_image(uploaded_file)

        # Predict
        prediction = model.predict(image)

        predicted_digit = int(
            np.argmax(prediction[0])
        )

        confidence = float(
            np.max(prediction[0])
        )

        # Create scores dataframe
        scores_df = pd.DataFrame({
            "Digit": [str(i) for i in range(10)],
            "Probability": prediction[0],
            "Percent": [
                f"{p:.2%}" for p in prediction[0]
            ]
        })

        # Layout
        col1, col2, col3 = st.columns([1.5, 1, 1.5])

        # Uploaded image
        with col1:

            st.subheader("Uploaded Image")

            st.image(
                uploaded_file,
                use_container_width=True
            )

        # Processed image
        with col2:

            st.subheader("Processed Image")

            st.image(
                image.reshape(28, 28),
                width=250
            )

        # Prediction results
        with col3:

            st.subheader("Prediction")

            st.write(
                f"### Predicted Digit: {predicted_digit}"
            )

            st.write(
                f"### Confidence: {confidence:.2%}"
            )

            st.subheader("All Digit Scores")

            st.dataframe(
                scores_df,
                use_container_width=True
            )

            # Bar chart
            chart_df = scores_df[
                ["Digit", "Probability"]
            ].set_index("Digit")

            st.bar_chart(chart_df)


if __name__ == "__main__":
    main()
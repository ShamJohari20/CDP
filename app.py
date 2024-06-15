import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import io

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load class indices from JSON file
with open('VGG_CPD_class_indices.json', 'r') as f:
    class_indices = json.load(f)

def predict_disease(image):
    # Preprocess the image
    image = image.resize((224, 224))  # VGG16 input size
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize and convert to float32

    # Ensure the input tensor matches the model's expected input shape
    expected_shape = input_details[0]['shape'][1:]  # Skip the batch dimension if present
    image = np.reshape(image, expected_shape)

    # Set the tensor to the resized image
    interpreter.set_tensor(input_details[0]['index'], [image])

    # Perform prediction
    interpreter.invoke()

    # Get the predicted class
    prediction = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = class_indices[str(np.argmax(prediction))]

    return predicted_class

# Streamlit UI
st.title('Crops Disease Prediction')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image file
    image = Image.open(io.BytesIO(uploaded_file.read()))

    # Resize the image to a smaller size for display
    resized_image = image.resize((300, 200))

    st.image(resized_image, caption='Uploaded Image', use_column_width=True)

    if st.button('Predict'):
        try:
            prediction = predict_disease(image)
            st.success(f"Result: {prediction}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

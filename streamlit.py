import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

# Load the pre-trained model
model = load_model('asl_resnet.h5', compile=False)

# Class labels for ASL (A-Z, space, delete, etc.)
class_labels = {i: chr(65 + i) for i in range(26)}
class_labels[26] = 'Delete'
class_labels[27] = 'Space'
class_labels[28] = 'Nothing'

last_prediction_time = time.time()

# Function to preprocess the image before sending it to the classifier
def preprocess_image(image, size=(64, 64)):
    image = cv2.resize(image, size)
    image = np.array(image, dtype='float32') / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to draw a rectangle (demarcated area)
def draw_demarcation(frame):
    height, width, _ = frame.shape
    x_start, y_start, x_end, y_end = int(width * 0.3), int(height * 0.2), int(width * 0.7), int(height * 0.8)
    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    return frame, (x_start, y_start, x_end, y_end)

# Function to manage recognized text
def check_condition(recognized_text, recognized_letter):
    if recognized_letter == "Space":
        recognized_text += " "
    elif recognized_letter == "Delete":
        recognized_text = recognized_text[:-1]
    return recognized_text

# Streamlit App
def main():
    st.title("ASL Recognition System")
    st.write("This app captures sign language gestures and predicts the corresponding letters.")
    
    cap = cv2.VideoCapture(0)
    recognized_text = ''
    last_prediction_time = time.time()

    stframe = st.empty()  # For the camera feed
    textframe = st.empty()  # For the recognized text

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to access webcam.")
            break

        # Flip frame for mirror effect and draw demarcation
        frame = cv2.flip(frame, 1)
        frame, (x_start, y_start, x_end, y_end) = draw_demarcation(frame)

        # Extract and preprocess ROI for classification
        roi = frame[y_start:y_end, x_start:x_end]
        if time.time() - last_prediction_time >= 2:  # Prediction every 2 seconds
            processed_roi = preprocess_image(roi)
            prediction = model.predict(processed_roi)
            predicted_class = np.argmax(prediction)
            recognized_letter = class_labels[predicted_class]
            
            recognized_text = check_condition(recognized_text, recognized_letter)
            last_prediction_time = time.time()

        # Display the camera feed with demarcated area
        stframe.image(frame, channels="BGR", use_column_width=True)
        
        # Display recognized text
        text_frame = np.zeros((300, 600, 3), dtype='uint8')  # Black background
        cv2.putText(text_frame, f'Recognized Text: {recognized_text}', (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        textframe.image(text_frame, channels="BGR", use_column_width=True)

        # Add a stop button
        if st.button("Stop"):
            break

    cap.release()

if __name__ == "__main__":
    main()

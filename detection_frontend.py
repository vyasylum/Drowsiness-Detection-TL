import streamlit as st
import cv2
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

st.title("Welcome to Driver's Drowsiness System")
st.write("This system uses a deep learning model to detect drowsiness in drivers.")
st.write("If the system detects closed eyes for a longer period, it will sound an alarm to alert the driver.")

# Function to run the drowsiness detection
def run_drowsiness_detection():
    try:
        model = load_model(r'model/model.h5')
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    score = 0

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Downscale frame for faster processing
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

        for (ex, ey, ew, eh) in eyes:
            eye = frame[ey:ey+eh, ex:ex+ew]
            eye = cv2.resize(eye, (80, 80))
            eye = eye / 255
            eye = eye.reshape(80, 80, 3)
            eye = np.expand_dims(eye, axis=0)

            prediction = model.predict(eye)

            if prediction[0][0] > 0.30:
                score += 1
                if score > 5:
                    st.error("Drowsiness detected! Please wake up.")
                    break
            else:
                score -= 1
                if score < 0:
                    score = 0

        if st.button("End Detection"):
            break

        if st.checkbox("Show Webcam"):
            stframe = st.empty()
            stframe.image(frame, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()

# Run drowsiness detection when button is clicked
if st.button("Try Drowsiness Detection"):
    run_drowsiness_detection()

import queue
import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from tensorflow.keras.models import load_model
from pygame import mixer  # Import mixer for sound alert
import pygame 

webrtc_streamer(key="sample",  rtc_configuration={"iceServers": [{"urls": ["stun:stun.xten.com:3478"]}]} )# Load the drowsiness detection model
# @st.cache(allow_output_mutation=True)
# def load_drowsiness_model():
#     model_path = "model/model.h5"  # Adjust the path accordingly
#     return load_model(model_path)

# model = load_drowsiness_model()

# # Suppress Pygame audio initialization error
# try:
#     mixer.init()
# except pygame.error:
#     pass

# # Define the drowsiness detection function
# def run_drowsiness_detection(frame: av.VideoFrame) -> av.VideoFrame:
#     if model is None:
#         return frame

#     image = frame.to_ndarray(format="bgr24")

#     # Convert the frame to grayscale for drowsiness detection
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Your drowsiness detection logic here
#     # Use the `gray` image for detecting faces and eyes
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
#     eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

#     # Process the detections and determine drowsiness
#     for (ex, ey, ew, eh) in eyes:
#         # Extract and preprocess the eye region
#         eye = gray[ey:ey+eh, ex:ex+ew]
#         eye = cv2.resize(eye, (80, 80))  # Resize the eye region
#         eye = eye / 255.0  # Normalize the pixel values
#         eye = np.expand_dims(eye, axis=0)  # Add batch dimension

#         # Pass the preprocessed eye region to your model for prediction
#         prediction = model.predict(eye)

#         # Analyze the prediction to determine drowsiness and take necessary actions
#         if prediction[0][0] > 0.30:
#             # Detected drowsiness
#             # Perform actions like sounding an alarm, displaying an alert, etc.
#             sound = mixer.Sound(r'alarm.wav')
#             sound.play()  # Sound the alarm

#     # For demonstration purposes, let's just return the input frame
#     return frame

# # Set up the WebRTC streamer
# webrtc_ctx = webrtc_streamer(
#     key="drowsiness-detection"
# )

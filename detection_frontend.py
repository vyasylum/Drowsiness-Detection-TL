import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import cv2
import numpy as np
from tensorflow.keras.models import load_model
webrtc_ctx = webrtc_streamer(key="sample")
class DrowsinessDetection(VideoTransformerBase):
    def __init__(self, model):
        self.model = model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.score = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
        eyes = self.eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

        for (ex, ey, ew, eh) in eyes:
            eye = img[ey:ey+eh, ex:ex+ew]
            eye = cv2.resize(eye, (80, 80))
            eye = eye / 255
            eye = eye.reshape(80, 80, 3)
            eye = np.expand_dims(eye, axis=0)

            prediction = self.model.predict(eye)

            if prediction[0][0] > 0.30:
                self.score += 1
                if self.score > 5:
                    st.error("Drowsiness detected! Please wake up.")
                    break
            else:
                self.score -= 1
                if self.score < 0:
                    self.score = 0

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def main():
    st.title("Welcome to Driver's Drowsiness System")
    st.write("This system uses a deep learning model to detect drowsiness in drivers.")
    st.write("If the system detects closed eyes for a longer period, it will sound an alarm to alert the driver.")

    try:
        model = load_model(r'model/model.h5')
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return

    webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=lambda: DrowsinessDetection(model))
    if not webrtc_ctx.state.playing:
        st.warning("Please allow access to your webcam.")

if __name__ == "__main__":
    main()

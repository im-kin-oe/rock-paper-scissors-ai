import streamlit as st
import numpy as np
import cv2
import sys, os
from PIL import Image

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.predict import predict_image

st.set_page_config(page_title="RPS AI", page_icon="✊")

st.title("✊ Rock Paper Scissors AI (Global)")
st.write("📸 Take a photo of your hand and get prediction")

# -------------------------
# Camera Input (GLOBAL)
# -------------------------
image = st.camera_input("Show your hand")

# -------------------------
# Prediction
# -------------------------
if image is not None:
    # Convert image to OpenCV format
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Flip (mirror view like real camera)
    frame = cv2.flip(frame, 1)

    # Display captured image
    st.image(frame, channels="BGR", caption="Captured Image")

    # Preprocess
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (128, 128))

    # Predict
    label = predict_image(resized)

    # Show result
    st.markdown("---")
    st.markdown(f"## 🧠 Prediction: **{label.upper()}**")
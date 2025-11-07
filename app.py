# ----------------------------
# ✅ Headless Environment Fix
# ----------------------------
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

# ----------------------------
# 📦 Imports
# ----------------------------
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from model import EmotionDetector

# ----------------------------
# ⚙️ Load model (cached)
# ----------------------------
@st.cache_resource
def load_model():
    return EmotionDetector()

# ----------------------------
# 🎭 Main App
# ----------------------------
def main():
    st.set_page_config(page_title="Emotion Detection", page_icon="😊", layout="wide")
    st.title("🎭 Emotion Detection System")
    st.write("Upload an image or use your webcam to detect emotions!")

    # Load model
    detector = load_model()

    # Sidebar options
    st.sidebar.header("Options")
    input_method = st.sidebar.radio("Select Input Method:", ["Upload Image", "Camera"])

    # ----------------------------
    # 📤 Upload Image
    # ----------------------------
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)

            with st.spinner("Detecting emotions..."):
                result_img, emotions = detector.detect_emotions(img_array)

            with col2:
                st.subheader("Detection Result")
                st.image(result_img, use_container_width=True)

            display_emotions(emotions)

    # ----------------------------
    # 📸 Camera Input
    # ----------------------------
    else:
        st.write("Click the button below to capture an image from your camera")
        camera_image = st.camera_input("Take a picture")

        if camera_image is not None:
            image = Image.open(camera_image)
            img_array = np.array(image)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Captured Image")
                st.image(image, use_container_width=True)

            with st.spinner("Detecting emotions..."):
                result_img, emotions = detector.detect_emotions(img_array)

            with col2:
                st.subheader("Detection Result")
                st.image(result_img, use_container_width=True)

            display_emotions(emotions)

    # ----------------------------
    # ℹ️ Info Section
    # ----------------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info(
        "This application detects emotions from facial expressions using deep learning. "
        "It can identify: angry, disgust, fear, happy, sad, surprise, and neutral emotions."
    )


# ----------------------------
# 🧠 Helper Function
# ----------------------------
def display_emotions(emotions):
    """Display detected emotions neatly in Streamlit."""
    if emotions:
        st.subheader("Detected Emotions")
        for emotion_data in emotions:
            st.write(f"**Face {emotions.index(emotion_data) + 1}:**")

            emotion_cols = st.columns(len(emotion_data['emotions']))
            for idx, (emotion, score) in enumerate(emotion_data['emotions'].items()):
                with emotion_cols[idx]:
                    st.metric(emotion.capitalize(), f"{score:.1%}")

            st.write(f"**Dominant Emotion:** {emotion_data['dominant_emotion'].upper()}")
            st.write("---")
    else:
        st.warning("No faces detected in the image.")


# ----------------------------
# 🚀 Run App
# ----------------------------
if __name__ == "__main__":
    main()

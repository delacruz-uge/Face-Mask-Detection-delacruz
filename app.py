import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# Flexible imports for different environments
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
except ImportError:
    from keras.models import load_model
    from keras.preprocessing.image import img_to_array
    from keras.applications.mobilenet_v2 import preprocess_input

# App title and description
st.set_page_config(page_title="Face Mask Detector", layout="wide")
st.title("Face Mask Detection System")
st.write("Upload an image to detect whether people are wearing masks")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    confidence_threshold = st.slider(
        "Detection Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Adjust how confident the model needs to be to detect a face"
    )
    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("1. Upload an image (JPEG/PNG)")
    st.markdown("2. Click 'Detect Masks'")
    st.markdown("3. View results")

@st.cache_resource
def load_models():
    # Load face detector
    face_detector_dir = "face_detector"
    prototxt = os.path.join(face_detector_dir, "deploy.prototxt")
    weights = os.path.join(face_detector_dir, "res10_300x300_ssd_iter_140000.caffemodel")
    
    try:
        net = cv2.dnn.readNet(prototxt, weights)
    except Exception as e:
        st.error(f"Failed to load face detector: {str(e)}")
        st.stop()
    
    # Load mask classifier
    model_path = "mask_detector.model"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {os.path.abspath(model_path)}")
        st.stop()
    
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load mask detector model: {str(e)}")
        st.stop()
    
    return net, model

def detect_masks(image, net, model, confidence_threshold=0.5):
    """Detect faces and classify mask usage"""
    # Convert PIL image to OpenCV format
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    (h, w) = image.shape[:2]
    
    # Detect faces
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    # Process each detection
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Ensure bounding box is within image dimensions
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            # Extract face ROI and preprocess
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            
            # Predict mask/no mask
            (mask, without_mask) = model.predict(face)[0]
            
            # Determine label and color
            label = "Mask" if mask > without_mask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            
            # Include confidence percentage
            label = f"{label}: {max(mask, without_mask) * 100:.2f}%"
            
            # Display bounding box and label
            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
    
    # Convert back to RGB for Streamlit
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Main application logic
def main():
    try:
        net, model = load_models()
    except Exception as e:
        st.error(f"Failed to initialize models: {str(e)}")
        st.stop()
    net, model = load_models()
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        
        if st.button("Detect Masks"):
            with st.spinner("Processing image..."):
                processed_image = detect_masks(
                    image,
                    net,
                    model,
                    confidence_threshold
                )
            
            with col2:
                st.image(
                    processed_image,
                    caption="Detection Results",
                    use_column_width=True
                )
            
            st.success("Detection complete!")

if __name__ == "__main__":
    main()

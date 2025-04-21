import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from PIL import Image

@st.cache_resource
def load_models():
    # Load face detector model
    print("loading face detector model")
    face_detector_dir = "face_detector"
    prototxtPath = os.path.sep.join([face_detector_dir, "deploy.prototxt"])
    weightsPath = os.path.sep.join([face_detector_dir, "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)
    
    # Load mask detector model
    print("loading face mask detector model...")
    model = load_model("model.h5")
    
    return net, model

net, model = load_models()

st.write("""
# Face Mask Detection System
""")

confidence_threshold = st.slider("Confidence Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

def detect_mask(image, confidence):
    # Convert PIL image to OpenCV format
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    orig = image.copy()
    (h, w) = image.shape[:2]

    # Construct blob and perform face detection
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        conf = detections[0, 0, i, 2]
        
        # Filter out weak detections
        if conf > confidence:
            # Compute bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Ensure bounding boxes fall within image dimensions
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            # Extract face ROI, preprocess it
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            
            # Predict mask or no mask
            (mask, withoutMask) = model.predict(face)[0]
            
            # Determine label and color
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            
            # Include confidence in label
            label = f"{label}: {max(mask, withoutMask) * 100:.2f}%"
            
            # Display label and bounding box
            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
    
    # Convert back to RGB for display in Streamlit
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Process image when button is clicked
    if st.button("Detect Face Masks"):
        processed_image = detect_mask(image, confidence_threshold)
        st.image(processed_image, caption="Detection Result", use_column_width=True)
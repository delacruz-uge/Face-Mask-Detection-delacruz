import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# Flexible imports for TF/Keras
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
except:
    from keras.models import load_model
    from keras.preprocessing.image import img_to_array
    from keras.applications.mobilenet_v2 import preprocess_input

@st.cache_resource
def load_models():
    # Load face detector model
    face_detector_dir = "face_detector"
    prototxtPath = os.path.join(face_detector_dir, "deploy.prototxt")
    weightsPath = os.path.join(face_detector_dir, "res10_300x300_ssd_iter_140000.caffemodel")
    net = cv2.dnn.readNet(prototxtPath, weightsPath)
    
    # Load mask detector model
    model = load_model("model.h5")
    return net, model

net, model = load_models()

st.title("Face Mask Detection System")
confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05)
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

def detect_mask(image, confidence):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            
            (mask, withoutMask) = model.predict(face)[0]
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = f"{label}: {max(mask, withoutMask) * 100:.2f}%"
            
            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Detect Masks"):
        processed_image = detect_mask(image, confidence_threshold)
        st.image(processed_image, caption="Detection Result", use_column_width=True)
else:
    st.write("Please upload an image file.")

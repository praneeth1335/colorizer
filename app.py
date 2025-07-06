import numpy as np
import cv2
import streamlit as st
from PIL import Image
import os

def colorizer(img):
    # Convert to grayscale then back to RGB (simulating B&W input)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Load models using relative paths
    prototxt = "./models/models_colorization_deploy_v2.prototxt"
    model = "./models/colorization_release_v2.caffemodel"
    points = "./models/pts_in_hull.npy"
    
    # Verify model files exist
    if not all(os.path.exists(f) for f in [prototxt, model, points]):
        st.error("Model files not found. Please ensure they are in the models directory.")
        return img
    
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    pts = np.load(points)
    
    # Add cluster centers to model
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    
    # Process image
    scaled = img.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    
    # Predict color
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")
    
    return colorized

# Streamlit UI
st.title("Colorizing Black & White Images")
st.write("Upload a black and white image to see it colorized")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    
    with col2:
        st.text("Colorizing...")
        colorized = colorizer(img)
        st.image(colorized, caption="Colorized Image", use_column_width=True)
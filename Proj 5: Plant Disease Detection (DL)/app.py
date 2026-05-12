import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="Plant AI", page_icon="🌿")

@st.cache_resource # Helper function to load model safely
def load_my_model():
    if os.path.exists('plant_model.h5'):
        return tf.keras.models.load_model('plant_model.h5')
    return None

model = load_my_model()

st.title("Leaf Disease Classifier 🌿")

if model is None:
    st.error("Model file not found! Please run 'python model_training.py' first to generate the model.")
else:
    uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        img = image.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        if st.button('Run AI Diagnosis'):
            prediction = model.predict(img_array)
            result_index = np.argmax(prediction)
            
            class_names = ['Early Blight', 'Healthy', 'Late Blight']
            diagnosis = class_names[result_index]
            confidence = np.max(prediction) * 100

            st.success(f"Diagnosis: **{diagnosis}**")
            st.info(f"Confidence Level: {confidence:.2f}%")

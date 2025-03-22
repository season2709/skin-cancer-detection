import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import shap
import lime
import lime.lime_image
import cv2
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

# Load model
model = load_model('skin_cancer_model.h5')

# Classes dictionary
classes = {0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df', 4: 'nv', 5: 'vasc', 6: 'mel'}

# Preprocessing function
def preprocess_image(image):
    image = cv2.resize(image, (28, 28))
    image = image / 255.0
    return image.reshape(1, 28, 28, 3)

# SHAP explanation function
def explain_shap(img):
    background = np.random.rand(100, 28, 28, 3)
    explainer = shap.Explainer(model.predict, background)
    shap_values = explainer(img)
    shap.image_plot(shap_values.values, img)

# LIME explanation function
def explain_lime(img):
    explainer = lime.lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img[0].astype('double'), classifier_fn=model.predict, top_labels=1, num_samples=1000
    )
    lime_img, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, hide_rest=False
    )
    fig, ax = plt.subplots()
    ax.imshow(mark_boundaries(lime_img, mask.astype(int)))
    ax.axis('off')
    st.pyplot(fig)

# Streamlit UI
st.title('Skin Cancer Detection')

uploaded_files = st.file_uploader("Upload one or more skin lesion images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.write(f"### Results for: {uploaded_file.name}")
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        processed_img = preprocess_image(img)
        prediction = model.predict(processed_img)[0]
        predicted_class = classes[np.argmax(prediction)]

        st.write(f"**Prediction:** {predicted_class}")
        st.write(f"**Confidence:** {np.max(prediction) * 100:.2f}%")

        st.write("#### SHAP Explanation")
        fig_shap, ax_shap = plt.subplots()
        background = np.random.rand(100, 28, 28, 3)
        explainer = shap.Explainer(model.predict, background)
        shap_values = explainer(processed_img)
        shap.image_plot(shap_values.values, processed_img)
        st.pyplot(plt.gcf())

        st.write("#### LIME Explanation")
        explain_lime(processed_img)

        st.markdown("---")

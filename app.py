import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from PIL import Image
import lime
from lime import lime_image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("model.h5")

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function for prediction
def predict(image):
    prediction = model.predict(image)
    class_names = ["Melanoma", "Nevus", "Seborrheic Keratosis"]
    return class_names[np.argmax(prediction)], prediction

# Function for LIME explanation
def explain_with_lime(image):
    explainer = lime_image.LimeImageExplainer()
    
    def model_predict(imgs):
        imgs = np.array([preprocess_image(Image.fromarray((img * 255).astype(np.uint8))).reshape(224,224,3) for img in imgs])
        return model.predict(imgs)

    explanation = explainer.explain_instance(
        np.array(image) / 255.0,
        model_predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False
    )
    return temp, mask

st.title("Skin Cancer Detection")

uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    preprocessed_img = preprocess_image(image)
    prediction_class, prediction_prob = predict(preprocessed_img)

    st.write(f"### Prediction: {prediction_class}")
    st.write(f"### Confidence: {np.max(prediction_prob) * 100:.2f}%")

    lime_img, lime_mask = explain_with_lime(image)
    lime_boundary_img = mark_boundaries(lime_img, lime_mask)
    lime_boundary_img = (lime_boundary_img * 255).astype(np.uint8)

    st.write("## LIME Explanation:")
    st.image(lime_boundary_img, caption="LIME Explanation", use_column_width=True)

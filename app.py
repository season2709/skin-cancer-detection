import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import shap
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Load the trained model (.h5)
model = tf.keras.models.load_model('Skin.h5')

# Class names mapping:
classes = {
    0: 'Actinic keratoses and intraepithelial carcinomae (akiec)',
    1: 'Basal cell carcinoma (bcc)',
    2: 'Benign keratosis-like lesions (bkl)',
    3: 'Dermatofibroma (df)',
    4: 'Melanocytic nevi (nv)',
    5: 'Vascular lesions (vasc)',
    6: 'Melanoma (mel)'
}

st.title("Skin Cancer Detection Web App")
st.write("Upload an image of a skin lesion to get prediction, probability scores, and visual explanations (SHAP & LIME).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Processing and classifying...")

    # Preprocessing
    image_resized = image.resize((28, 28))
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0]
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.write(f"## 🩺 Predicted Cancer Type: **{classes[predicted_class]}**")
    st.write(f"### ✅ Confidence: **{confidence:.2f}%**")

    # Probability bar chart
    st.write("### 📊 Class Probability Scores:")
    fig, ax = plt.subplots()
    ax.bar(classes.values(), prediction, color='skyblue')
    ax.set_ylabel("Probability")
    ax.set_xticklabels(classes.values(), rotation=45, ha="right")
    st.pyplot(fig)

    # SHAP Explanation
    st.write("---")
    st.write("## 🤖 SHAP Explanation")
    st.write("""
    SHAP (SHapley Additive exPlanations) shows how each pixel in the image influences the model's decision. 
    Red areas push the prediction towards the predicted class, while blue areas push it away.
    """)
    background = np.random.rand(1, 28, 28, 3)
    explainer = shap.Explainer(model, background)
    shap_values = explainer(img_array)
    shap.image_plot(shap_values.values, img_array)

    # LIME Explanation
    st.write("---")
    st.write("## 🔎 LIME Explanation")
    st.write("""
    LIME (Local Interpretable Model-agnostic Explanations) highlights superpixels (image segments) that contribute most to the prediction.
    Green areas indicate positive influence toward the predicted class.
    """)
    lime_explainer = lime_image.LimeImageExplainer()
    explanation = lime_explainer.explain_instance(
        np.array(image_resized), 
        model.predict, 
        top_labels=1, 
        hide_color=0, 
        num_samples=1000
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], 
        positive_only=True, 
        num_features=5, 
        hide_rest=False
    )
    fig, ax = plt.subplots()
    ax.imshow(mark_boundaries(temp / 255.0, mask))
    ax.axis('off')
    st.pyplot(fig)

    st.write("---")
    st.write("## 📜 Prediction Mapping:")
    for i, c in classes.items():
        st.write(f"**{i}** ➡️ {c}")

    st.success("Prediction and explanations generated successfully!")


import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import shap
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# Load the trained model
model = load_model("model.h5")

# Class labels
classes = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]

# Streamlit UI
st.title("Skin Cancer Detection using CNN")

st.write("Upload skin lesion images (PNG/JPG) for classification.")

uploaded_files = st.file_uploader("Choose images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

def preprocess_image(image_np):
    img_resized = cv2.resize(image_np, (28, 28))
    img_array = np.expand_dims(img_resized, axis=0) / 255.0
    return img_array

if uploaded_files:
    st.subheader("Results")

    explainer_lime = lime_image.LimeImageExplainer()

    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        st.write(f"### Image {idx+1}")
        st.image(image, caption="Uploaded Image", width=250)

        # Preprocess image and predict
        img_array = preprocess_image(image_np)
        prediction = model.predict(img_array)[0]
        predicted_class = np.argmax(prediction)
        class_name = classes[predicted_class]
        confidence = prediction[predicted_class] * 100

        st.write(f"**Predicted Class:** {class_name}")
        st.write(f"**Confidence:** {confidence:.2f}%")

        # SHAP Explanation
        st.write("**SHAP Explanation:**")
        # Use SHAP DeepExplainer for TensorFlow/Keras models
        background = np.zeros((1, 28, 28, 3))
        shap_explainer = shap.DeepExplainer(model, background)
        shap_values = shap_explainer.shap_values(img_array)
        shap.image_plot(shap_values, img_array, show=False)
        st.pyplot(plt.gcf())
        plt.clf()

        # LIME Explanation
        st.write("**LIME Explanation:**")
        explanation = explainer_lime.explain_instance(
            image_np.astype('double'),
            classifier_fn=lambda x: model.predict(
                np.array([cv2.resize(i, (28, 28)) / 255.0 for i in x])
            ),
            top_labels=1,
            hide_color=0,
            num_samples=500
        )
        lime_img, lime_mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=False,
            num_features=5,
            hide_rest=False
        )
        lime_img_normalized = lime_img / 255.0  # normalize
        lime_boundary_img = mark_boundaries(lime_img_normalized, lime_mask)
        lime_boundary_img = np.clip(lime_boundary_img, 0, 1)
        st.image(lime_boundary_img, caption="LIME Explanation", width=250)

        st.markdown("---")

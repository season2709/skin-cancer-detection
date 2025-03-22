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

# Load model
model = load_model("model.h5")

# Class labels
classes = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]

# Streamlit UI
st.title("🌟 Skin Cancer Detection using CNN 🌟")
st.write("Upload a skin lesion image (PNG/JPG) for classification.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with st.spinner("Uploading and processing image... 🔄"):
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        st.image(image, caption="Uploaded Image", width=250)

        # Preprocess image
        img_resized = cv2.resize(image_np, (28, 28))
        img_array = np.expand_dims(img_resized, axis=0) / 255.0

        # Prediction
        st.info("Predicting class... 🤖")
        prediction = model.predict(img_array)[0]
        predicted_class = np.argmax(prediction)
        class_name = classes[predicted_class]
        confidence = prediction[predicted_class] * 100

        st.success(f"✅ **Predicted Class:** {class_name}")
        st.write(f"**Confidence:** {confidence:.2f}%")

        # SHAP Explanation
        st.info("Generating SHAP explanation... 🧩")
        background = np.zeros((1, 28, 28, 3))
        shap_explainer = shap.DeepExplainer(model, background)
        shap_values = shap_explainer.shap_values(img_array)
        shap.image_plot(shap_values, img_array, show=False)
        st.pyplot(plt.gcf())
        plt.clf()

        # LIME Explanation
        st.info("Generating LIME explanation... 🟢")
        explainer_lime = lime_image.LimeImageExplainer()
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
        lime_img_normalized = lime_img / 255.0
        lime_boundary_img = mark_boundaries(lime_img_normalized, lime_mask)
        lime_boundary_img = np.clip(lime_boundary_img, 0, 1)
        st.image(lime_boundary_img, caption="LIME Explanation", width=250)

        st.balloons()
        st.success("🎉 Done! Your prediction and explanations are ready!")


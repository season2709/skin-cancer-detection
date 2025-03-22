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
st.title("🌟 Skin Cancer Detection 🌟")

st.write("Upload skin lesion images (PNG/JPG/JPEG) for classification.")

st.markdown(
    """
    <h3 style="background: linear-gradient(90deg, #ff4b4b, #4bffb0, #ffb84d); -webkit-background-clip: text; color: transparent;">
   🩺 Upload your skin images — let's get you a quick checkup! 🩺
    </h3>
    """,
    unsafe_allow_html=True
)

st.markdown("""
    <style>
    /* Style the file uploader box */
    .stFileUploader > div > div {
        border: 2px dashed #ff4b4b;
        background: linear-gradient(135deg, #ffb84d, #4bffb0, #4b86f5);
        padding: 20px;
        border-radius: 20px;
        box-shadow: 0 0 10px rgba(0,0,0,0.2);
        transition: 0.3s ease;
    }
    .stFileUploader > div > div:hover {
        box-shadow: 0 0 20px rgba(0,0,0,0.3);
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader("Choose images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

st.markdown("""
    <style>
    .colorful-subheader {
        font-size: 30px;
        font-weight: bold;
        background: linear-gradient(90deg, #ff4b4b, #ffb84d, #4bffb0, #4b86f5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 3s ease infinite;
        text-align: center;
        padding: 10px 0;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    </style>
""", unsafe_allow_html=True)

if uploaded_files:
    st.markdown('<div class="colorful-subheader">🎨 Results are Ready! 🎉</div>', unsafe_allow_html=True)

    # SHAP explainer
    explainer = shap.GradientExplainer(model, np.zeros((1, 28, 28, 3)))
    explainer_lime = lime_image.LimeImageExplainer()

    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        st.write(f"### Image {idx+1}")
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

        st.write(f"**Predicted Class:** {class_name}")
        st.write(f"**Confidence:** {confidence:.2f}%")

        # SHAP Explanation
        st.write("**SHAP Explanation:**")
        st.info("Generating SHAP explanation... 🤖")
        shap_values = explainer.shap_values(img_array)
        shap.image_plot(shap_values, img_array, show=False)
        st.pyplot(plt.gcf())
        plt.clf()

        # LIME Explanation
        st.write("**LIME Explanation:**")
        st.info("Generating LIME explanation... 🤖")
        explanation = explainer_lime.explain_instance(
            img_resized.astype('double'),
            classifier_fn=lambda x: model.predict(x / 255.0),
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
        lime_img_normalized = lime_img / 255.0  # ensure lime image is in [0,1]
        lime_boundary_img = mark_boundaries(lime_img_normalized, lime_mask)
        lime_boundary_img = np.clip(lime_boundary_img, 0, 1)  # clip just in case
        st.image(lime_boundary_img, caption="LIME Explanation", width=250)


        st.markdown("---")
        st.balloons()
        st.success("🎉 Mission Completed!")

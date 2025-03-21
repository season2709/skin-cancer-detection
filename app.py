import streamlit as st
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import shap
import lime
import lime.lime_image
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries, slic


# Load the trained model
model = load_model('model.h5')

# Class dictionary
classes = {
    4: ('nv', 'Melanocytic nevi'),
    6: ('mel', 'Melanoma'),
    2: ('bkl', 'Benign keratosis-like lesions'),
    1: ('bcc', 'Basal cell carcinoma'),
    5: ('vasc', 'Pyogenic granulomas and hemorrhage'),
    0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
    3: ('df', 'Dermatofibroma')
}

st.title("Skin Cancer Detection")
st.write("Upload a skin lesion image for prediction and explanation.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocessing
    image_resized = image.resize((28, 28))
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Predict"):
        prediction = model.predict(img_array)[0]
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]

        st.write(f"**Prediction:** {classes[predicted_class][1]} ({classes[predicted_class][0]})")
        st.write(f"**Confidence:** {confidence * 100:.2f}%")

        st.write("### Prediction probabilities:")
        probs = prediction * 100
        class_names = [classes[i][0] for i in range(len(classes))]

        fig, ax = plt.subplots()
        ax.bar(class_names, probs, color='skyblue')
        ax.set_ylabel("Probability (%)")
        ax.set_title("Class Probabilities")
        st.pyplot(fig)

        st.write("### SHAP Explanation:")
        background = np.random.rand(100, 28, 28, 3)
        masker = shap.maskers.Image("inpaint_telea", (28, 28, 3))
        explainer = shap.Explainer(model.predict, masker)
        shap_values = explainer(img_array)

        fig, ax = plt.subplots()
        shap.image_plot(shap_values.values, -img_array, show=False)
        st.pyplot(fig)

        st.write("### Improved LIME Explanation:")

        def custom_segmentation(image):
            return slic(image, n_segments=150, compactness=10, sigma=1)

        explainer_lime = lime.lime_image.LimeImageExplainer()
        explanation = explainer_lime.explain_instance(
            np.array(image_resized).astype('double'),
            classifier_fn=lambda x: model.predict(x / 255.0),
            top_labels=1,
            hide_color=0,
            num_samples=2000,
            segmentation_fn=custom_segmentation
        )

        lime_img, mask = explanation.get_image_and_mask(
            label=explanation.top_labels[0],
            positive_only=True,
            hide_rest=False,
            num_features=10,
            min_weight=0.0
        )

        # Normalize mask and upscale for better visualization
        mask_normalized = mask / mask.max()
        lime_img_upscaled = cv2.resize(lime_img, (224, 224), interpolation=cv2.INTER_CUBIC)
        mask_upscaled = cv2.resize(mask_normalized, (224, 224), interpolation=cv2.INTER_NEAREST)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(mark_boundaries(lime_img_upscaled, mask_upscaled))
        im = ax.imshow(mask_upscaled, cmap='jet', alpha=0.4)
        plt.colorbar(im, ax=ax, shrink=0.7)
        ax.axis('off')
        st.pyplot(fig)

        st.success("SHAP and LIME explanations generated!")

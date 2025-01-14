import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json
import os
from googletrans import Translator
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key="AIzaSyAAwCfWZ0RhLtyQYzPKgGcWjz7BHP3HHeE")
model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize the Google Translator
translator = Translator()

# Function to load and preprocess image
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]  # Remove alpha channel if present
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype('float32') / 255.  # Scale image values
    return img_array

# Predict the class of an image
def predict_image_class(model, image, class_indices_rev):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices_rev.get(predicted_class_index, "Unknown")
    confidence = np.max(predictions) * 100
    return predicted_class_name, confidence

# Translate text to a specified language
def translate_text(text, target_language):
    try:
        translated = translator.translate(text, dest=target_language)
        return translated.text
    except Exception as e:
        return f"Error translating text: {e}"

# Get detailed solution using Gemini API
def get_disease_solution(disease_name):
    try:
        prompt = (f"Provide a detailed explanation for the plant disease '{disease_name}', "
                  f"including why it occurs, conditions that cause it, reasons for its spread, "
                  f"and in which Indian states it is most prevalent.")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error fetching disease solution: {e}"

# Main app function
def main():
    st.title("🌾 Plant Disease Detection App")
    st.write("Upload an image of a crop leaf, and the app will predict the disease.")

    # Load model
    model_path = 'plant_disease_prediction_model.h5'
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please ensure the model is trained and saved.")
        return

    @st.cache_resource  # Cache the model
    def load_trained_model():
        try:
            return load_model(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

    model = load_trained_model()
    if model is None:
        return

    # Load class indices
    class_indices_path = 'class_indices.json'
    if not os.path.exists(class_indices_path):
        st.error(f"Class indices file '{class_indices_path}' not found. Please ensure it exists.")
        return

    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)

    class_indices_rev = {int(k): v for k, v in class_indices.items()}

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Language selection for translation
    target_language = st.selectbox("Select a language for the solution:", ["en", "hi", "te", "ta", "bn", "ml", "mr"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("Classifying...")

            predicted_class, confidence = predict_image_class(model, image, class_indices_rev)
            st.write(f"### Prediction: **{predicted_class}**")
            st.write(f"**Confidence:** {confidence:.2f}%")

            # Get detailed solution for the disease from Gemini API
            disease_solution = get_disease_solution(predicted_class)
            
            # Display the solution in a colored container with Streamlit-native styling
            with st.container():
                st.subheader("📝 Disease Solution")
                st.markdown(
                    f"""
                    **{predicted_class} Disease Information**  
                    - {disease_solution}
                    """,
                    unsafe_allow_html=True
                )

            # Translate the solution into the selected language and display it with native Streamlit styling
            st.subheader("🌐 Translated Solution")
            translated_solution = translate_text(disease_solution, target_language)
            st.markdown(
                f"""
                **Translation ({target_language}):**  
                {translated_solution}
                """,
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Error processing the image: {e}")

if __name__ == "__main__":
    main()

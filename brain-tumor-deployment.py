# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 20:35:27 2023

@author: Richard Adeyemi
"""


import keras
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import cv2
import tensorflow



def welcome_page():
    st.title("Welcome to Brain Tumor Detection App")
    st.header("Your First Step Towards Early Detection and Peace of Mind")

    # Key Features section
    st.subheader("Key Features")
    st.write("- **Early Detection:** Our advanced algorithm and image recognition technology can help identify potential brain tumors at an early stage.")
    st.write("- **User-Friendly:** With an intuitive interface, the app is easy to navigate and use. It's designed for users of all ages and levels of tech-savviness.")
    st.write("- **Instant Results:** Receive quick and accurate results, providing you with peace of mind or prompting you to seek further medical advice.")

    # How it Works section
    st.subheader("How it Works")
    st.write("Using the Brain Tumor Detection App is as simple as 1-2-3:")
    st.write("1. **Upload an Image:** Take or upload a head MRI or CT scan image. Our app will securely process the image.")
    st.write("2. **Processing and Analysis:** Our powerful algorithm will analyze the image to identify any potential anomalies.")
    st.write("3. **Receive Results:** You will receive the results promptly.")


    # Disclaimer section
    st.subheader("Disclaimer")
    st.write("Please note that the Brain Tumor Detection App is intended for informational purposes and is not a replacement for professional medical advice. Always consult with a healthcare professional for accurate diagnosis and treatment recommendations.")


st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
def load_tumor_type_model():
    model = keras.models.load_model("C:/Users/lenovo/Documents/Final Year Project/My Final Year Project/External Projects/Andrew/Masters project/Richard_Adeyemi_Source_Codes/Model-MRI-Kaggle-Dataset.hdf5")
    
    return model


st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
def load_tumor_presence_model():
    model = keras.models.load_model("C:/Users/lenovo/Documents/Final Year Project/My Final Year Project/External Projects/Andrew/Masters project/Richard_Adeyemi_Source_Codes/Model-BR3H-Dataset.hdf5")
    
    return model

def prediction_model(img, model):
    opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(opencvImage,(224,224))
    img = img.reshape(1,224,224,3)
    prediction = model.predict(img)
    
    prediction = np.argmax(prediction, axis=1)[0]
    
    return prediction

def main():
    with st.sidebar:
        st.write("**Name:** Richard Adeyemi")
        st.write("**ID Number:** 22842201")
        
        model_selection = st.selectbox("Select a Model", ["Model 1", "Model 2"])
        if model_selection == "Model 1":
            st.write("**Model one determines if the MRI image is one of the types of Brain Tumor: Glioma, Meningioma, Pituitary, or No tumor.**")
        elif model_selection == "Model 2":
            st.write("**Model two determines the presence of tumor or no tumor.**")
        
    uploaded_file = st.file_uploader("Upload a Brain MRI File", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            new_image = image.resize((200, 200))
            st.image(new_image, use_column_width=False)
            
            if model_selection == "Model 1":
                model = load_tumor_type_model()
                label = prediction_model(image, model)
                
                if st.button("RUN TEST"):
                    
                    if label == 0:
                        st.write(" **TEST RESULT: Glioma Tumor Detected**")
                    elif label == 1:
                        st.write("**TEST RESULT: No Tumor Detected**")
                    elif label == 2:
                        st.write("**TEST RESULT: Meningioma Tumor Detected**")
                    else:
                        st.write("**TEST RESULT: Pituitary Tumor Detected**")
                else:
                    st.write("")
            
        
            elif model_selection == "Model 2":
                model = load_tumor_presence_model()
                label = prediction_model(image, model)
                
                if st.button("RUN TEST"):
                    
                    if label == 0:
                        st.write(" **Tumor Status: No Tumor Detected**")
                    elif label == 1:
                        st.write("**Tumor Status: Tumor Detected**")
                        
        except Exception as e:
            st.write("Error processing the image:", str(e))
        
tab_title = ["**Welcome Page**",
             "**Prediction Page**"]

tabs = st.tabs(tab_title)

with tabs[0]:
    welcome_page()

with tabs[1]:
    if __name__ == "__main__":
        main()
        
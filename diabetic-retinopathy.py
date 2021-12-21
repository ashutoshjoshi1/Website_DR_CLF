import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import os

def makeitcool(image):
    x,y = image.size
    a = 400/x
    b = 400/y
    width = int(image.size[1] * b )
    height = int(image.size[0] * a)
    dim = (width, height)
    resized = image.resize((width,height), Image.ANTIALIAS)
    return resized

def import_and_predict(image_data, model):
    size = (400,400)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = (image.astype(np.float32) / 255.0)
    
    img_reshape = image[np.newaxis,...]

    prediction = model.predict(img_reshape)
    
    return prediction

model = tf.keras.models.load_model('full_retina_model.h5')

st.write("""
         # Diabetic-Retinopathy Classifier
         """
         )

st.write("Diabetic retinopathy is an eye disease caused by the high blood sugar from diabetes.")
st.write("This is a simple image classification web app to predict DR severity")

P_name = st.text_input("Enter Full Name")

P_no = st.text_input("Enter Mobile Number")

P_ID = st.text_input("Enter Unique ID")

Pres = st.text_input("Enter Prescription")

file = st.file_uploader("Please upload an image file. Note: Image dim (400,400)", type=["jpg","jpeg", "png","tif"])
#


if file is None:
    st.text("You haven't uploaded an image file")

else:
    img = Image.open(file)
    image = makeitcool(img)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)

    # file_details = {"FileName":file.name,"FileType":file.type}
    # st.write(file_details)
    # st.image(img)
    with open(os.path.join("tempDir",P_ID),"wb") as f: 
        f.write(file.getbuffer())         
    st.write("Saved File")
    if np.argmax(prediction) == 0:
        st.success("No DR")
    # elif np.argmax(prediction) == 1:
    #     st.write("One : Mild")
    # elif np.argmax(prediction) == 2:
    #     st.write("Two : Moderate")
    # elif np.argmax(prediction) == 3:
    #     st.write("Three : Severe")
    else:
        st.warning("DR Present")
    
    st.text(Pres)
    # st.write(prediction)
import streamlit as st
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
from utils import pil_resize_image
import numpy as np
import cv2

# Load model
model = tf.keras.models.load_model("./digit_recognizer.h5")

st.title("Hey, Welcome to my digit recognition app")

st.write("Type a digit")

canvas_board = st_canvas(width=300, height=300, stroke_color='red')

if canvas_board.image_data is not None:
    img = canvas_board.image_data
    img_resized = cv2.resize(img.astype('uint8'), (28, 28))
    rescaled = cv2.resize(img_resized, (300, 300), interpolation=cv2.INTER_NEAREST)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    pred = model.predict(tf.reshape(img_gray, [1, 28, 28, 1]))
    st.image(rescaled)
    st.subheader(f"I'm try to predict, the digit is {np.argmax(pred)}, right ?")

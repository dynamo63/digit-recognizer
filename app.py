import streamlit as st
st.set_page_config(page_title="Digit Recognition")
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2

# Load model
model = tf.keras.models.load_model("./models/digit_recognizer.h5")

st.title("Hey, Welcome to my digit recognition app")
st.markdown("""
    [Here](https://www.kaggle.com/code/mouckeytoumoulongui/get-started-cnn) the Kaggle link for the model.
    Interest to code ? Here my [repo](https://github.com/dynamo63/digit-recognizer)
""")

col1, col2 = st.columns(2)

with col1:
    st.write("Type a digit")
    canvas_board = st_canvas(width=300, height=300, stroke_color='red')
    is_clicked = st.button("predict")


with col2:
    if canvas_board.image_data is not None and is_clicked:
        img = canvas_board.image_data
        img_resized = cv2.resize(img.astype('uint8'), (28, 28))
        rescaled = cv2.resize(img_resized, (300, 300), interpolation=cv2.INTER_NEAREST)
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        pred = model.predict(tf.reshape(img_gray, [1, 28, 28, 1]))
        st.image(rescaled)
        st.subheader(f"I'm try to predict, the digit is {np.argmax(pred)}, right ?")

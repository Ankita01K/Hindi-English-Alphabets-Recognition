import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Define class names
class_names = ['a', 'b', 'c', 'character_10_yna', 'character_11_taamatar', 'character_12_thaa', 'character_13_daa',
               'character_14_dhaa', 'character_15_adna', 'character_16_tabala', 'character_17_tha', 'character_18_da',
               'character_19_dha', 'character_1_ka', 'character_20_na', 'character_21_pa', 'character_22_pha',
               'character_23_ba', 'character_24_bha', 'character_25_ma', 'character_26_yaw', 'character_27_ra',
               'character_28_la', 'character_29_waw', 'character_2_kha', 'character_30_motosaw', 'character_31_petchiryakha',
               'character_32_patalosaw', 'character_33_ha', 'character_34_chhya', 'character_35_tra', 'character_36_gya',
               'character_3_ga', 'character_4_gha', 'character_5_kna', 'character_6_cha', 'character_7_chha',
               'character_8_ja', 'character_9_jha', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
               'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# Streamlit app
st.title("Handwritten Hindi-English Letter Recognition App")

uploaded_file = st.file_uploader("Choose a handwritten image...", type="png")
# ...

# ...

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = Image.open(uploaded_file)
    image = image.resize((32, 32))  # Resize to (32, 32)
    image_array = np.asarray(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension for grayscale

    if image_array.shape[-1] == 1:
        # Convert grayscale to RGB
        image_array = np.concatenate([image_array, image_array, image_array], axis=-1)

    image_array = tf.convert_to_tensor(image_array, dtype=tf.float32)  # Convert to tf.Tensor
    image_array = tf.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image_array)
    predicted_class = class_names[np.argmax(prediction)]

    # Display the prediction
    st.write(f"Prediction: {predicted_class}")

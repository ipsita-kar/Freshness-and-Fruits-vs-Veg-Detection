import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import load_model
import os
import datetime

# Load the trained model
model = load_model('FV.h5')

# Labels corresponding to the output of the model
labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalapeño', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'radish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

# Classify which are fruits and which are vegetables
fruits = ['apple', 'banana', 'bell pepper', 'chilli pepper', 'grapes', 'jalapeño', 'kiwi', 'lemon', 'mango', 'orange',
          'paprika', 'pear', 'pineapple', 'pomegranate', 'watermelon']
vegetables = ['beetroot', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'corn', 'cucumber', 'eggplant', 'ginger',
              'lettuce', 'onion', 'peas', 'potato', 'radish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato',
              'tomato', 'turnip']

# Function to prepare the image for prediction
def prepare_image(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = int(y_class[0])
    result = labels[y]
    return result.capitalize()

# Main function for the app
def run():
    st.title("Fruits🍍-Vegetable🍅 Classification")

    # Image upload section
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])

    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)

        # Create a directory to store uploaded images, if it doesn't exist
        save_dir = './upload_images/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save the uploaded image to the folder with a timestamp
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_image_path = os.path.join(save_dir, f'{current_time}_{img_file.name}')

        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        # Display the timestamp of the upload
        st.info(f'**Upload Time:** {current_time}')

        # Predict the class of the image
        result = prepare_image(save_image_path)

        # Check if the result is a fruit or a vegetable
        if result.lower() in vegetables:
            st.info('**Category : Vegetables**')
        else:
            st.info('**Category : Fruit**')

        st.success(f"**Predicted : {result}**")

run()

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained model
model = load_model("/content/my_model.hdf5")

# Function to preprocess and make predictions
def process_image(image_path, target_size=(180, 180)):
    img = Image.open(image_path)
    img = img.resize(target_size)  # Resize the image to a fixed size
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array

# Load the class names used during training
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']  # Replace with your actual class names

# Streamlit App
st.title("Flower Classification App")

# Notice for users
st.markdown(
    """
    **Note for Users:** Please upload an image of one of the following flowers:
    - Daisy
    - Dandelion
    - Roses
    - Sunflowers
    - Tulips
    """,
    unsafe_allow_html=True,
)

# Upload image
uploaded_file = st.file_uploader("Choose a flower image...", type="jpg")

# If image is uploaded
if uploaded_file is not None:
    # Make predictions
    img_array = process_image(uploaded_file)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # Display the uploaded image with a fixed size
    st.image(Image.fromarray(np.squeeze(img_array.numpy().astype("uint8"))),
             caption="Uploaded Image.", width=400)

    # Prediction section
    st.subheader("Prediction:")
    st.write(
        "This image most likely belongs to **{}** with a **{:.2f}%** confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

    # Class probabilities section
    st.subheader("Class Probabilities:")
    st.markdown("<div style='margin-top: 05px; padding: 05px; background-color: #f0f0f0; border-radius: 8px;'>", unsafe_allow_html=True)
    # for class_name, prob in zip(class_names, score):
    #     st.markdown(
    #         "{}: **{:.2f}%**".format(class_name, 100 * prob),
    #         unsafe_allow_html=True,
    #     )
    st.markdown("</div>", unsafe_allow_html=True)

    # Create and display the table
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot the image on the first subplot
    ax1.imshow(np.squeeze(img_array.numpy().astype("uint8")))
    ax1.axis('off')

    # Create a table on the second subplot
    table_data = [(class_name, f'{prob:.2%}') for class_name, prob in zip(class_names, score)]
    table = ax2.table(cellText=table_data, colLabels=['Class', 'Probability'], cellLoc='center', loc='center',
                        cellColours=[['lightgray']*2]*len(class_names),colColours=['lightblue', 'lightgreen'])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(0.9, 3.5)  # Adjust the table size

    # Hide axes
    ax2.axis('off')

    # Display the table in Streamlit
    st.pyplot(fig)

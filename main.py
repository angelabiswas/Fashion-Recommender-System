import streamlit as st
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import pickle

# Initialize or load feature list and filenames
if os.path.exists('embeddings.pkl') and os.path.exists('filenames.pkl'):
    # Load precomputed embeddings and filenames
    with open('embeddings.pkl', 'rb') as f:
        feature_list = pickle.load(f)
    with open('filenames.pkl', 'rb') as f:
        filenames = pickle.load(f)
else:
    # If not precomputed, initialize empty lists
    feature_list = []
    filenames = []

# Initialize the model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

st.title('Fashion Recommender System')

# Function to save uploaded file
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        st.error(f"Error: {e}")
        return 0

# Function to extract features
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features, feature_list):
    # Nearest neighbors search
 # Process the image
 
   neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
   neighbors.fit(feature_list)

   distances, indices = neighbors.kneighbors([features])
   return indices

# File upload
uploaded_file = st.file_uploader('Choose an image')

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display uploaded image
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        
        # Extract features of the uploaded image
        features = feature_extraction(os.path.join('uploads', uploaded_file.name), model)
        # st.text(features)
        # recommendation
        indices = recommend(features, feature_list)
        # show
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
    else:
        st.header('An error occurred while uploading the file.')
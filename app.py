#!pip install streamlit
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np

detector = MTCNN()
model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
feature_list = pickle.load(open('embedding.pkl','rb'))
filenames = pickle.load(open('filenames.pkl','rb'))


def extract_features(img, model, detector):
    #img = cv2.imread(img_path)
    results = detector.detect_faces(img)

    x, y, width, height = results[0]['box']

    face = img[y:y + height, x:x + width]

    #  extract its features
    image = Image.fromarray(face)
    image = image.resize((224, 224))

    face_array = np.asarray(image)

    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()

    return result

def recommend(feature_list,features):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

st.title('Which bollywood celebrity are you?')

uploaded_image = st.file_uploader('Choose an image')

if uploaded_image is not None:
        # load the image
    display_image = Image.open(uploaded_image)
    sample_image = cv2.imread(uploaded_image)
        # extract the features
    #features = extract_features(uploaded_image, model, detector)

    results = detector.detect_faces(sample_image)

    x, y, width, height = results[0]['box']

    face = sample_image[y:y + height, x:x + width]

    #  extract its features
    image = Image.fromarray(face)
    image = image.resize((224, 224))

    face_array = np.asarray(image)

    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    features = model.predict(preprocessed_img).flatten()

        # recommend
    index_pos = recommend(feature_list,features)
    predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
        # display
    col1,col2 = st.beta_columns(2)

    with col1:
        st.header('Your uploaded image')
        st.image(display_image)
    with col2:
        st.header("Seems like " + predicted_actor)
        st.image(filenames[index_pos],width=300)
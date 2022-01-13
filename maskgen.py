import streamlit as st
import pickle

import imageio
import numpy as np
import time

from skimage import measure

import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


@st.cache
def load_model():
    return pickle.load(open('model_v1.0.sav', 'rb'))


@st.cache(suppress_st_warning=True)
def make_prediction(image):
    # Image preprocesing for mask prediction
    image_arr = tf.keras.utils.img_to_array(image)
    HEIGHT = image_arr.shape[0]
    WIDTH = image_arr.shape[1]
    image_arr = tf.image.resize(image_arr, (512, 704))
    if image_arr.shape[2] == 1:
        image_arr = np.concatenate([image_arr, image_arr, image_arr], axis=2)
    image_arr = np.reshape(image_arr, (1, 512, 704, 3))

    # prediction
    model = load_model()
    mask_pred = model(image_arr)[0]
    mask_pred = tf.image.resize(mask_pred, (HEIGHT, WIDTH))
    return mask_pred


@st.cache(suppress_st_warning=True)
def generate_binary_masks(arr):
    image_list = []
    for i in range(19):
        thresh = 0.05+i*0.05
        mask_thresh = (arr >= thresh).astype(np.uint8)
        mask = tf.keras.utils.array_to_img(mask_thresh)
        image_list.append(mask)
    return image_list


st.title('Cell Segmentation')

# User inputs
st.sidebar.header('User input features')
image_type = st.sidebar.radio(
    'What image do you want to see?',
    options=['Original', 'Soft mask',
             'Binary mask', 'Colored mask']
)

uploaded_file = st.file_uploader(
    'Upload an image',
    ['png', 'jpg']
)
if uploaded_file is not None:
    upload = uploaded_file.read()
    image = imageio.imread(upload)

    if image_type == 'Original':
        st.image(upload)

    elif image_type == 'Soft mask':
        mask_pred = make_prediction(image)
        soft_mask = tf.keras.utils.array_to_img(mask_pred)
        st.image(soft_mask)

    else:
        threshold = st.sidebar.slider(
            'Binary classification threshold', min_value=0.05, max_value=0.95, step=0.05, value=0.5)
        if image_type == 'Binary mask':
            mask_pred = make_prediction(image)
            mask_thresh = (mask_pred >= threshold).astype(np.uint8)
            mask = tf.keras.utils.array_to_img(mask_thresh)
            st.image(mask)
        else:
            mask_pred = make_prediction(image)
            mask_thresh = (mask_pred >= threshold).astype(np.uint8)
            components = measure.label(np.array(mask_thresh), background=0)
            components_list = np.unique(components)
            color_list = []
            for i in components_list:
                color_list.append(np.random.choice(range(256), size=3))
            HEIGHT = components.shape[0]
            WIDTH = components.shape[1]
            colored_arr = np.zeros((HEIGHT, WIDTH, 3))
            for i in range(HEIGHT):
                for j in range(WIDTH):
                    if components[i, j, 0] != 0:
                        colored_arr[i, j] = color_list[components[i, j, 0]]
            colored_mask = tf.keras.utils.array_to_img(colored_arr)
            st.image(colored_mask)

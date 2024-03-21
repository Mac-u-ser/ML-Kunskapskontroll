import streamlit as st
from PIL import Image
from io import BytesIO
import base64

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import bz2file as bz2
import pickle

# convert to RGB
def rgba2rgb( rgba, background=(255,255,255) ):
    '''rgba2rgb converts image to RGB from RGBA'''
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8' )    
# load
data_f = bz2.BZ2File('model.pbz2', 'rb')
extra_trees_clf_400 = pickle.load(data_f)


st.set_page_config(layout="wide", page_title="Which digit are you?")

st.write("## Which digit are you?")
st.write(
    ":dog: Try uploading an image to find out which digit is hidden in there.:grin:"
)
st.sidebar.write("## Upload :gear:")

MAX_FILE_SIZE = 7 * 1024 * 1024  # 7MB

col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["jpg", "webp"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        col1.write("Original Image :camera:")
        col1.image(my_upload)
        image_upload = Image.open(my_upload)
        my_upload = np.array(image_upload)
        img = np.dot(rgba2rgb(my_upload), [0.299 , 0.587, 0.114])
        n, m = img.shape
        window_n=n//28
        window_m=m//28
        n, m = n - n % 28, m - m % 28
        img1 = np.zeros((28,28))
        X_job = np.zeros((1,28*28))
        ind = 0
        for x in range(0, n, window_n):
            for y in range(0, m-1, window_m):
                threshold = img[x:x+window_n,y:y+window_m].mean() 
                img1[x//window_n,y//window_m] = threshold
                X_job[0,ind] = 255 - threshold
                ind = ind + 1
        col2.image(img1/255)
        col2.write("Digit hidden in the image is: ")
        col2.write(extra_trees_clf_400.predict(X_job))
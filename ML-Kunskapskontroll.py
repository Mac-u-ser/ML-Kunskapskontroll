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
    
# load
data_f = bz2.BZ2File('model.pbz2', ‘rb’)
extra_trees_clf_400 = pickle.load(data_f)


st.set_page_config(layout="wide", page_title="Which digit are you?")

st.write("## Which digit are you?")
st.write(
    ":dog: Try uploading an image to find out which digit is hidden in there.:grin:"
)
st.sidebar.write("## Upload :gear:")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["jpg", "webp"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        fix_image(upload=my_upload)
    image = Image.open(upload)
    col1.write("Original Image :camera:")
    col1.image(image)

    img = np.dot(rgba2rgb(image), [0.299 , 0.587, 0.114])
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
        	img1[x//window_n,y//window_m] = 255 - threshold
        	X_job[0,ind] = 255 - threshold
        	ind = ind + 1
			plt.imshow(img1, cmap=mpl.cm.binary)
			print(f'Digit hidden in the image is: {extra_trees_clf_400.predict(X_job)}')
 #   col2.image(fixed)
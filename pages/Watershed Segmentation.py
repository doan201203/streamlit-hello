import streamlit as st
import cv2 as cv
from PIL import Image

st.title('Watershed Segmentation')

st.header('Phân đoạn kí tự biển số xe với Watershed')
st.subheader('1. Training')

images_train = [
  cv.imread('../datasets/biensoxe/train/images/1xemay322.jpg'),
  cv.imread('../datasets/biensoxe/train/images/1xemay362.jpg'),
  Image.open('../datasets/biensoxe/train/images/1xemay322.jpg'),
  Image.open('../datasets/biensoxe/train/images/1xemay362.jpg'),
]

images_test = [
  cv.imread('../datasets/biensoxe/test/images/2xemay103.jpg'),
  cv.imread('../datasets/biensoxe/test/images/2xemay142.jpg'),
  Image.open('../datasets/biensoxe/test/images/2xemay103.jpg'),
  Image.open('../datasets/biensoxe/test/images/2xemay142.jpg'),
]

labels_train = [
  cv.imread('../datasets/biensoxe/train/labels/1xemay322.png', cv.IMREAD_GRAYSCALE),
  cv.imread('../datasets/biensoxe/train/labels/1xemay362.png', cv.IMREAD_GRAYSCALE),
  Image.open('../datasets/biensoxe/train/labels/1xemay322.png'),
  Image.open('../datasets/biensoxe/train/labels/1xemay362.png'),
]

labels_test = [
  cv.imread('../datasets/biensoxe/test/labels/2xemay103.png', cv.IMREAD_GRAYSCALE),
  cv.imread('../datasets/biensoxe/test/labels/2xemay142.png', cv.IMREAD_GRAYSCALE),
  Image.open('../datasets/biensoxe/test/labels/2xemay103.png'),
  Image.open('../datasets/biensoxe/test/labels/2xemay142.png'),
]

#display anh tap train
st.subheader('1.1. Training images')
st.caption('Original images')
col = st.columns(2)
col[0].image(images_train[2], caption='1xemay322.jpg', channels='BGR', use_column_width=True)
col[1].image(images_train[3], caption='1xemay362.jpg', channels='BGR', use_column_width=True)
st.caption('Ground truth')
col = st.columns(2)
col[0].image(labels_train[2], caption='1xemay322.png', channels='BGR', use_column_width=True)
col[1].image(labels_train[3], caption='1xemay362.png', channels='BGR', use_column_width=True)


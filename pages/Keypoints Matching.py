import streamlit as st
import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image

@st.cache_resource(show_spinner=False)
def load_data():
  return np.load('./datasets/sythetic/keypoints_matching/mc_data.npy', allow_pickle=True)

data_set = load_data()

def display_dataset():
    col1 = st.columns(4)
    col2 = st.columns(4)
    for i in range(4):
        col1[i].image(data_set[i], use_column_width=True)
    for i in range(4, 8):
        col2[i-4].image(data_set[i], use_column_width=True)
st.set_page_config(page_title="Keypoints Matching", layout="wide")
st.header('1. Dataset', divider=True)
st.markdown("""
            - Trích từ tập dữ liệu được giới thiệu trước đó ở [Semantic Keypoints](truongdoan.streamlit.app/Semantic_Keypoints) sao cho SIFT hoặc ORB đạt 100% về phát hiện Keypoints (theo Ground Truth)
            - Với SIFT: 59 ảnh.
            - Với ORB: 110 ảnh.
            - ORB & SIFT: 3 ảnh.
            - Một số hình ảnh trong tập dữ liệu.
            """)

display_dataset()
st.header('2. Methods')
st.markdown("""
            -  [Brute-Force Matcher](https://www.researchgate.net/publication/328991586_Image_Feature_Matching_and_Object_Detection_Using_Brute-Force_Matchers) là một thuật toán trong OpenCV được sử dụng để so khớp các đặc trưng (features) giữa hai hình ảnh.
            """)


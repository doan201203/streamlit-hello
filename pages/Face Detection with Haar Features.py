from __future__ import print_function
import streamlit as st
import cv2 as cv
import os
from PIL import Image 
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="Face Detection với Haar Features", initial_sidebar_state="expanded")
st.title("Face Detection với Haar Features")

st.header("1. Huấn luyện model")
st.subheader("1.1. Tập dữ liệu")
st.markdown(
        """- Gồm 400 ảnh (faces) 24x24 thu thập từ ORL dataset, 400 ảnh (non-faces) 24x24 lấy nhiều nguồn trên internet. Một số hình ảnh mô tả tập dữ liệu được minh họa ở (hình 1).
        """
)

@st.cache_data(show_spinner=False)
def display_dataset():
  neg = './datasets/faces_and_non_faces_data/n/'
  pos = './datasets/faces_and_non_faces_data/p/'

  #take 10 images from each folder
  neg = os.listdir(neg)
  pos = os.listdir(pos)
  neg = np.array(neg[:10])
  pos = np.array(pos[:10])
  
  all = np.concatenate((pos, neg))
  print(all)
  fig, axs = plt.subplots(2, 10, figsize=(20, 5))
  fig.suptitle('Hình 1: Một số hình ảnh trong tập dữ liệu')
  #turn off xticks and yticks
  for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])
  
  #set height between subplots row
  # plt.subplots_adjust(hspace=0)
  for tick, ax in enumerate(axs.flat):
    # print(ax)
    if tick < 10:
      ax.imshow(cv.imread(os.path.join('./datasets/faces_and_non_faces_data/p', pos[tick])))
    else:
      ax.imshow(cv.imread(os.path.join('./datasets/faces_and_non_faces_data/n', neg[tick-10])))
    
  return fig
  # st.pyplot(fig)
  # plt.show()
st.pyplot(display_dataset())

st.subheader("1.2. Huấn luyện")
st.markdown(
        """
           ### Tham số huấn luyện với cv::train_cascades:
            - numStage: 5.
            - minHitRate: 0.995 (cứ mỗi 1000 ảnh cho rằng là ảnh có khuôn mặt phải đúng 995 ảnh).
            - maxFalseAlarmRate: 0.5 (cứ mỗi 1000 ảnh cho rằng là ảnh non-face chỉ được sai 500 ảnh.

        """
)  

haar_features = []

def extract_ft(img):
    features = []
    image = cv.resize(cv.cvtColor(img, cv.COLOR_BGR2GRAY), (24, 24))
    pref = cv.integral(image)
    for feature in haar_features:
        re = 0
        for rec in feature:
            i, j, w, h, weight = rec
            re += weight * (pref[j + h][i + w] + pref[j][i] - pref[j][i + w] - pref[j + h][i])
        features.append(re)
    return np.asarray(features)
  
X, y = [], []

with open('./miscs/X.pkl', 'rb') as f:
    X = pickle.load(f)
with open('./miscs/y.pkl', 'rb') as f:
    y = pickle.load(f)
# print(X)
@st.cache_data(show_spinner=False)
def load_data():
  with open('./miscs/X.pkl', 'rb') as f:
    X = pickle.load(f)
  with open('./miscs/y.pkl', 'rb') as f:
    y = pickle.load(f)
  with open('./miscs/haar_features.pkl', 'rb') as f:
    haar_features = pickle.load(f)
  return X, y, haar_features
X, y, haar_features = load_data()

st.header("2. Phát hiện khuôn mặt")

def detect_faces_multiscale(image, classifier, scaleFactor=1.1, minNeighbors=3, minSize=(24, 24), stepSize=10):
    faces = []
    h, w = image.shape[:2]
    
    # Vòng lặp scale ảnh (pyramid scaling)
    scale = 1.0
    while scale > 0.1:
        print(scale)
        # if (int(w * scale) > 0 and int(h * scale) > 0):
        resized_img = cv.resize(image, (int(w * scale), int(h * scale)))

        # Di chuyển cửa sổ quét với kích thước cố định minSize
        for x in range(0, resized_img.shape[1] - minSize[0], stepSize):
            for y in range(0, resized_img.shape[0] - minSize[1], stepSize):
                # Cắt ảnh theo cửa sổ quét
                window = resized_img[y:y + minSize[1], x:x + minSize[0]]
                
                # Trích xuất đặc trưng từ cửa sổ
                features = extract_ft(window)
                
                # Dự đoán nếu là khuôn mặt
                if classifier.predict([features])[0] == 1:
                    # Lưu lại tọa độ khuôn mặt trên ảnh gốc
                    real_x = int(x / scale)
                    real_y = int(y / scale)
                    real_w = int(minSize[0] / scale)
                    real_h = int(minSize[1] / scale)
                    faces.append((real_x, real_y, real_w, real_h))

        # Giảm kích thước ảnh theo scaleFactor
        scale /= scaleFactor
 # Lọc các khuôn mặt bằng cách kiểm tra các neighbors
    return faces

file = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png"])
if file is not None:
    # choose k 
    k = st.slider('Chọn k (cho kNN)', 1, 31, 21, 2)
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X, y)
    image = Image.open(file)
    st.image(image, caption="Ảnh gốc", use_column_width=True)
    image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
    img = image.copy()
    # print(img.shape)
    with st.spinner('Đang xử lí ...'):
        faces = detect_faces_multiscale(img, classifier, 1.3, 5, (24, 24), 15)
    for x, y, w, h in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    with st.expander("Ảnh sau khi phát hiện khuôn mặt", expanded=False):
      st.image(img, caption="Ảnh sau khi phát hiện khuôn mặt", use_column_width=True, channels="BGR")


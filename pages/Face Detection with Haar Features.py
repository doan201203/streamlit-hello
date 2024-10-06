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
import algorithm.facedetecthaar as fdh
from my_utils.metrics import iou
import pandas as pd
import altair as alt

st.set_page_config(page_title="Face Detection với Haar Features", initial_sidebar_state="expanded", layout="wide")
st.title("Face Detection với Haar Features")

### TRAINING
st.header("1. Thiết lập thí nghiệm")
st.subheader("1.1. Training images")
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

st.pyplot(display_dataset())

st.subheader("1.2. Testing images")
st.markdown(
  """ 
    - Gồm 10 ảnh thu thập từ nhiều nguồn trên internet được detect bằng mô hình Haar Classifier mặc định do OpenCV cung cấp.
  """
)

# @st.cache_data(show_spinner=False)
def bbox_to_rect(pos, label):
  img = cv.imread(pos)
  height, width = img.shape[:2]
  with open(label, 'r') as f:
    #convert yolov8 format to x, y, w, h
    for line in f:
      line = line.split()
      x, y, w, h = map(int, line)
      print(x, y, w, h)
      cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 4)
  return img

def display_testimg():
  test_img_dir = './datasets/faces_and_non_faces_data/test/'
  imgs = os.listdir(os.path.join(test_img_dir, 'images'))
  labels = os.listdir(os.path.join(test_img_dir, 'annotations'))
  imgs.sort()
  labels.sort()
  
  for i in range(2):
    col = st.columns(5)
    for j in range(5):
      pos = os.path.join(test_img_dir, 'images', imgs[i*5+j])
      label = os.path.join(test_img_dir, 'annotations', labels[i*5+j])
      ii = bbox_to_rect(pos, label)
      #resize all ii with same size to display
      ii = cv.resize(ii, (512, 512)) 
      col[j].image(ii, channels="BGR", use_column_width=True)
display_testimg()

st.subheader("1.3. Huấn luyện với Cascade Classifier")
st.markdown(
        """
          - Tham số:
            - numStage: 5.
            - minHitRate: 0.995 (cứ mỗi 1000 ảnh cho rằng là ảnh có khuôn mặt phải đúng 995 ảnh).
            - maxFalseAlarmRate: 0.5 (cứ mỗi 1000 ảnh cho rằng là ảnh non-face chỉ được sai 500 ảnh.
            - width: 24, height: 24. (Chiều rộng và chiều cao của ảnh).
          - Kết quả:
            - Tổng số lượng features: 14
            - Số lượng features cho từng stages lần lượt là: {3, 2, 3, 3, 3} 
        """
)  

st.subheader("1.4. Huấn luyện kNN")
st.markdown(
    """
        - Tham số:
            - Metric: {'euclidean', 'cosine', 'manhattan'} dùng để tính toán khoảng cách.
            - Tham số k: {1, 5, 9, ..., 49}
        - Độ đo sử dụng để đánh giá kết quả:
            - Intersection over Union:\n
          [![Metric](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSHKmi5ITcexRNHETgdO4b5jfjvC6QZ1YbL6w&s)]()
        - Kết quả:

    """
)

  
# @st.cache_data(show_spinner=False)
def load_data(file_path):
  haar_feat = fdh.HaarFeature(file_path)
  return haar_feat


#TESTING

def table_result():
  with open('./miscs/knn_result.pkl', 'rb') as f:
    p = pickle.load(f)  

  tb = pd.DataFrame(p)

  st.columns(3)[1] = st.write("Tham số k và metric tốt nhất: ", tb.loc[tb['Average IoU'].idxmax()])
  ch = alt.Chart(tb).mark_line().encode(
                                          alt.X('k', title='Tham số k'),
                                          alt.Y('{}'.format('Average IoU'), title='Average {}'.format('IoU')),
                                          color='metric:N',
                                      ).properties(
                                          title='Biểu đồ sự thay đổi của {} theo tham số k'.format('IoU'),
                                      )
  st.altair_chart(ch, use_container_width=True)
  
  st.markdown("""
                - Một số kết quả detect được trên tập dữ liệu testing:
              """)
  for i in range(2):
    col = st.columns(5)
    for j in range(5):
      test_img_dir = './datasets/faces_and_non_faces_data/test/'
      imgs = os.listdir(os.path.join(test_img_dir, 'images'))
      labels = os.listdir(os.path.join(test_img_dir, 'annotations'))
      imgs.sort()
      labels.sort()
      img = cv.imread(os.path.join(test_img_dir, 'images', imgs[i*5+j]))
      with open(os.path.join(test_img_dir, 'annotations', labels[i*5+j]) + 'txd', 'r') as f:
        for line in f:
          line = line.split()
          x, y, w, h = map(int, line)
          cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 4)
      #write all faces to file
      img = cv.resize(img, (512, 512))
      col[j].image(img, channels="BGR", use_column_width=True)    
  
  
table_result()

# tb = {
#   'metric': [],
#   'k': [],
#   'Average IoU': []
# }

# def train():
#   for k in range(1, 52, 4):
#     for metric in ['euclidean', 'cosine', 'manhattan']:
#       haar_features = load_data('./datasets/faces_and_non_faces_data/output/cascade.xml')
#       X, y = haar_features.X, haar_features.y
#       classifier = KNeighborsClassifier(n_neighbors=k, metric=metric, weights='distance')
#       classifier.fit(X, y)
#       ave = 0
#       print(k, metric)
#       for i in range(10):
#         test_img_dir = './datasets/faces_and_non_faces_data/test/'
#         imgs = os.listdir(os.path.join(test_img_dir, 'images'))
#         labels = os.listdir(os.path.join(test_img_dir, 'annotations'))
#         imgs.sort()
#         labels.sort()
#         img = cv.imread(os.path.join(test_img_dir, 'images', imgs[i]))
#         gray = img.copy()
#         if len(img.shape) == 3:
#           gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#         # print(k, metric, imgs[i])
#         faces = haar_features.detect_faces_multiscale(gray, classifier, 1.3, 0, (24, 24), 24)
#         mark1 = np.zeros(img.shape[:2], dtype=np.uint8)
#         mark2 = np.zeros(img.shape[:2], dtype=np.uint8)
#         with open(os.path.join(test_img_dir, 'annotations', labels[i]), 'r') as f:
#           for line in f:
#             line = line.split()
#             x, y, w, h = map(int, line)
#             #fill rectange with white color
#             mark1[y:y+h, x:x+w] = 255
#             # cv.rectangle(mark1, (x, y), (x+w, y+h), 255, -1)
#         for x, y, w, h in faces:
#           mark2[y:y+h, x:x+w] = 255
#           # cv.rectangle(mark2, (x, y), (x+w, y+h), 255, -1)
#         iou_score = iou(mark1, mark2)
#         ave += iou_score
#       print("SUCCSED", ave)
#       tb['metric'].append(metric)
#       tb['k'].append(k)
#       tb['Average IoU'].append(ave/10)       
#   with open('knn_result.pkl', 'wb') as f:
#     pickle.dump(tb, f)
# train()


st.header("2. Phát hiện khuôn mặt")

file = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png"])
if file is not None:
    # choose k 
    k = st.slider('Chọn k (cho kNN) ', 1, 49, 1, 2)
    print(k)
    haar_features = load_data('./datasets/faces_and_non_faces_data/output/cascade.xml')
    X, y = haar_features.X, haar_features.y
    classifier = KNeighborsClassifier(n_neighbors=k, metric='euclidean', weights='distance')
    classifier.fit(X, y)
    image = Image.open(file)
    st.image(image, caption="Ảnh gốc", use_column_width=True)
    image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)

    #gray
    gray = image.copy()
    if len(image.shape) == 3:
      gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    with st.spinner('Đang xử lí ...'):
        faces = haar_features.detect_faces_multiscale(gray, classifier, 1.3, 0, (24, 24), 24)
    for x, y, w, h in faces:
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    with st.expander("Ảnh sau khi phát hiện khuôn mặt", expanded=False):
      st.image(image, caption="Ảnh sau khi phát hiện khuôn mặt", use_column_width=True, channels="BGR")


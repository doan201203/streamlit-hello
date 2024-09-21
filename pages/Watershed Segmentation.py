import streamlit as st
import cv2 as cv
from PIL import Image
import pandas as pd
import altair as alt

from algorithm.watershed import (
  watershed,
  get_mask,
  _2dice,
  iou,
)
import numpy as np
st.set_page_config(layout="wide")
st.title('Thuật toán Watershed Segmentation')

st.header('Phân đoạn kí tự biển số xe với Watershed')
st.subheader('1. Training')

images_train = [
  cv.imread('./datasets/biensoxe/train/images/1xemay322.jpg'),
  cv.imread('./datasets/biensoxe/train/images/1xemay362.jpg'),
]

images_test = [
  cv.imread('./datasets/biensoxe/test/images/2xemay103.jpg'),
  cv.imread('./datasets/biensoxe/test/images/2xemay142.jpg'),
]

labels_train = [
  cv.imread('./datasets/biensoxe/train/labels/1xemay322.png', cv.IMREAD_GRAYSCALE),
  cv.imread('./datasets/biensoxe/train/labels/1xemay362.png', cv.IMREAD_GRAYSCALE),
]

labels_test = [
  cv.imread('./datasets/biensoxe/test/labels/2xemay103.png', cv.IMREAD_GRAYSCALE),
  cv.imread('./datasets/biensoxe/test/labels/2xemay142.png', cv.IMREAD_GRAYSCALE),
]

labels_train[0][labels_train[0] != 0] = 255
labels_train[1][labels_train[1] != 0] = 255
labels_test[0][labels_test[0] != 0] = 255
labels_test[1][labels_test[1] != 0] = 255

#display anh tap train
st.subheader('1.1. Training images')
col = st.columns(4)
col[0].image(images_train[0], caption='1xemay322 Original', channels='BGR', use_column_width=True)
col[1].image(images_train[1], caption='1xemay362 Original', channels='BGR', use_column_width=True)

col[2].image(labels_train[0], caption='1xemay322 Ground truth', use_column_width=True)
col[3].image(labels_train[1], caption='1xemay362 Ground truth', use_column_width=True)

threshs = np.arange(0, 1, 1/50)
kernel_sizes = np.arange(3, 10, 2)

#display parameters
st.subheader('1.2. Thiết lập thí nghiệm')
st.markdown(
    """
        ##### Các tham số:
        - Kernel Size : [(3, 3); (5, 5); (7, 7); (9, 9)]. Sử dụng trong các toán tử morphological.
        - Thresh: [0, 0.02, 0.04, ..., 1]. Sử dụng để phân ngưỡng foreground. 
        ##### Độ đo:
        - Dice Coefficient:          [![Metric](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQxRLmy2JV3U80ZCKO_PGDNd8UZeOB3XSAQAw&s)]()
    """
)

image_result = {}
ave_dice = {}

#@st.cache_data(show_spinner=False)
def train_phase(save_rs):
  global prog
  s = 0
  best_params = None
  for j in kernel_sizes:
    all_ave = []
    per_kernel = {}
    for i in threshs:
      ave = 0
      all_mask = []
      for k in range(len(images_train)):
        img, mask = watershed(images_train[k], j, i)
        mask = get_mask(mask)
        all_mask.append(mask)
        dice = _2dice(mask, labels_train[k])
        ave += dice
      per_kernel[i] = all_mask
      ave /= len(images_train)
      all_ave.append(ave)
      if best_params is None or ave > best_params[0]:
        best_params = (ave, i, j)
      s += 1     
      prog.progress(s / tot, f'Training... {round(s/tot*100)}%')
    if save_rs:
      image_result[j] = per_kernel    
    ave_dice[j] = all_ave
  return best_params

def visualize_result():
  numRow = (len(kernel_sizes) + 1) // 2
  for i in range(numRow):
    cols = st.columns(2)
    for j in range(2):
      if i * 2 + j >= len(kernel_sizes):
        continue
      ch = alt.Chart(pd.DataFrame({'Threshold': threshs, 'Average Dice Coefficient': ave_dice[kernel_sizes[i * 2 + j]]})).mark_line(interpolate='basis').encode(
        alt.X('Threshold', title='Threshold'),
        alt.Y('Average Dice Coefficient', title='Average Dice Coefficient'),
        # color=
      ).properties(
        width=670,
        title='Biểu đồ sự thay đổi của Dice Coefficient theo Threshold với Kernel Size: ({}, {})'.format(kernel_sizes[i * 2 + j], kernel_sizes[i * 2 + j]),
      )
      
      cols[j].altair_chart(ch)

col = st.columns(2)
bp = None

# if col[0].button('Bấm vào đây để bắt đầu huấn luyện'):
tot = len(threshs) * len(kernel_sizes)

with st.status('Đang thực hiện...', expanded=True) as sts:
  prog = st.progress(0)    
  bp = train_phase(True)
  st.write('- Tham số tốt nhất:')
  st.write('Average Dice Coefficient:', bp[0])
  st.write('Threshold:', bp[1])
  st.write('Kernel Size:', bp[2])
  sts.update(label='Hoàn thành!', state='complete', expanded=True)
  
  
  #other results
st.subheader('1.3. Kết quả')

st.write('- Điều chỉnh các thông số ở sidebar để hiển thị các kết quả khác')
st.sidebar.header('Parameters')
thresh = st.sidebar.slider('Threshold', 0.0, 1.0, bp[1], 0.02, key="thresh_slider")
kernel_size = st.sidebar.slider('Kernel size', 3, 9, bp[2], 2, key="kernel_slider")

col = st.columns(2)
col[0].image(
  image_result[kernel_size][thresh][0],
  caption='Dice Coefficient = {}'.format(_2dice(image_result[kernel_size][thresh][0],
  labels_train[0])),
  use_column_width=True
)
col[1].image(
  image_result[kernel_size][thresh][1],
  caption='Dice Coefficient = {}'.format(_2dice(image_result[kernel_size][thresh][1],
  labels_train[1])),
  use_column_width=True
)

visualize_result()
    
print(st.session_state)

st.subheader('2. Testing')

#display anh tap test
st.subheader('2.1. Testing images')
col = st.columns(4)
col[0].image(images_test[0], caption='Original 2xemay103', channels='BGR', use_column_width=True)
col[1].image(images_test[1], caption='Original 2xemay142', channels='BGR', use_column_width=True)

col[2].image(labels_test[0], caption='Ground truth 2xemay103', use_column_width=True)
col[3].image(labels_test[1], caption='Ground truth 2xemay142', use_column_width=True)

st.subheader('2.2. Kết quả')
st.caption('Tham số sử dụng:')
st.write('Threshold:', bp[1])
st.write('Kernel Size:', bp[2])

#apply to test images
st.caption('Hình ảnh sau khi áp dụng thuật toán :')
col = st.columns(2)
cost = 0
for i in range(len(images_test)):
  img, mask = watershed(images_test[i], bp[2], bp[1])
  mask = get_mask(mask)
  c = _2dice(mask, labels_test[i])
  col[i].image(mask, caption='Dice Coefficient = {}'.format(c), use_column_width=True)
  cost += c 
st.write('Average Dice Coefficient:', cost / len(images_test))
  
  

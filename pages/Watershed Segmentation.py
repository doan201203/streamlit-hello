import streamlit as st
import cv2 as cv
from PIL import Image
import pandas as pd
import altair as alt
import pickle
import os

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
col = st.columns(2)
col[0].image(images_train[0], caption='1xemay322.jpg', channels='BGR', use_column_width=True)
col[1].image(images_train[1], caption='1xemay362.jpg', channels='BGR', use_column_width=True)

threshs = np.arange(0.0, 0.3+0.00001, 0.02)
kernel_sizes = np.arange(3, 8, 2)

#display parameters
st.subheader('1.2. Thiết lập thí nghiệm')
st.markdown(
    """
        ##### Các tham số:
        - Kernel Size : [(3, 3); (5, 5); (7, 7)]. Sử dụng trong các toán tử morphological.
        - Thresh: [0, 0.02, 0.04, ..., 0.3]. Sử dụng để phân ngưỡng foreground. 
        ##### Độ đo:
        - Dice Coefficient:          [![Metric](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQxRLmy2JV3U80ZCKO_PGDNd8UZeOB3XSAQAw&s)]()
    """
)

image_result = {}
ave_dice = {}

@st.cache_data(show_spinner=False)
def train_phase(save_rs):
  prog = st.progress(0)
  s = 0
  best_params = None
  for j in kernel_sizes:
    all_ave = []
    per_kernel = {}
    for i in threshs:
      ave = 0
      all_mask = []
      for k in range(len(images_train)):
        mask = watershed(images_train[k], j, i)
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

def make_data(thresh, kernel_size, mark):
    res = {
        'Threshold' : thresh,
        'KernelSize' : kernel_size,
        'Average Dice Coefficient' : mark,
    }
    return res
    
def visualize_result():
    #numRow = (len(kernel_sizes) + 1) // 2
  df = []
  for i in kernel_sizes:
      for j in range(len(threshs)):
        df.append(make_data(threshs[j], '({}, {})'.format(i, i), ave_dice[i][j]))
  df = pd.DataFrame(df)
  ch = alt.Chart(df).mark_line().encode(
                                        alt.X('Threshold', title='Threshold'),
                                        alt.Y('Average Dice Coefficient', title='Average Dice Coefficient'),
                                        color='KernelSize:N',
                                    ).properties(
                                        title='Biểu đồ sự thay đổi của Dice Coefficient theo Threshold',
                                    )
  st.altair_chart(ch, use_container_width=True)

col = st.columns(2)
bp = None

#create session_state
if 'load_state' not in st.session_state:
    st.session_state['load_state'] = False

#reload data
if st.session_state.load_state or (not image_result and os.path.isfile('./miscs/img_rs')):
    with open('./miscs/img_rs', 'rb') as f:
        image_result = pickle.load(f)
    with open('./miscs/ave_dice', 'rb') as f:
        ave_dice = pickle.load(f)


# if col[0].button('Bấm vào đây để bắt đầu huấn luyện'):
tot = len(threshs) * len(kernel_sizes)

with st.status('Đang thực hiện...', expanded=True) as sts:
  bp = train_phase(True)
  sts.update(label='Hoàn thành!', state='complete' if st.session_state['load_state'] else 'running', expanded=True)
  st.write('- Tham số tốt nhất:')
  st.write('Average Dice Coefficient:', bp[0])
  st.write('Threshold:', bp[1])
  st.write('Kernel Size:', bp[2])
    
  if not st.session_state['load_state']:
      with open('./miscs/img_rs', 'wb') as f:
          pickle.dump(image_result, f)
      with open('./miscs/ave_dice', 'wb') as f:
          pickle.dump(ave_dice, f)
  st.session_state['load_state'] = True
  
  
#other results
st.subheader('1.3. Kết quả')
    
st.caption('Điều chỉnh các thông số ở sidebar để hiển thị các kết quả khác')
st.sidebar.header('Parameters')
thresh = st.sidebar.slider('Threshold', 0.0, 0.3, bp[1], 0.02, key="thresh_slider")
kernel_size = st.sidebar.slider('Kernel size', 3, 7, bp[2], 2, key="kernel_slider")

st.write('* Ground Truth')

col1 = st.columns(2)
col1[0].image(
  labels_train[0],
  use_column_width=True
)
col1[1].image(
  labels_train[1],
  use_column_width=True
)

st.write('* Predict')

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
    
#testing phase
st.subheader('2. Testing')

#display anh tap test
st.subheader('2.1. Testing images')
col = st.columns(2)
col[0].image(images_test[0], caption='2xemay103.jpg', channels='BGR', use_column_width=True)
col[1].image(images_test[1], caption='2xemay142.jpg', channels='BGR', use_column_width=True)

st.subheader('2.2. Kết quả')
st.caption('Tham số sử dụng:')
st.write('Threshold:', bp[1])
st.write('Kernel Size:', bp[2])

#apply to test images
st.caption('Hình ảnh sau khi áp dụng thuật toán :')
cost = 0

st.write('* Ground Truth')
col = st.columns(2)
col[0].image(labels_test[0], use_column_width=True)
col[1].image(labels_test[1], use_column_width=True)

st.write('* Predict')
col = st.columns(2)
for i in range(len(images_test)):
  mask = watershed(images_test[i], bp[2], bp[1])
  mask = get_mask(mask)
  c = _2dice(mask, labels_test[i])
  col[i].image(mask, caption='Dice Coefficient = {}'.format(c), use_column_width=True)
  cost += c 
st.write('Average Dice Coefficient:', cost / len(images_test))
  
  

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
)

from utils.metrics import (
    iou,
    _2dice,
)

import numpy as np
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
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

METRIS = [
  {
    'name': 'Dice Coefficient',
    'function': _2dice,
    'image': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQxRLmy2JV3U80ZCKO_PGDNd8UZeOB3XSAQAw&s'
  },
  {
    'name': 'Intersection over Union',
    'function': iou,
    'image': 'https://cdn.prod.website-files.com/5d7b77b063a9066d83e1209c/647a0de43f2d758b954db3f2_IoU%20formula.webp'
  },
]

# Convert to black-white color 
labels_train[0][labels_train[0] != 0] = 255
labels_train[1][labels_train[1] != 0] = 255
labels_test[0][labels_test[0] != 0] = 255
labels_test[1][labels_test[1] != 0] = 255

# Display train image set
st.subheader('1.1. Training images')
col = st.columns(2)
col[0].image(images_train[0], caption='1xemay322.jpg', channels='BGR', use_column_width=True)
col[1].image(images_train[1], caption='1xemay362.jpg', channels='BGR', use_column_width=True)

threshs = np.arange(0.0, 0.3+0.00001, 0.02)
kernel_sizes = np.arange(3, 8, 2)

# Display hyperparameters
st.subheader('1.2. Thiết lập thí nghiệm')
st.markdown(
    """
        ##### Các tham số:
        - Kernel Size : [(3, 3); (5, 5); (7, 7)]. Sử dụng trong các toán tử morphological.
        - Thresh: [0, 0.02, 0.04, ..., 0.3]. Sử dụng để phân ngưỡng foreground. 
        ##### Độ đo:
        - Dice Coefficient:\n
          [![Metric](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQxRLmy2JV3U80ZCKO_PGDNd8UZeOB3XSAQAw&s)]()
        - Intersection over Union:\n
          [![Metric](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSHKmi5ITcexRNHETgdO4b5jfjvC6QZ1YbL6w&s)]()
    """
)

# Select metric
metric = st.radio('', METRIS, format_func=lambda x: x['name'], horizontal=True, key='metric')
st.write('Độ đo được chọn:', metric['name'])

image_result = {}
ave_dice = {}

@st.cache_data(show_spinner=False)
def train_phase(save_rs, metricName):
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
        dice = METRIS[0 if metricName.startswith('Dice') else 1]['function'](mask, labels_train[k])
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

def make_data(thresh, kernel_size, mark,metric):
    res = {
        'Threshold' : thresh,
        'KernelSize' : kernel_size,
        metric['name'] : mark,
    }
    return res
    
def visualize_result(metric):
  df = []
  for i in kernel_sizes:
      for j in range(len(threshs)):
        df.append(make_data(threshs[j], '({}, {})'.format(i, i), ave_dice[i][j], metric))
  df = pd.DataFrame(df)
  ch = alt.Chart(df).mark_line().encode(
                                        alt.X('Threshold', title='Threshold'),
                                        alt.Y('{}'.format(metric['name']), title='Average {}'.format(metric['name'])),
                                        color='KernelSize:N',
                                    ).properties(
                                        title='Biểu đồ sự thay đổi của {} theo Threshold'.format(metric['name']),
                                    )
  st.altair_chart(ch, use_container_width=True)

col = st.columns(2)
bp = None

# Create session_state
if 'load_state' not in st.session_state:
    st.session_state['load_state'] = False

# Reload data
if not image_result:
  if metric['name'] == 'Dice Coefficient' and os.path.exists('./miscs/dice.pk'):
    with open('./miscs/dice.pk', 'rb') as f:
      image_result = pickle.load(f)
    with open('./miscs/dice', 'rb') as f:
      ave_dice = pickle.load(f)
  else:
    if os.path.exists('./miscs/iou.pk'):
      with open('./miscs/iou.pk', 'rb') as f:
        image_result = pickle.load(f)
      with open('./miscs/iou', 'rb') as f:
        ave_dice = pickle.load(f)

# Waiting training and processing data
tot = len(threshs) * len(kernel_sizes)
with st.status('Đang thực hiện...', expanded=True) as sts:
  bp = train_phase(True, metric['name'])
  sts.update(label='Hoàn thành!', state='complete' if st.session_state['load_state'] else 'running', expanded=True)
  st.write('- Tham số tốt nhất:')
  st.write('Average {}:'.format(metric['name']), bp[0])
  st.write('Threshold:', bp[1])
  st.write('Kernel Size:', bp[2])
    
  if not os.path.exists('./miscs/dice.pk') and metric['name'] == 'Dice Coefficient':
    with open('./miscs/dice.pk', 'wb') as f:
      pickle.dump(image_result, f)
    with open('./miscs/dice', 'wb') as f:
      pickle.dump(ave_dice, f)
  else:
    if not os.path.exists('./miscs/iou.pk'):
      with open('./miscs/iou.pk', 'wb') as f:
        pickle.dump(image_result, f)
      with open('./miscs/iou', 'wb') as f:
        pickle.dump(ave_dice, f)
  
# Results
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
  caption='{} = {}'.format(metric['name'],metric['function'](image_result[kernel_size][thresh][0],
  labels_train[0])),
  use_column_width=True
)
col[1].image(
  image_result[kernel_size][thresh][1],
  caption='{} = {}'.format(metric['name'], metric['function'](image_result[kernel_size][thresh][1],
  labels_train[1])),
  use_column_width=True
)

visualize_result(metric)
    
# Testing phase
st.subheader('2. Testing')

# Display testing image set
st.subheader('2.1. Testing images')
col = st.columns(2)
col[0].image(images_test[0], caption='2xemay103.jpg', channels='BGR', use_column_width=True)
col[1].image(images_test[1], caption='2xemay142.jpg', channels='BGR', use_column_width=True)

st.subheader('2.2. Kết quả')
st.caption('Tham số sử dụng:')
st.write('Threshold:', bp[1])
st.write('Kernel Size:', bp[2])

#   Apply to test images
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
  c = metric['function'](mask, labels_test[i])
  col[i].image(mask, caption='{} = {}'.format(metric['name'], c), use_column_width=True)
  cost += c 

str = 'Average {} :'.format(metric['name'])
st.write(str, cost / len(images_test))
  
  

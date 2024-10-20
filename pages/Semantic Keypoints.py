import streamlit as st
import cv2 as cv
import matplotlib.pyplot as plt
import os
import requests
from PIL import Image
import numpy as np
import altair as alt
import pandas as pd

st.set_page_config(page_title="Semantic Keypoints", layout="wide")
st.header('1. Dataset', divider=True)
st.markdown("""
                - Thu thập từ Synthetic Shapes Dataset.
                - Gồm 4000 ảnh được chia thành 8 loại hình cụ thể, với mỗi loại hình bao gồm 500 ảnh được minh họa như hình bên dưới. 
            """)

def display_dataset(fd):
    col1 = st.columns(4)
    col2 = st.columns(4)
    files = os.listdir(f'./datasets/sythetic/{fd}')
    for i in range(4):
        col1[i].image(f'./datasets/sythetic/{fd}/' + files[i], use_column_width=True, caption=files[i].split('.')[0])
    for i in range(4, 8):
        col2[i-4].image(f'./datasets/sythetic/{fd}/' + files[i], use_column_width=True, caption=files[i].split('.')[0])
    

display_dataset('sample')
st.header('2. Đánh giá kết quả dựa trên độ đo Precision và Recall', divider=True)
col2 = st.columns(2)
# result = requests.get('https://lh3.googleusercontent.com/X3iU9gn6QFMV-rMWevV2W_w562vdbdr9n-lBlVJxFDyv-XcIwR_s1ZAkZMqnmfsIjXviKKT4KoYb4HI7rp8upFUpCN7DZv39Ys5Bv-o-_RsWFT-nP-ecjqm3UxEJr98cwhhDKijvOy5obdkEekIBMLnFjQzZ5y6b-zwMOwo72L5PLUmLcdPwhl5PVcI5dw')
col2[0].image(Image.open('./datasets/sythetic/de.png'),channels='BGR', use_column_width=True)

col2[1].markdown("""
                - Precision: Tỷ lệ số keypoints dự đoán đúng trên tổng số keypoints dự đoán.
                - Recall: Tỷ lệ số keypoints dự đoán đúng trên tổng số keypoints thực tế.
                - Một dự đoán được xem là đúng nếu khoảng cách ***Manhattan*** giữa keypoint dự đoán và keypoint thực tế chênh lệch không quá 4.
                - Các tham số của ORB và SIFT được đặt mặc định.
            """)

import pickle
def make_data(lb, dat):
  return {
    'Type Shape': lb,
    'Precision': dat[0],
    'Recall': dat[1],
  }
with open('./datasets/sythetic/sift.pk', 'rb') as f:
  sift = pickle.load(f)
with open('./datasets/sythetic/orb.pk', 'rb') as f:
  orb = pickle.load(f)
print("CC", sift['draw_cube'])
df1 = []
df2 = []
for i, j in sift.items():
    # print(i, j)
    df1.append(make_data(i, j))
for i, j in orb.items():
    df2.append(make_data(i, j))

df1 = pd.DataFrame(df1)
df2 = pd.DataFrame(df2)

# data = [make_data('SIFT', sift), make_data('ORB', orb)]
# for i in sift:
    # print(i)
st.subheader('2.1. SIFT', divider=True)
st.markdown("""
        - Minh họa kết quả dựa trên thuật toán SIFT.
        """, unsafe_allow_html=True)
display_dataset('SIFT')
col2 = st.columns(2)
# draw chart for both precision and recall
ch = alt.Chart(df1).mark_bar().encode(
    x='Type Shape',
    y='Recall',
    
    color='Type Shape'
).properties(
    title='Recall của SIFT theo từng loại hình'
)
col2[0].altair_chart(ch, use_container_width=True)
ch = alt.Chart(df1).mark_bar().encode(
    x='Type Shape',
    y='Precision',
    
    color='Type Shape'
).properties(
    title='Precision của SIFT theo từng loại hình'
)
col2[1].altair_chart(ch, use_container_width=True)

st.subheader('2.2. ORB', divider=True)
st.markdown("""
            - Minh họa kết quả dựa trên thuật toán ORB.
            """, unsafe_allow_html=True)
display_dataset('ORB')
col2 = st.columns(2)
# draw chart for both precision and recall
ch = alt.Chart(df2).mark_bar().encode(
    x='Type Shape',
    y='Recall',
    
    color='Type Shape'
).properties(
    title='Recall của ORB theo từng loại hình'
)
col2[0].altair_chart(ch, use_container_width=True)
ch = alt.Chart(df2).mark_bar().encode(
    x='Type Shape',
    y='Precision',
    
    color='Type Shape'
).properties(
    title='Precision của ORB theo từng loại hình'
)
col2[1].altair_chart(ch, use_container_width=True)
st.header('3. So sánh kết quả giữa SIFT và ORB', divider=True)
st.markdown("""
            - So sánh kết quả giữa SIFT và ORB dựa trên độ đo Precision và Recall.
            """, unsafe_allow_html=True)
def make_data(lb, dat1, det):
  return {
    'Type Shape': lb,
    'Precision': dat1[0],
    'Recall': dat1[1],
    'Type': det
  }

df = []
for i, j in sift.items():
    df.append(make_data(i, j, 'SIFT'))
for i, j in orb.items():
    df.append(make_data(i, j, 'ORB'))

df = pd.DataFrame(df)
col = st.columns(2)
# draw chart for both precision and recall
ch = alt.Chart(df).mark_bar().encode(
    x='Type Shape',
    y='Recall',
    
    color='Type'
).properties(
    title='Recall của SIFT và ORB theo từng loại hình'
)
col[0].altair_chart(ch, use_container_width=True)
ch = alt.Chart(df).mark_bar().encode(
    x='Type Shape',
    y='Precision',
    
    color='Type'
).properties(
    title='Precision của SIFT và ORB theo từng loại hình'
)
col[1].altair_chart(ch, use_container_width=True)
# st.write(df)

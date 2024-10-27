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
st.header('2. Methods')

import pickle
def make_data(lb, dat):
  return {
    'Type Shape': lb.split('_')[1],
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


st.subheader('2.1. SIFT', divider=True)
col2 = st.columns(2)

col2[0].markdown("""
            - [SIFT (Scale-Invariant Feature Transform)](https://link.springer.com/article/10.1023/B:VISI.0000029664.99615.94) được thiết kế để phát hiện và mô tả các đặc trưng (features) cục bộ trong hình ảnh. Được phát triển bởi David Lowe , nó đã trở thành một trong những thuật toán được sử dụng rộng rãi nhất để phát hiện đặc trưng, nhận dạng đối tượng và khớp hình ảnh do tính mạnh mẽ của nó trong việc xử lý tỷ lệ, xoay và những thay đổi nhỏ về độ sáng hoặc góc nhìn.
            """)
col2[1].image('./datasets/sythetic/sift_pl.jpg', use_column_width=True, caption='Các bước phát hiện keypoints của SIFT')
st.markdown("""
        - Minh họa kết quả dựa trên thuật toán SIFT.
        """, unsafe_allow_html=True)
display_dataset('SIFT')
col2 = st.columns(2)

st.subheader('2.2. ORB', divider=True)
col2 = st.columns(2)
col2[0].markdown("""
            - [ORB (Oriented FAST and Rotated BRIEF)](https://ieeexplore.ieee.org/document/6126544) là một thuật toán phát hiện keypoints và mô tả đặc trưng cục bộ trong hình ảnh. ORB là một phương pháp kết hợp giữa FAST và BRIEF, được thiết kế để cung cấp một phương pháp nhanh chóng, nhỏ gọn và hiệu quả để phát hiện keypoints.
            """)
col2[1].image('./datasets/sythetic/orb_pl.png', use_column_width=True, caption='Các bước của thuật toán ORB')
st.markdown("""
            - Minh họa kết quả dựa trên thuật toán ORB.
            """, unsafe_allow_html=True)
display_dataset('ORB')

st.header('3. Evaluation', divider=True)
col2 = st.columns(2)
col2[0].image(Image.open('./datasets/sythetic/de.png'),channels='BGR', use_column_width=True)

col2[1].markdown("""
                - Precision: Tỷ lệ số keypoints dự đoán đúng trên tổng số keypoints dự đoán.
                - Recall: Tỷ lệ số keypoints dự đoán đúng trên tổng số keypoints thực tế.
                - Một dự đoán được xem là đúng nếu khoảng cách ***Euclidean*** giữa keypoint dự đoán và keypoint thực tế chênh lệch không quá 4.
                - Các tham số của ORB và SIFT được đặt mặc định.
                - Công thức tính khoảng cách ***Euclidean*** giữa 2 điểm:
                $$d = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}$$
            """)

st.header('4. So sánh kết quả giữa SIFT và ORB', divider=True)
st.markdown("""
            - So sánh kết quả giữa SIFT và ORB dựa trên độ đo Precision và Recall.
            """, unsafe_allow_html=True)
def make_data(lb, dat1, det):
  return {
    'Type Shape': lb.split('_')[1],
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
    xOffset="Type:N",
    color='Type:N'
).properties(
    title='Recall của SIFT và ORB theo từng loại hình'
)
col[0].altair_chart(ch, use_container_width=True)
ch = alt.Chart(df).mark_bar().encode(
    x='Type Shape',
    y='Precision',
    xOffset="Type:N",
    color='Type:N'
).properties(
    title='Precision của SIFT và ORB theo từng loại hình'
)
col[1].altair_chart(ch, use_container_width=True)
# st.write(df)

st.header('5. Kết luận', divider=True)
st.markdown("""
            - Có thể dễ dàng thấy được, các keypoints được ORB phát hiện chủ yếu tập trung ở các vùng chênh lệch cường độ sáng, còn SIFT thì phân bố rộng rãi hơn (xem hình dưới) vì thế kết quả của ORB trên một số tập hình có sự chênh lệch cường độ sáng như checker_board, cube cao hơn nhiều so với SIFT.
            """)

col2 = st.columns(2)
col2[0].image('./datasets/sythetic/ORB/checker_board.png', 'ORB')
col2[1].image('./datasets/sythetic/SIFT/checker_board.png','SIFT')


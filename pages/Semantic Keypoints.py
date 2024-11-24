import streamlit as st
import cv2 as cv
import matplotlib.pyplot as plt
import os
import requests
from PIL import Image
import numpy as np
import altair as alt
import pandas as pd

st.set_page_config(page_title="Semantic Keypoints")
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
with open('./datasets/sythetic/sift.pk', 'rb') as f:
  sift = pickle.load(f)
with open('./datasets/sythetic/orb.pk', 'rb') as f:
  orb = pickle.load(f)
  
st.subheader('2.1. SIFT', divider=True)
col2 = st.columns(2)

col2[0].markdown("""
            - Được công bố lần đầu ở bài báo [SIFT (Distinctive Image Features from Scale-Invariant Keypoints)](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=cc58efc1f17e202a9c196f9df8afd4005d16042a) được thiết kế để phát hiện và mô tả các đặc trưng (features) cục bộ trong hình ảnh. Được phát triển bởi David Lowe , nó đã trở thành một trong những thuật toán được sử dụng rộng rãi nhất để phát hiện đặc trưng, nhận dạng đối tượng và khớp hình ảnh do tính mạnh mẽ của nó trong việc xử lý tỷ lệ, xoay và những thay đổi nhỏ về độ sáng hoặc góc nhìn.
            """)
col2[1].image('./datasets/sythetic/sift_pl.jpg', use_column_width=True, caption='Các bước phát hiện keypoints của SIFT')
st.markdown("""
        - Minh họa kết quả dựa trên thuật toán SIFT.
        """, unsafe_allow_html=True)
display_dataset('SIFT')

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
    df.append(make_data(i, j[0], 'SIFT'))
for i, j in orb.items():
    df.append(make_data(i, j[0], 'ORB'))

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
st.markdown("""

### Nhận xét kết quả so khớp Keypoint giữa SIFT và ORB

#### 1. **Recall của SIFT và ORB**
- **ORB vượt trội hơn SIFT** ở hầu hết các loại hình, đặc biệt là:
  - **Cube**, **Star**, **Checkboard**: ORB có Recall cao (~0.58, ~0.69, ~0.4), trong khi SIFT thấp hơn đáng kể (~0.26, ~0.43, ~0.25).
  - **Polygon**: ORB đạt gần (~0.7), trong khi SIFT hầu như không có kết quả (~0.05).
- **SIFT tốt hơn ORB** trong một vài trường hợp như:
  - **Lines** và **Stripes**: SIFT có Recall cao hơn (~0.33, ~0.27), còn ORB chỉ đạt khoảng (~0.29, ~0.07).
- **Loại hình Multiple**: Cả hai phương pháp đều có kết quả thấp, nhưng ORB vẫn nhỉnh hơn.

#### 2. **Precision của SIFT và ORB**
- **ORB vượt trội hơn SIFT** trong nhiều trường hợp, đặc biệt:
  - **Polygon**: ORB đạt Precision cao nhất (~0.44), trong khi SIFT thấp hơn đáng kể.
  - **Star**: SIFT cũng vượt trội với Precision cao (~0.42), còn ORB chỉ đạt mức trung bình.
  - **Multiple**: SIFT có kết quả tốt hơn ORB nhưng không quá vượt trội.
- **SIFT có kết quả tốt hơn hoặc gần tương đương với ORB** ở:
  - **Lines**, **Star**, **Stripes**: khoảng (0.36-0.38 và ~0.20) so với (0.24-0.28 và ~0.05). 
  - **Cube**: Precision của cả hai gần bằng nhau (~0.18-0.2).            """)
# st.write(df)

st.header('5. Thảo luận', divider=True)

st.markdown("""
                - Trên tập hình ***checkerboard, polygon, multiple_polygon*** các keypoints thường tập trung ở các góc của hình. ORB phát hiện keypoints dựa trên thuật toán FAST bằng cách xem xét độ sáng điểm ảnh xung quanh một khu vực nhất định, nên nó thường nhận diện tốt các điểm đặc trưng trên những dạng hình này.
                - Dưới đây là một số hình minh họa kết quả của 2 thuật toán:
                    - <span style="color:green">Màu xanh</span>: những điểm Ground Truth.
                    - <span style="color:red">Màu đỏ</span>: những điểm keypoint mà thuật toán phát hiện được.
                """, unsafe_allow_html=True)

col2 = st.columns(2) 
col2[0].write(''' - Kết quả minh họa của ORB''')
col2[1].write(''' - Kết quả minh họa của SIFT''')

col3 = st.columns(2)
col3[0].image('./datasets/sythetic/results/polygon_1.png', caption='Kết quả của tập hình polygon với số keypoints được phát hiện = 4, precision = 0.5, recall = 0.66', use_column_width=True)
col3[1].image('./datasets/sythetic/results/polygon_2.png', caption='Kết quả của tập hình polygon với số keypoints được phát hiện = 7, precision = 0., recall = 0.', use_column_width=True)

col3[0].image('./datasets/sythetic/results/check_1.png', caption='Kết quả của tập hình checkerboard với số keypoints được phát hiện = 34, precision = 0.1176, recall = 0.25', use_column_width=True)
col3[1].image('./datasets/sythetic/results/check_2.png', caption='Kết quả của tập hình checkerboard với số keypoints được phát hiện = 13, precision = 0.0769, recall = 0.0625', use_column_width=True)

col3[0].image('./datasets/sythetic/results/line_1.png', caption='Kết quả của tập hình lines với số keypoints được phát hiện = 24, precision = 0.0833, recall = 0.25', use_column_width=True)
col3[1].image('./datasets/sythetic/results/line_2.png', caption='Kết quả của tập hình lines với số keypoints được phát hiện = 7, precision = 0.7142, recall = 0.625', use_column_width=True)

# st.write("""
#         - Giải thích cho lí do ***Recall*** của ORB cao hơn trên tập hình star, Cube, checkerboard là vì:
#             - ORB phát hiện được số lượng keypoints nhiều hơn so với SIFT trên tập hình này, dẫn đến việc có nhiều điểm dự đoán đúng hơn (true positives), làm tăng Recall. Tuy nhiên, do phát hiện nhiều điểm hơn, ORB cũng tạo ra nhiều điểm nhiễu (false positives) hơn, đây cũng là lí do khiến Precision giảm xuống so với SIFT trong hầu hết các tập hình.
#          """)

# col3 = st.columns(2)
# col3[0].image('./datasets/sythetic/results/st_1.png', caption='Kết quả của tập hình star với số keypoints được phát hiện = 15, precision = 0.2, recall = 0.75', use_column_width=True)
# col3[1].image('./datasets/sythetic/results/st_2.png', caption='Kết quả của tập hình star với số keypoints được phát hiện = 2, precision = 1., recall = 0.5', use_column_width=True)

# col3[0].image('./datasets/sythetic/results/st_3.png', caption='Kết quả của tập hình Cube với số keypoints được phát hiện = 8, precision = 0.5, recall = 0.57', use_column_width=True)
# col3[1].image('./datasets/sythetic/results/st_4.png', caption='Kết quả của tập hình Cube với số keypoints được phát hiện = 2, precision = 0., recall = 0.', use_column_width=True)


# col3[0].image('./datasets/sythetic/results/st_5.png', caption='Kết quả của tập hình checkerboard với số keypoints được phát hiện = 97, precision = 0.15, recall = 0.46', use_column_width=True)
# col3[1].image('./datasets/sythetic/results/st_6.png', caption='Kết quả của tập hình checkerboard với số keypoints được phát hiện = 50, precision = 0.2, recall = 0.31', use_column_width=True)






import streamlit as st
import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import pickle
import altair as alt
import pandas as pd

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
st.markdown("""
            ***SuperPoint*** (Self-Supervised Interest Point Detection and Description) là một phương pháp tiên tiến dùng để phát hiện và mô tả điểm đặc trưng (keypoints) trong ảnh. Được phát triển với sự kết hợp của học sâu (deep learning), SuperPoint là một mô hình mạng nơ-ron có khả năng học các điểm đặc trưng một cách tự động và tự giám sát, từ đó tạo ra mô tả đặc trưng mạnh mẽ và nhất quán cho các nhiệm vụ thị giác máy tính được giới thiệu lần đầu trong bài báo [SuperPoint: Self-Supervised Interest Point Detection and Description." Daniel DeTone, Tomasz Malisiewicz, Andrew Rabinovich.](https://arxiv.org/abs/1712.07629).
            """)
st.image('./datasets/sythetic/keypoints_matching/kientruc_superpoint.png', caption='Kiến trúc SuperPoint', use_column_width=True)
st.write("""
         Kiến trúc của SuperPoint gồm 2 phần chính: 
            - **Detector Head**: Phần này chịu trách nhiệm phát hiện các keypoints trong ảnh. Nó được huấn luyện để nhận diện các điểm có ý nghĩa trong ảnh như các góc, biên, hoặc các vùng có sự thay đổi cường độ lớn.
            - **Descriptor Head**: Phần này trích xuất các vector đặc trưng (feature descriptors) từ các điểm được phát hiện. Các vector này đại diện cho vùng xung quanh mỗi keypoint.
         """)

st.header('3. Evaluation')
st.write(   """
              - Để đánh giá hiệu suất của SuperPoint, SIFT, ORB trên tập dữ liệu Synthetic Shapes Dataset dưới ảnh hưởng của phép quay, thực hiện theo các bước sau: 
                1. Trích xuất đặc trưng:
                    - Trích xuất vector đặc trưng sử dụng SIFT, ORB và SuperPoint tại các keypoint ground truth ở từng góc quay.
                2. So khớp keypoint: 
                    - Sử dụng Brute-Force Matching để so khớp các vector đặc trưng đã trích xuất của các keypoint ground truth giữa ảnh gốc và các ảnh đã quay.
                    
              - Sử dụng độ đo để tính phần trăm của các keypoint được so khớp chính xác cho mỗi phương pháp ở mỗi góc quay. 
            
            """
         )
st.image('datasets/sythetic/keypoints_matching/evaluation_matching.png', caption='Quy trình đánh giá', use_column_width=True)

st.header('4. Results')

def visualize_result():
  df = []
  with open('./datasets/sythetic/result_matching.pkl', 'rb') as f:
    df = pickle.load(f)
  df = pd.DataFrame(df)
  print (df)    
  ch = alt.Chart(df).mark_line().encode(
        alt.X('Angle', title='Angle'),
        alt.Y('Score', title='Accuracy'),
        color='Method:N'  # 
    ).properties(
        title='Biểu đồ sự thay đổi của Accuracy theo Góc quay'
    )
  st.altair_chart(ch, use_container_width=True)
  st.write("""
           - Từ biểu đồ trên, ta thấy rằng:
                - ORB đạt độ chính xác cao nhất ở mọi góc quay. Ban đầu, với góc 0-15 độ, độ chính xác luôn giữ ở mức > 0.8. Khi góc tăng lên, độ chính xác giảm nhưng vẫn duy trì trên mức 0.4, cho thấy khả năng kháng xoay tốt.

                - SIFT có hiệu suất khá nhưng thấp hơn ORB. Ở góc nhỏ (0-10 độ), độ chính xác cũng cao (~0.8), nhưng giảm nhanh hơn khi góc quay tăng. Điều này cho thấy SIFT ổn định ở góc nhỏ nhưng nhạy cảm hơn với góc quay lớn.

                - SuperPoint có hiệu năng kém nhất khi thực hiện các phép quay. Độ chính xác giảm rất nhanh, gần như bằng 0 ở 45 độ. Điều này chỉ ra thuật toán này không phù hợp với các bài toán biến đổi xoay lớn.
           """)
visualize_result()

with open('./datasets/sythetic/keypoints_matching/result_exper.pkl', 'rb') as f:
  de = pickle.load(f)
  
col = st.columns([1, 3, 3, 3])
for _, i in enumerate(["SIFT", "ORB", "SuperPoint"]):
  col[_ + 1].write(f'***{i}***')
for j in ["Checkerboard", "Cube", "Lines"]:
  # st.write()
  col = st.columns([1, 3, 3, 3])
  col[0].write(f'***{j}***')
  for angle in [10, 20, 30]:
    for idx, i in enumerate(de):
      if i['type'] == j and i['Angle'] == angle and i['Method'] == 'SIFT':
        col[1].image(i["img"], caption=f'Góc quay {angle}, số lượng keypoints khớp chính xác {i["correct"]}/{i["total"]}', use_column_width=True)
      if i['type'] == j and i['Angle'] == angle and i['Method'] == 'ORB':
        col[2].image(i["img"], caption=f'Góc quay {angle}, số lượng keypoints khớp chính xác {i["correct"]}/{i["total"]}', use_column_width=True)
      if i['type'] == j and i['Angle'] == angle and i['Method'] == 'SuperPoint':
        col[3].image(i["img"], caption=f'Góc quay {angle}, số lượng keypoints khớp chính xác {i["correct"]}/{i["total"]}', use_column_width=True)      


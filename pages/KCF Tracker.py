import streamlit as st
import cv2 as cv
import matplotlib.pyplot as plt
from my_utils.cbri import CBRI
import numpy as np
from PIL import Image

st.title('Theo dõi đối tượng bằng KCF Tracker')
st.header('1. Giới thiệu')
st.write("""
        - [KCF (Kernelized Correlation Filters) được giới thiệu bởi João F. Henriques, Rui Caseiro, Pedro Martins và Jorge Batista vào năm 2015](https://arxiv.org/pdf/1404.7584) là một thuật toán theo dõi đối tượng nhanh và chính xác, đặc biệt hiệu quả trong việc theo dõi các đối tượng có chuyển động nhanh hoặc bị che khuất một phần. KCF được phát triển dựa trên kỹ thuật sử dụng bộ lọc tương quan (Correlation Filter) với không gian đặc trưng phi tuyến tính (kernelized feature space). 
        """)

st.header("2. Phương pháp")
st.image('./images/kcf.png', use_column_width=True, caption='Pipeline thuật toán KCF')
st.write("""
        - Nguyên lý hoạt động của KCF chủ yếu dựa vào:
            - Sử dụng bộ lọc tương quan (Correlation Filter):
                - KCF dựa vào việc học một bộ lọc tối ưu trong không gian Fourier, giúp tăng tốc độ tính toán khi thực hiện phép tương quan (correlation).
            - Tích hợp kernel:
                - KCF sử dụng kernel (như: Gaussian hoặc Polynomial) để mở rộng khả năng biểu diễn của bộ lọc tương quan, làm cho mô hình có thể nắm bắt các mối quan hệ phức tạp giữa các đặc trưng.
            - Huấn luyện và theo dõi:
                - Trong quá trình huấn luyện, KCF xây dựng một bộ lọc bằng cách tối ưu hóa hàm mất mát dựa trên dữ liệu của đối tượng cần theo dõi.
                - Khi theo dõi, bộ lọc được sử dụng để tìm vị trí đối tượng mới nhất trong khung hình kế tiếp bằng cách tìm điểm tương quan cao nhất.
         """)

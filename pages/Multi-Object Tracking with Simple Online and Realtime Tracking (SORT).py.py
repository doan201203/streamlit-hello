import streamlit as st
import cv2 as cv
import matplotlib.pyplot as plt
import os
import requests
from PIL import Image
import numpy as np
import altair as alt
import pandas as pd


st.set_page_config(page_title="Multi-Object Tracking with SORT (Simple Online and Realtime Tracking)")
st.header('1. Giới thiệu')
st.write("""
        SORT (Simple Online and Realtime Tracking) là một thuật toán theo dõi đối tượng trực tuyến (online tracking) tối ưu để đạt hiệu suất cao cả về tốc độ và độ chính xác, phù hợp với các ứng dụng thời gian thực được giới thiệu vào năm 2016 trong bài báo cáo [Simple Online and Realtime Tracking](https://arxiv.org/pdf/1602.00763). Mục tiêu chính của SORT bao gồm:

    - ***Theo dõi thời gian thực***: Đạt tốc độ xử lý cao.
    - ***Sự đơn giản***: Sử dụng các kỹ thuật cơ bản nhưng hiệu quả như Kalman Filter và Hungarian Algorithm.
         """)
st.header('2. Phương pháp')
st.write("""
         Gồm 3 phần chính như sau:
            - ***Detection***.
            - ***Estimation Model***.
            - ***Data Association***.
         """)
st.image('./images/SORT/pipline.png', caption='Cấu tạo và nguyên lý hoạt động của SORT', channels='BGR')
st.subheader('2.1 Detection')
st.write("""
        - Để xác định vị trí của đối tượng qua từng khung hình ***SORT*** cần sử dụng một mô hình detect đủ nhanh nhưng vẫn đảm bảo được độ chính xác như [***Faster R-CNN***](https://arxiv.org/abs/1506.01497), [***YOLO***](https://arxiv.org/abs/1506.02640), ...
         """)
st.subheader("2.2 Estimation Model")
st.write("""
        - Một trong những nhiệm vụ quan trọng nhất của Object Tracking là dự đoán vị trí của đối tượng.
        - ***SORT*** sử dụng ***Kalman Filter*** để ước tính vị trí mới của một vật thể bằng cách ngoại suy chuyển động của vật thể.
         """)
st.image('./images/SORT/kalm_filter.png', channels='BGR', use_column_width=True)
st.write("""
        - Uớc tính vị trị gồm 2 bước chính:
            - ***Dự đoán (prediction)***: Tính toán trạng thái dự kiến của mục tiêu dựa trên trạng thái đã được ước lượng trước đó, cùng với mô hình vận tốc tuyến tính và các thông tin sẵn có.
            - ***Hiệu chỉnh (correction)***: Kết hợp trạng thái dự đoán với thông tin thực tế đo được (bounding box từ detector) để tính toán trạng thái chính xác hơn.
         """)
st.markdown("""
Với mỗi trạng thái của đối tượng $\mathbf{x} = [u, v, s, r, \dot{u}, \dot{v}, \dot{s}]^T$ được định nghĩa như sau:

- $(u, v)$: Tọa độ tâm $(x, y)$ bounding box của đối tượng.  
- $s$: Diện tích của bounding box.  
- $r$: Tỷ lệ khung hình của bounding box (giả định không thay đổi).  
- $ \dot{u}, \dot{v}, \dot{s} $: Tốc độ thay đổi của các tham số tương ứng.  

**Ý nghĩa:**  
- Nếu bounding box từ **detection** được liên kết với mục tiêu đang theo dõi, bounding box này sẽ được sử dụng để cập nhật trạng thái của mục tiêu bằng **Kalman Filter**.  
- Nếu mục tiêu không được liên kết với detection (unmatched), trạng thái của mục tiêu sẽ chỉ được dự đoán không có bước ***hiệu chỉnh***.
            """, unsafe_allow_html=True)
st.subheader("2.3 Data Association")
st.image('./images/SORT/hungarian.png', caption='Áp dụng thuật toán Hungarian để ghép nối các trạng thái [Nguồn](https://www.linkedin.com/pulse/object-tracking-sort-deepsort-daniel-pleus)')
st.write("""        
Để liên kết các đối tượng được phát hiện trong frame hiện tại với các đối tượng đang được theo dõi, đầu tiên ***SORT*** thực hiện tính toán ***Assignment Cost Matrix (IOU Matrix)***.  

- **IoU Distance** giữa bounding box được phát hiện và bounding box dự đoán của các đối tượng đang theo dõi được sử dụng làm cơ sở cho quá trình này.  
- Việc gán assignment cuối cùng được thực hiện bằng **Hungarian Algorithm**.  
- Ngoài ra, nếu **IoU score** được tính thấp hơn ngưỡng **IoUₘᵢₙ**, đối tượng sẽ bị loại bỏ.  
         """)
st.image('./images/SORT/iou.png', caption='Minh họa quá trình liên kết các đối tượng', use_column_width=True)
st.write("""
         Trong đó:
            - (a): Kết quả từ Kalman Filter trước đó.  
            - (b): Các bounding box màu xanh lá từ detector hiện tại và các bounding box dự đoán (màu đỏ). IoU được tính giữa chúng.  
            - (c): Hungarian Algorithm ánh xạ các box, gán ID phù hợp với từng đối tượng.  
         """)

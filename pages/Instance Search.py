import streamlit as st
import cv2 as cv
import matplotlib.pyplot as plt
from my_utils.cbri import CBRI
import numpy as np
from PIL import Image

st.title('Instance Search')

st.header('1. Dataset')
st.write("""
            - Trích từ bộ dữ liệu COCO-2017 với đối tượng ***Vehicle*** gồm 1160 ảnh phương tiện được chia thành 8 class ["bicycle", "car", "motorcycle","airplane", "bus", "train", "truck", "boat"].
            - Một số hình ảnh trong tập dữ liệu.
         """)

@st.cache_resource(show_spinner=False)
def get_db_image():
    cbri = CBRI()
    return cbri

cbri = get_db_image()
@st.fragment
def display_db_image():
    col = st.columns(4)
    for i in range(4):
        col[i].image(cbri.db[i], caption='Image {}'.format(i))
    col = st.columns(4)
    for i in range(8, 12):
        col[i-8].image(cbri.db[i], caption='Image {}'.format(i))
display_db_image()

st.header('2. Methods')
def display_methods():
   st.markdown("""
                - Extract features:
                    - Sử dụng mô hình SuperPoint để trích xuất đặc trưng của ảnh.
                - Cluster features:
                    - Xây dựng bag of visual word (BOVW) mục tiêu là biến đổi các đặc trưng của hình ảnh thành một tập hợp các từ đại diện (visual words), sau đó tạo thành một biểu diễn histogram.
            """)
   st.image('./images/bovw.png', caption='Bag of Visual Word')
   st.markdown("""
             - Compare:
               - Sử dụng cosine similarity để so sánh vector đặc trưng của ảnh.
               - Công thức tính cosine similarity giữa 2 vector a và b:
               - $cosine = \\frac{a.b}{||a||.||b||}$
            """)
display_methods()
st.header('3. Evaluation')
st.header('4. Results')
st.header('5. Application')

@st.fragment
def display_search():
   st.write("Tìm kiếm hình ảnh tương tự")
   
   fil = st.file_uploader('Upload an image', type=['jpg', 'png', 'jpeg'])
   if fil is not None:
      with st.form(key='my_form'):
         K = st.slider("Chọn số lượng ảnh tìm kiếm", 1, 10, 5)
         submit_button = st.form_submit_button(label='Submit')
         img = Image.open(fil)
         st.image(img, caption='Uploaded Image')
         if submit_button:
            img = np.array(img)
            max_width = 640
            
            if img.shape[1] > max_width:
               scale = max_width / img.shape[1]
               img = cv.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
            
            idx, conf = cbri.top_k(img, K)

            # col = st.columns(K)
            for i in range(K):
               st.image(cbri.db[idx[i]], caption='Cosine Similarity {}'.format(conf[i]), channels='BGR')
display_search()

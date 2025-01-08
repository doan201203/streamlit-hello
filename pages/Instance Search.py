import streamlit as st
import cv2 as cv
import matplotlib.pyplot as plt
from my_utils.cbri import CBRI
import numpy as np
from PIL import Image

st.title('Instance Search')

st.header('1. Dataset')
st.write("""
            - Trích từ bộ dữ liệu COCO-2017 gồm 1160 ảnh được chia thành 8 class ["bicycle", "car", "motorcycle","airplane", "bus", "train", "truck", "boat"].
            - Một số hình ảnh trong tập dữ liệu.
         """)


@st.cache_resource(show_spinner=False, ttl="10m")
def get_db_image():
    cbri = CBRI()
    return cbri

cbri = get_db_image()
@st.fragment
def display_db_image():
    col = st.columns(4)
    for i in range(0, 4):
        col[i].image(cbri.db[i], caption='Image {}'.format(i), channels='BGR')
    col = st.columns(4)
    for i in range(120, 124):
        col[i-120].image(cbri.db[i - 120 + 40], caption='Image {}'.format(i), channels='BGR')
display_db_image()

st.header('2. Methods')
def display_methods():
   st.markdown("""
                - Extract features:
                    - Sử dụng mô hình SuperPoint để trích xuất đặc trưng của ảnh.
                - Cluster features:
                    - Xây dựng bag of visual word (BOVW) mục tiêu là biến đổi các đặc trưng của hình ảnh thành một tập hợp các từ đại diện (visual words), sau đó tạo thành một biểu diễn histogram.
            """)
   
   st.markdown("""
             - Compare:
               - Sử dụng cosine similarity để so sánh vector đặc trưng của ảnh.
               - Công thức tính cosine similarity giữa 2 vector a và b:
               - $cosine = \\frac{a.b}{||a||.||b||}$
            """)
   st.image('./images/bovw.png', caption='Pipeline quá trình tìm kiếm')
   st.write("""
            - Các bước thực hiện:
                - ***(1)***: Trích xuất các desciptors từ tập dữ liệu.
                - ***(2)***: Sử dụng K-means để phân cụm các desciptors với K = 512.
                - ***(3)***: Sau khi phân cụm các descriptors bằng K-means, ta sẽ thu được 512 cụm (centroids). Các cụm này chính là từ điển (visual vocabulary), đại diện cho các đặc trưng của toàn bộ dataset.
                - ***(4)***: Với mỗi hình ảnh trong dataset, ta gán các descriptors của hình ảnh đó vào cụm gần nhất (dựa trên khoảng cách với centroids). Từ đó, tính histogram vector cho hình ảnh.
                - ***(5)***: Áp dụng TF-IDF để tăng cường thông tin đặc trưng
                - ***(6)***: Trích xuất descriptors từ Query Image
                - ***(7)***: Sử dụng visual vocabulary từ bước ***(3)***, gán descriptors của query image vào các cụm gần nhất để tạo histogram vector.
                - ***(8)***: Áp dụng TF-IDF cho Query Image.
                - ***(9)***: Sử dụng ***Cosine Similarity*** để so sánh độ tương đồng giữa histogram vector của query image và từng histogram vector trong dataset.
            """)
   #Giai thich li do chon K = 512 dua vao eblow method
   st.write("""- Giải thích lý do chọn K = 512 dựa vào phương pháp [Elbow](https://www.analyticsvidhya.com/blog/2021/01/in-depth-intuition-of-k-means-clustering-algorithm-in-machine-learning/)""")
   st.image('./images/eblow_bovw.png', caption='Elbow Method', use_column_width=True)
display_methods()
# st.header('3. Evaluation')
# st.header('4. Results')
st.header('3. Application')

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
            idx, conf = cbri.top_k(img, K)
            for i in range(K):
               st.image(cbri.db[idx[i]], caption='Cosine Similarity {}'.format(conf[i]), channels='BGR')
display_search()

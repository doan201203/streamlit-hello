import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow import keras
from my_utils.free_draw_2_mask import Free2Mask
import cv2
from skimage import measure
import numpy as np

st.set_page_config(layout="wide")
@st.cache_resource(show_spinner=False)
def load_model():
    model = keras.models.load_model('./miscs/hand_writing/mnist_model.h5')
    return model

def p1():
   st.header("1. Dataset")
   st.write("""
            -   Sử dụng tập dữ liệu MNIST Hình ảnh chứa 70.000 hình ảnh chữ số viết tay từ 0 đến 9.
            -   Mỗi hình ảnh có kích thước 28x28 pixel. 
            """)
   st.image('./miscs/hand_writing/example.png', use_column_width=True, caption='Một số hình ảnh trong tập dữ liệu')

def p2():
   st.header("2. Phương Pháp")
   st.image('./images/kien_truc_hand_writing.png', use_column_width=True, caption='Kiến trúc mạng sử dụng trong bài toán')
   st.markdown("""
        - Tập dữ liệu được chia thành 60.000 hình ảnh cho quá trình training và 10.000 quá trình testing.
        - Số lượng epoch: 10
        - Batch size: 128
    """)
   st.image('./miscs/hand_writing/results.png', use_column_width=True, caption='Biểu đồ accuracy, loss sau khi training')
def predict_with_mask(model, mask):
    
    mask = cv2.resize(mask, (28, 28))
    # st.image(mask, use_column_width=True)
    mask = mask / 255.0
    mask = mask.reshape(1, 28, 28, 1)
    return model.predict(mask)

def preprocess_mask(model, mask):
    #noise removal with erosion
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=3)
    # dilate to fill the gap
    mask = cv2.dilate(mask, kernel, iterations=3)
    # st.image(mask, use_column_width=True)
    
    labels = measure.label(mask, background=0)
    # print(np.unique(labels))
    regions = measure.regionprops(labels)
    
    #sort by y coordinate after x coordinate
    for rg in regions:
        if rg.bbox[2] - rg.bbox[0] <= 0.15 * mask.shape[0]:
            mask[rg.bbox[0]:rg.bbox[2], rg.bbox[1]:rg.bbox[3]] = 0
    
    regions.sort(key=lambda x: x.bbox[1])
    
    #remove region that is too small to be a digit if height < 10% of the image height
    regions = [rg for rg in regions if rg.bbox[2] - rg.bbox[0] > mask.shape[0] * 0.15]
    print([rg.bbox[2] - rg.bbox[0] for rg in regions], mask.shape[0] * 0.15)
    # print(regions[0].bbox)
    # return regions[0].bbox
    a = []
    for rg in regions:
        minr, minc, maxr, maxc = rg.bbox
        crop = mask[minr:maxr, minc:maxc]
        # crop = mask.crop((minc, minr, maxc, maxr))
        # st.image(crop, use_column_width=True)
        #add padding = height * 0.25
        pad = int(crop.shape[0] * 0.25)
        crop = np.pad(crop, ((pad, pad), (pad, pad)), 'constant', constant_values=(0, 0))
        
        a.append(np.argmax(predict_with_mask(model, crop)))
    return a

def p3():
   st.header("2. Ứng dụng")
   col = st.columns([6, 4])
   col[0].write("""
            -   Ứng dụng sử dụng mô hình có kết quả tốt nhất sau khi training 10 epoch với:
                - Accuracy: 0.987
            -   Huớng dẫn sử dụng:
                -   1. Vẽ các chữ số từ 0-9.
                -  2. Sau khi vẽ xong bấm nút Submit để tiến hành nhận diện.
            - ***Lưu ý:***
                - Để đảm bảo kết quả nhận diện đúng theo thứ tự vui lòng vẽ các chữ số trên cùng ***1 hàng duy nhất*** xem (hình 2).
                - Các chữ số không được chồng chéo lên nhau.
            """)
   col[1].image('./miscs/hand_writing/pipeline.png', use_column_width=True, caption='Hình 1. Các bước để nhận diện chữ số viết tay')
   col[1].image('./miscs/hand_writing/vidu1.png', use_column_width=True, caption='Hình 2. một số lưu ý khi sử dụng')
   with st.form(key='my_form'):
       col = st.columns([7, 3])
       with col[0]:
           canvas_result = st_canvas(
               fill_color="white",  # Fixed fill color with some opacity
               stroke_width=10,
               stroke_color="white",
               background_color="black",
               width=800,
               height=100,
               drawing_mode="freedraw",
               key="canvas",
            )
           submit_button = st.form_submit_button(label='Submit')
           if submit_button:
               model = load_model()
               
               draw = Free2Mask(canvas_result.image_data)
               if canvas_result.json_data is not None:
                   record = canvas_result.json_data['objects']
                   for obj in record:
                       if obj['type'] == 'path':
                            thickness = obj['strokeWidth']
                            draw.path(obj['path'], 255, thickness//2)
               with col[1]:
                   st.write("Kết quả dự đoán:")
                   st.write(*preprocess_mask(model, draw.get_mask()))
            #    print(preprocess_mask(model, draw.get_mask()))
    #    with col[1]:
        #    st.image()
            
              
p1()
p2()
p3()
